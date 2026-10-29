#ifndef GUFO_MODELS_QWEN_HIP_OPS_MOE_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_MOE_HPP_

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Flint sigmoid router selection (F-E1): reads `logits[expert_count]`,
/// applies sigmoid, selects the top `used` experts (ties to the lowest
/// index, matching the CPU reference and the oracle), renormalises the
/// selected probabilities when `norm`, scales by `scale`, and writes
/// `selected[used]` + `weights[used]`. Single block, deterministic order.
void LaunchFlintRouterTopK(const float* logits, std::size_t expert_count,
                           std::uint32_t used, bool norm, float scale,
                           std::uint32_t* selected, float* weights,
                           hipStream_t stream = nullptr);

/// Accumulates one expert output: `out[i] += weight * expert_out[i]`
/// (F-E1/F-E3; the shared expert enters with weight 1.0).
void LaunchFlintExpertAccumulate(float* out, const float* expert_out,
                                 float weight, std::size_t n,
                                 hipStream_t stream = nullptr);

/// Zeroes `out[0..n)` before the expert accumulation loop.
void LaunchFlintZero(float* out, std::size_t n, hipStream_t stream = nullptr);

/// Capacity of the fused-dispatch job list: the top-`used` selection plus
/// the shared expert, with headroom. used <= kMaxSelected (8), so 32 covers
/// every reachable configuration.
inline constexpr std::size_t kFlintMoeMaxJobs = 32;

/// Fused-dispatch router (F-J1): same selection as LaunchFlintRouterTopK,
/// but instead of the size-`used` readback pair it writes the full
/// `weights_full[expert_count + 1]` — renormalised, scaled, zero for
/// non-selected experts, and the ungated shared expert (unit weight) at
/// slot `expert_count` — so the expert kernels can self-gate on device.
/// When `job_list` is non-null it also compacts the selected ids (top-k
/// order, shared expert last) into `job_list[0..used]`, padding the rest
/// of the `kFlintMoeMaxJobs` slots with -1; the fused kernels then index
/// experts through it so their grids span only real jobs instead of the
/// whole expert stack (block dispatch of ~257x-slice grids dominated the
/// step time). Also zeroes `zero_out[0..zero_n)` (the mlp accumulator) in
/// the same launch.
void LaunchFlintRouterTopKFull(const float* logits, std::size_t expert_count,
                               std::uint32_t used, bool norm, float scale,
                               float* weights_full, float shared_weight,
                               float* zero_out, std::size_t zero_n,
                               int* job_list, hipStream_t stream = nullptr);

/// Fused MoE branch, phase 1 (F-J1): for every job in `job_list` (or every
/// expert when `job_list` is null — synthetic tests), computes
/// act[e][r] = silu(gate_e[r]·x) * (up_e[r]·x) over expert_ff rows
/// (shared_ff for the shared expert) from the quantised gate/up stacks.
/// The grid spans `max_jobs` blocks when `job_list` is given, else
/// expert_count + 1; zero-weight jobs exit early. `type` decodes the
/// routed stacks; the shared matrices decode with their own types (UD
/// checkpoints quantise them independently, e.g. Q6_K routed / Q8_0 shared).
void LaunchFlintFusedMoESwiGLU(const void* gate_stack, const void* up_stack,
                               const void* gate_shared, const void* up_shared,
                               core::GgmlType gate_type, core::GgmlType up_type,
                               core::GgmlType shared_gate_type,
                               core::GgmlType shared_up_type, const float* x,
                               float* act, const float* weights_full,
                               const int* job_list, std::size_t max_jobs,
                               std::size_t expert_count, std::size_t expert_ff,
                               std::size_t shared_ff, std::size_t hidden,
                               std::size_t gate_slab_bytes,
                               std::size_t up_slab_bytes,
                               hipStream_t stream = nullptr);

/// Fused MoE branch, phase 2 (F-J1 + 3.6 determinism): for every job in
/// `job_list` (or every expert when `job_list` is null) the down kernel
/// writes weights_full[e] * (down_e · act[e]) for all hidden rows into the
/// DISJOINT partials[job_slot * hidden + row] slot — no float atomics — and
/// the reduce kernel then sums the slots in fixed job order. With
/// `residual_hidden` (3.3 epilogue fusion) the reduce adds the sum straight
/// into that vector and `mlp_out` is not written; with null it stores the
/// sum into `mlp_out` (test/bench path). `partials` needs
/// max_jobs * hidden floats. `slab_bytes` must be the encoded per-expert
/// slab of the DOWN tensor's own type; the shared matrix decodes with
/// `shared_type`.
void LaunchFlintFusedMoEDown(const void* down_stack,
                             const void* down_shared, core::GgmlType type,
                             core::GgmlType shared_type, const float* act,
                             float* partials, float* mlp_out,
                             float* residual_hidden,
                             const float* weights_full,
                             const int* job_list, std::size_t max_jobs,
                             std::size_t expert_count, std::size_t expert_ff,
                             std::size_t shared_ff, std::size_t hidden,
                             std::size_t slab_bytes,
                             hipStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// Batched chunk-prefill MoE (F-L4). The chunk path replaces the per-token
// decode dispatch: one batched router GEMM, one top-k block per row, one
// device-side slot compaction, and two expert-grouped kernels whose blocks
// reuse each decoded weight block across a whole expert bucket.
// ---------------------------------------------------------------------------

/// logits[t, e] = x[t] . W_router[e] for the whole chunk. `type` must be
/// kF32, kF16 or kBF16 (the router tensor's storage type). Uses the tiled
/// LDS kernel when hidden % 64 == 0 (GUFO_FLINT_ROUTER_SCALAR=1 forces the
/// scalar fallback for A/B measurements).
void LaunchFlintRouterBatch(const float* x, std::size_t n_tokens,
                            const void* w, core::GgmlType type,
                            std::size_t expert_count, std::size_t hidden,
                            float* logits, hipStream_t stream = nullptr);

/// Sigmoid top-k for every chunk row: selected[t, k] + weights[t, k],
/// same selection rule (ties to the lowest index), normalisation and scale
/// as the decode router.
void LaunchFlintRouterTopKBatch(const float* logits, std::size_t n_tokens,
                                std::size_t expert_count, std::uint32_t used,
                                bool norm, float scale,
                                std::uint32_t* selected, float* weights,
                                hipStream_t stream = nullptr);

/// Compacts the chunk selection into expert-sorted slots. Writes
/// counts[E], offsets[E+1], slot_tokens[slots] (token index per slot),
/// slot_weights[slots] (router weight per slot) and token_slots[T*used]
/// (slot index per selection, selection order preserved). `cursors` is
/// scratch; counts/offsets/cursors are E, E+1 and E ints.
///
/// Bucket starts are padded to the 16-row WMMA token tile: offsets[E+1] is
/// the PADDED total, and the tile table (tile_expert/tile_slot0, count in
/// tile_count) maps each 16-row tile to its expert and first slot for the
/// grouped WMMA kernels. The slot tables must hold slots + 15 * expert_count
/// rows (see kFlintPrefillSlotCapacity).
void LaunchFlintMoeSlots(const std::uint32_t* selected,
                         const float* sel_weights, std::size_t n_tokens,
                         std::size_t used, std::size_t expert_count,
                         int* counts, int* offsets, int* cursors,
                         int* slot_tokens, float* slot_weights,
                         int* token_slots, int* tile_expert, int* tile_slot0,
                         int* tile_count, hipStream_t stream = nullptr);

/// Expert-grouped SwiGLU over the slot buckets:
/// act[slot, r] = silu(gate_e[r] . x(token(slot)]) * (up_e[r] . x[token(slot)]).
/// `offsets`/`slot_tokens` come from LaunchFlintMoeSlots; `slab_bytes`/
/// `slab_up` are the encoded per-expert slabs of the two stacks (types may
/// differ per tensor — bartowski IQ4_XS blk.2 has gate IQ4_XS, up Q5_K).
void LaunchFlintMoeBatchSwiGLU(const void* gate_stack, const void* up_stack,
                               core::GgmlType type, core::GgmlType up_type,
                               const float* x, float* act,
                               const int* offsets, const int* slot_tokens,
                               std::size_t expert_count, std::size_t expert_ff,
                               std::size_t hidden, std::size_t slab_bytes,
                               std::size_t slab_up,
                               hipStream_t stream = nullptr);

/// Expert-grouped down projection over the slot buckets, writing compact
/// slot rows already scaled by the slot's router weight:
/// down_out[slot, r] = slot_weights[slot] * (down_e[r] . act[slot]).
void LaunchFlintMoeBatchDown(const void* down_stack, core::GgmlType type,
                             const float* act, float* down_out,
                             const int* offsets, const float* slot_weights,
                             std::size_t expert_count, std::size_t expert_ff,
                             std::size_t hidden, std::size_t slab_bytes,
                             hipStream_t stream = nullptr);

/// Deterministic combine: mlp[t] = shared[t] + sum_j down_out[slot(t, j)]
/// in selection order.
void LaunchFlintMoeCombine(const float* shared_out, const float* down_out,
                           const int* token_slots, float* mlp_out,
                           std::size_t n_tokens, std::size_t used,
                           std::size_t hidden, hipStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// Grouped WMMA route (F-L4 tuning). The expert buckets run on the same int8
// tensor-core arithmetic as the projections: activations arrive gathered into
// the tiled Q8_1 layout in slot order, and the two grouped kernels launch a
// grid over (tile, weight rows) with one expert resolved per block.
// ---------------------------------------------------------------------------

/// Copies the tiled Q8_1 rows of `x_q8` (token-major, [n_tokens, hidden])
/// into slot order in `slot_q8` ([slot_bound, hidden]). `slot_bound` is the
/// host-side padded-slot upper bound.
void LaunchFlintGatherQ8Act(const void* x_q8, void* slot_q8,
                            const int* slot_tokens, const int* offsets,
                            std::size_t expert_count, std::size_t slot_bound,
                            std::size_t hidden, hipStream_t stream = nullptr);

/// True when the grouped WMMA kernels cover this layer's routed types and
/// shapes (gate/up in a no-offset WK format with gate == up, down in
/// Q8_0/Q6_K/IQ4_XS, 128-divisible m). Layers failing this — e.g. a Q5_K
/// (offset-format) gate/up stack — take the scalar grouped pair.
[[nodiscard]] bool FlintGroupedWmmaSupported(core::GgmlType gate_up_type,
                                             core::GgmlType up_type,
                                             core::GgmlType down_type,
                                             std::size_t expert_ff,
                                             std::size_t hidden) noexcept;

/// gate_out[slot, r] = gate_e[r] . x[slot] and up likewise for every tile,
/// with x read from the slot-major tiled Q8_1 buffer and the expert resolved
/// per block through the tile table. Outputs are f32 rows in slot order.
void LaunchFlintMoeGroupedGateUpWmma(const void* gate_stack,
                                     const void* up_stack, core::GgmlType type,
                                     const void* slot_q8, float* gate_out,
                                     float* up_out, const int* offsets,
                                     const int* tile_expert,
                                     const int* tile_slot0,
                                     const int* tile_count,
                                     std::size_t tile_bound,
                                     std::size_t expert_count,
                                     std::size_t expert_ff, std::size_t hidden,
                                     std::size_t slab_bytes,
                                     hipStream_t stream = nullptr);

/// down_out[slot, r] = slot_weights[slot] * (down_e[r] . act[slot]) reading
/// act from its slot-major tiled Q8_1 buffer. `type` is the down stack's
/// weight format (Q8_0, Q6_K or IQ4_XS — see FlintGroupedWmmaSupported).
void LaunchFlintMoeGroupedDownWmma(const void* down_stack,
                                   core::GgmlType type, const void* act_q8,
                                   float* down_out, const int* offsets,
                                   const float* slot_weights,
                                   const int* tile_expert,
                                   const int* tile_slot0,
                                   const int* tile_count,
                                   std::size_t tile_bound,
                                   std::size_t expert_count,
                                   std::size_t expert_ff, std::size_t hidden,
                                   std::size_t slab_bytes,
                                   hipStream_t stream = nullptr);

/// Softmax over the full router logits, then top-k selection with lowest-index
/// tie-break and renormalized weights (sum floored at 2^-14). logits is
/// [n_tokens, logits_stride] with n_experts valid entries per row; ids and
/// weights are [n_tokens, n_used].
void LaunchMoeRouterTopK(const float* logits, std::uint32_t logits_stride,
                         std::int32_t* ids, float* weights,
                         std::uint32_t n_tokens, std::uint32_t n_experts,
                         std::uint32_t n_used, hipStream_t stream = nullptr);

/// Whether LaunchMoeRouterGateTopK covers these router weights and shape.
[[nodiscard]] bool IsMoeRouterGateTopKSupported(core::GgmlType router_type,
                                                core::GgmlType shared_type,
                                                std::uint32_t n_experts,
                                                std::size_t k,
                                                std::uint32_t n_used) noexcept;

/// One-token router in one launch: the router GEMV (n_experts x k), the
/// shared-expert gate GEMV (1 x k) and LaunchMoeRouterTopK, each with its
/// own kernel's arithmetic. `done` is a zeroed device counter, private to
/// the stream, that the kernel leaves at zero.
void LaunchMoeRouterGateTopK(const void* router, core::GgmlType router_type,
                             const void* shared_gate_inp,
                             core::GgmlType shared_type, const float* x,
                             float* logits, float* shared_gate,
                             std::int32_t* ids, float* weights,
                             std::uint32_t n_experts, std::size_t k,
                             std::uint32_t n_used, std::uint32_t* done,
                             hipStream_t stream = nullptr);

/// MoE combine: out[t] =sum_s(weights[t,s] * expert_out[(t*k+s), :]) +
/// sigmoid(shared_gate[t]) * shared_out[t]. All buffers fp32.
void LaunchMoeEpilogue(const float* expert_out, const float* weights,
                       const float* shared_out, const float* shared_gate,
                       float* out, std::uint32_t n_tokens, std::uint32_t n_used,
                       std::uint32_t dim, hipStream_t stream = nullptr);

/// Expert formats the MMQ gated vector kernel (qfn_mmq_moe_gated_vec) covers
/// for decode-sized row counts.
[[nodiscard]] constexpr bool IsMoeMmqGatedVecType(
    core::GgmlType type) noexcept {
  return type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ5_K ||
         type == core::GgmlType::kQ8_0;
}

/// Expert formats the MMQ routed vector kernel (qfn_mmq_moe_vec) covers.
[[nodiscard]] constexpr bool IsMoeMmqVecType(core::GgmlType type) noexcept {
  return type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ5_K ||
         type == core::GgmlType::kQ6_K || type == core::GgmlType::kQ5_1 ||
         type == core::GgmlType::kQ8_0;
}

/// Routed per-slot GEMV fallback for formats the MMQ MoE kernels do not cover
/// (BF16/F32/Q6_K and friends, decoded via DecodeQuantSub16). For each slot s
/// the expert index is ids[s]; out[s, :] = W_e * x_row. With x_per_slot the
/// input row is s itself (down projection reading per-slot activations);
/// otherwise it is s / x_token_divisor (gate/up sharing the token's row).
void LaunchMoeSlotGemv(const void* w, core::GgmlType type, const float* x,
                       const std::int32_t* ids, float* out, std::size_t m,
                       std::size_t k, std::uint32_t slots,
                       std::uint32_t x_token_divisor,
                       hipStream_t stream = nullptr);

/// Fused routed SwiGLU fallback: act[s, :] = SiLU(gate_e * x) * (up_e * x)
/// for the slot's expert e = ids[s]; x is the shared token row (batch one) or
/// row s / x_token_divisor.
void LaunchMoeSlotSwigluGemv(const void* gate_w, core::GgmlType gate_type,
                             const void* up_w, core::GgmlType up_type,
                             const float* x, const std::int32_t* ids,
                             float* act, std::size_t expert_ff,
                             std::size_t hidden, std::uint32_t slots,
                             std::uint32_t x_token_divisor,
                             hipStream_t stream = nullptr);

/// Scratch for the grouped BF16 expert path. Every buffer is device memory;
/// sorted_slots and tiles hold at least `slots` entries, expert_bounds at
/// least n_experts + 1.
struct MoeGroupedScratch {
  std::int32_t* sorted_slots;
  std::int32_t* tiles;
  std::int32_t* expert_bounds;
};

/// Whether LaunchMoeGroupedBf16Gemm supports an [m, k] expert matrix with
/// n_experts experts.
[[nodiscard]] bool IsMoeGroupedBf16GemmSupported(
    std::size_t m, std::size_t k, std::uint32_t n_experts) noexcept;

/// Groups the routed slots by expert for LaunchMoeGroupedBf16Gemm. Run once
/// per ids array; the grouping is valid for every projection that shares it.
void LaunchMoeGroupSlots(const std::int32_t* ids, std::uint32_t slots,
                         std::uint32_t n_experts,
                         const MoeGroupedScratch& scratch,
                         hipStream_t stream = nullptr);

/// Batched prefill expert projection for BF16 expert weights stacked
/// [experts, m, k]: out[s, :] = W_{ids[s]} * x_row(s), with the same row
/// selection as LaunchMoeSlotGemv. Each expert's slots share one pass over its
/// weights on the WMMA matrix cores; activations are rounded to BF16 and
/// accumulated in FP32. Requires LaunchMoeGroupSlots on the same ids and
/// n_experts first, and IsMoeGroupedBf16GemmSupported(m, k, n_experts).
void LaunchMoeGroupedBf16Gemm(const void* w, const float* x, float* out,
                              std::size_t m, std::size_t k, std::uint32_t slots,
                              std::uint32_t n_experts,
                              std::uint32_t x_token_divisor,
                              const MoeGroupedScratch& scratch,
                              hipStream_t stream = nullptr);

/// Dense prefill GEMM on the same WMMA kernel: out[b, :] = W * x[b, :] for a
/// BF16 [m, k] matrix, activations rounded to BF16 and accumulated in FP32.
/// Requires IsMoeGroupedBf16GemmSupported(m, k, 1). Unlike the exact BF16
/// prefill route it does not reproduce decode's FP32-activation arithmetic.
void LaunchBf16WmmaGemm(const void* w, const float* x, float* out,
                        std::size_t batch_size, std::size_t m, std::size_t k,
                        hipStream_t stream = nullptr);

/// LaunchBf16WmmaGemm on the tiled BF16 activation [k / 64][batch][64] that
/// LaunchBatchedFusedResidualRMSNormQuantizeQ8_1 writes (`out_bf16_tiled`):
/// bit-identical, and read as contiguous runs rather than 8 KiB-strided rows.
/// Same requirements as LaunchBf16WmmaGemm.
void LaunchBf16WmmaGemmTiled(const void* w, const void* x_tiled, float* out,
                             std::size_t batch_size, std::size_t m,
                             std::size_t k, hipStream_t stream = nullptr);

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_MOE_HPP_
