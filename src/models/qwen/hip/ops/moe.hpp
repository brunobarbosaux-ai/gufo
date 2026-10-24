#ifndef GUFO_MODELS_QWEN_HIP_OPS_MOE_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_MOE_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"

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
                               core::GgmlType type,
                               core::GgmlType shared_gate_type,
                               core::GgmlType shared_up_type, const float* x,
                               float* act, const float* weights_full,
                               const int* job_list, std::size_t max_jobs,
                               std::size_t expert_count, std::size_t expert_ff,
                               std::size_t shared_ff, std::size_t hidden,
                               std::size_t slab_bytes,
                               hipStream_t stream = nullptr);

/// Fused MoE branch, phase 2 (F-J1): for every job in `job_list` (or every
/// expert when `job_list` is null), mlp_out += weights_full[e] *
/// (down_e · act[e]) over hidden rows, accumulated atomically. `slab_bytes`
/// must be the encoded per-expert slab of the DOWN tensor's own type; the
/// shared matrix decodes with `shared_type`.
void LaunchFlintFusedMoEDown(const void* down_stack,
                             const void* down_shared, core::GgmlType type,
                             core::GgmlType shared_type, const float* act,
                             float* mlp_out, const float* weights_full,
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
/// act[slot, r] = silu(gate_e[r] . x[token(slot)]) * (up_e[r] . x[token(slot)]).
/// `offsets`/`slot_tokens` come from LaunchFlintMoeSlots; `slab_bytes` is the
/// encoded per-expert slab of the routed stacks.
void LaunchFlintMoeBatchSwiGLU(const void* gate_stack, const void* up_stack,
                               core::GgmlType type, const float* x, float* act,
                               const int* offsets, const int* slot_tokens,
                               std::size_t expert_count, std::size_t expert_ff,
                               std::size_t hidden, std::size_t slab_bytes,
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
/// shapes (gate/up in a no-offset WK format, down Q8_0, 128-divisible m).
[[nodiscard]] bool FlintGroupedWmmaSupported(core::GgmlType gate_up_type,
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
/// act from its slot-major tiled Q8_1 buffer (down stacks are Q8_0).
void LaunchFlintMoeGroupedDownWmma(const void* down_stack, const void* act_q8,
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

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_MOE_HPP_
