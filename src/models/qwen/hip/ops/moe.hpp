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

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_MOE_HPP_
