#ifndef GUFO_MODELS_QWEN_HIP_OPS_MOE_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_MOE_HPP_

#include <cstddef>
#include <cstdint>

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

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_MOE_HPP_
