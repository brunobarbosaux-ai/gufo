// Flint MoE device kernels (F-E1/F-E3): sigmoid router top-k selection and
// weighted expert accumulation. Selection matches the CPU reference exactly:
// sigmoid over the full expert set, top-`used` with ties resolved to the
// lowest index, renormalisation over the selected probabilities, then the
// expert weights scale.
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops/moe.hpp"

namespace gufo::hip {

namespace {

constexpr int kMaxExperts = 1024;
constexpr int kMaxSelected = 8;

__global__ void FlintRouterTopKKernel(const float* __restrict__ logits,
                                      int expert_count, int used, int norm,
                                      float scale,
                                      std::uint32_t* __restrict__ selected,
                                      float* __restrict__ weights) {
  __shared__ float s_val[kMaxExperts];
  __shared__ float s_warp_best[32];
  __shared__ unsigned int s_warp_idx[32];
  __shared__ float s_winner_value[kMaxSelected];
  __shared__ float s_sum;

  const int tid = static_cast<int>(threadIdx.x);
  for (int e = tid; e < expert_count; e += static_cast<int>(blockDim.x)) {
    const float z = logits[e];
    s_val[e] = 1.0F / (1.0F + expf(-z));
  }
  __syncthreads();

  float sum = 0.0F;
  for (int k = 0; k < used; ++k) {
    float best_value = -INFINITY;
    unsigned int best_index = 0xFFFFFFFFU;
    for (int e = tid; e < expert_count; e += static_cast<int>(blockDim.x)) {
      const float v = s_val[e];
      if (v > best_value || (v == best_value && static_cast<unsigned int>(e) <
                                                      best_index)) {
        best_value = v;
        best_index = static_cast<unsigned int>(e);
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other_value = __shfl_down(best_value, offset);
      const unsigned int other_index =
          __shfl_down(best_index, offset);
      if (other_value > best_value ||
          (other_value == best_value && other_index < best_index)) {
        best_value = other_value;
        best_index = other_index;
      }
    }
    const int warp = tid >> 5;
    const int lanes = static_cast<int>(blockDim.x) >> 5;
    if ((tid & 31) == 0) {
      s_warp_best[warp] = best_value;
      s_warp_idx[warp] = best_index;
    }
    __syncthreads();
    if (tid == 0) {
      float value = s_warp_best[0];
      unsigned int index = s_warp_idx[0];
      for (int w = 1; w < lanes; ++w) {
        if (s_warp_best[w] > value ||
            (s_warp_best[w] == value && s_warp_idx[w] < index)) {
          value = s_warp_best[w];
          index = s_warp_idx[w];
        }
      }
      selected[k] = index;
      s_winner_value[k] = value;
      sum += value;
      s_val[index] = -INFINITY;
      if (k == used - 1) {
        s_sum = sum;
      }
    }
    __syncthreads();
  }

  if (tid < used) {
    float weight = s_winner_value[tid];
    if (norm && s_sum > 0.0F) {
      weight /= s_sum;
    }
    weights[tid] = weight * scale;
  }
}

__global__ void FlintExpertAccumulateKernel(float* __restrict__ out,
                                            const float* __restrict__ expert,
                                            float weight, std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       i < n; i += stride) {
    out[i] += weight * expert[i];
  }
}

__global__ void FlintZeroKernel(float* __restrict__ out, std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
       i < n; i += stride) {
    out[i] = 0.0F;
  }
}

}  // namespace

void LaunchFlintRouterTopK(const float* logits, std::size_t expert_count,
                           std::uint32_t used, bool norm, float scale,
                           std::uint32_t* selected, float* weights,
                           hipStream_t stream) {
  FlintRouterTopKKernel<<<1, 256, 0, stream>>>(
      logits, static_cast<int>(expert_count), static_cast<int>(used),
      norm ? 1 : 0, scale, selected, weights);
}

void LaunchFlintExpertAccumulate(float* out, const float* expert_out,
                                 float weight, std::size_t n,
                                 hipStream_t stream) {
  if (n == 0) {
    return;
  }
  FlintExpertAccumulateKernel<<<256, 256, 0, stream>>>(out, expert_out, weight,
                                                       n);
}

void LaunchFlintZero(float* out, std::size_t n, hipStream_t stream) {
  if (n == 0) {
    return;
  }
  FlintZeroKernel<<<256, 256, 0, stream>>>(out, n);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
