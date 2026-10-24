// Flint batched MoE (F-L4): chunk-prefill routing and expert-grouped GEMV.
//
// The decode path (flint_moe.hip, F-J1) runs one token through a router that
// compacts the selected experts into a job list and two fused kernels that
// index expert slabs through it. That structure has no reuse: every prompt
// token re-reads the ~850 MB of expert weights it selects and the layer pays
// ~590 kernel launches, which is why token-by-token prefill ran at 35 t/s.
//
// The chunk path below keeps llama.cpp's reuse structure. For a chunk of N
// tokens the sigmoid router runs as one batched GEMM, the top-k selection runs
// one block per row, and the selection is compacted on device into
// expert-sorted slots (token index + router weight per slot). The two expert
// kernels then launch a grid over (expert, row tile) and loop over that
// expert's slot bucket, so each weight block is decoded once per bucket
// instead of once per token. The down projection writes compact slot rows and
// a deterministic combine sums the `used` slots per token in selection order.
//
// Everything here is device-resident: no per-layer readback, no host
// synchronization inside the chunk.
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {
namespace {

constexpr int kFlintBatchWarps = 4;        // warps per expert block
constexpr int kFlintBatchRowsPerWarp = 2;  // weight rows per warp
constexpr int kFlintBatchTokens = 8;       // tokens sharing one weight read
constexpr int kFlintRouterWarps = 8;       // experts per router block
constexpr int kFlintRouterTokens = 8;      // tokens per router block

// ---------------------------------------------------------------------------
// Router: logits[N, E] = x[N, H] * W[E, H]^T for the whole chunk.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void FlintRouterBatchKernel(const float* __restrict__ x,
                                       int n_tokens,
                                       const T* __restrict__ w,
                                       int expert_count, int hidden,
                                       float* __restrict__ logits) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int e = static_cast<int>(blockIdx.x) * kFlintRouterWarps + warp;
  const int t0 = static_cast<int>(blockIdx.y) * kFlintRouterTokens;
  if (e >= expert_count || t0 >= n_tokens) {
    return;
  }
  const int tn = min(kFlintRouterTokens, n_tokens - t0);
  const T* row = w + static_cast<std::size_t>(e) * hidden;
  float acc[kFlintRouterTokens];
#pragma unroll
  for (int t = 0; t < kFlintRouterTokens; ++t) {
    acc[t] = 0.0F;
  }
  for (int h = lane; h < hidden; h += 32) {
    const float wv = static_cast<float>(row[h]);
#pragma unroll
    for (int t = 0; t < kFlintRouterTokens; ++t) {
      if (t < tn) {
        acc[t] += wv * x[static_cast<std::size_t>(t0 + t) * hidden + h];
      }
    }
  }
  for (int t = 0; t < tn; ++t) {
    float v = acc[t];
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_xor(v, off);
    }
    if (lane == 0) {
      logits[static_cast<std::size_t>(t0 + t) * expert_count + e] = v;
    }
  }
}

// ---------------------------------------------------------------------------
// Top-k: one block per row, same selection rule as the decode router.
// ---------------------------------------------------------------------------

constexpr int kFlintBatchMaxExperts = 1024;
constexpr int kFlintBatchMaxSelected = 8;

__global__ void FlintRouterTopKBatchKernel(
    const float* __restrict__ logits, int expert_count, int used, int norm,
    float scale, std::uint32_t* __restrict__ selected,
    float* __restrict__ weights) {
  __shared__ float s_val[kFlintBatchMaxExperts];
  __shared__ float s_warp_best[32];
  __shared__ unsigned int s_warp_idx[32];
  __shared__ float s_winner_value[kFlintBatchMaxSelected];
  __shared__ unsigned int s_winner_idx[kFlintBatchMaxSelected];
  __shared__ float s_sum;
  const int tid = static_cast<int>(threadIdx.x);
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  const float* row_logits = logits + row * expert_count;
  if (tid == 0) {
    s_sum = 0.0F;
  }
  __syncthreads();
  for (int e = tid; e < expert_count; e += static_cast<int>(blockDim.x)) {
    s_val[e] = 1.0F / (1.0F + expf(-row_logits[e]));
  }
  __syncthreads();
  float sum = 0.0F;
  for (int k = 0; k < used; ++k) {
    float best_value = -INFINITY;
    unsigned int best_index = 0xFFFFFFFFU;
    for (int e = tid; e < expert_count; e += static_cast<int>(blockDim.x)) {
      const float v = s_val[e];
      if (v > best_value ||
          (v == best_value &&
           static_cast<unsigned int>(e) < best_index)) {
        best_value = v;
        best_index = static_cast<unsigned int>(e);
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other_value = __shfl_down(best_value, offset);
      const unsigned int other_index = __shfl_down(best_index, offset);
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
      s_winner_value[k] = value;
      s_winner_idx[k] = index;
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
    weights[row * used + tid] = weight * scale;
    selected[row * used + tid] = s_winner_idx[tid];
  }
}

// ---------------------------------------------------------------------------
// Expert-sorted slot compaction (device-side, one pass over N*used entries).
// ---------------------------------------------------------------------------

__global__ void FlintMoeCountKernel(const std::uint32_t* __restrict__ selected,
                                    int total, int* __restrict__ counts) {
  const int stride = static_cast<int>(blockDim.x * gridDim.x);
  for (int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    atomicAdd(&counts[selected[i]], 1);
  }
}

__global__ void FlintMoeOffsetsKernel(const int* __restrict__ counts,
                                      int* __restrict__ offsets,
                                      int* __restrict__ cursors,
                                      int expert_count) {
  if (threadIdx.x != 0) {
    return;
  }
  int running = 0;
  for (int e = 0; e < expert_count; ++e) {
    offsets[e] = running;
    cursors[e] = running;
    running += counts[e];
  }
  offsets[expert_count] = running;
}

__global__ void FlintMoeScatterKernel(
    const std::uint32_t* __restrict__ selected,
    const float* __restrict__ sel_weights, int* __restrict__ cursors,
    int* __restrict__ slot_tokens, float* __restrict__ slot_weights,
    int* __restrict__ token_slots, int total, int used) {
  const int stride = static_cast<int>(blockDim.x * gridDim.x);
  for (int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    const int e = static_cast<int>(selected[i]);
    // cursors[e] starts at offsets[e], so the atomic itself yields the
    // expert's next compact global slot in [offsets[e], offsets[e+1]).
    const int slot = atomicAdd(&cursors[e], 1);
    slot_tokens[slot] = i / used;
    slot_weights[slot] = sel_weights[i];
    token_slots[i] = slot;
  }
}

// ---------------------------------------------------------------------------
// Expert-grouped SwiGLU: act[slot, r] = silu(gate_e[r] . x[token(slot)]) *
// (up_e[r] . x[token(slot)]). One block owns (expert, row tile); the bucket's
// tokens are walked in groups of kFlintBatchTokens so a decoded weight block
// feeds eight tokens before it is dropped.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void FlintReduceTokens(float* acc, int tn) {
  for (int t = 0; t < tn; ++t) {
    float v = acc[t];
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_xor(v, off);
    }
    acc[t] = v;
  }
}

/// Sub-16-decoded (Q6_K/Q4_K/Q5_K/Q3_K/IQ4) multi-token dot of one row.
__device__ __forceinline__ void FlintDotRowSub16(
    core::GgmlType type, const char* __restrict__ row, int lane,
    const float* __restrict__ x, int x_stride,
    const int* __restrict__ tokens, int tn, int hidden, float* acc) {
  const int num_sub = static_cast<int>(hidden / 16);
  QuantSub16 sub;
  for (int s = lane; s < num_sub; s += 32) {
    DecodeQuantSub16(type, row, static_cast<std::size_t>(s), sub);
    for (int t = 0; t < tn; ++t) {
      const float* xb =
          x + static_cast<std::size_t>(tokens[t]) * x_stride + s * 16;
      float dot = 0.0F;
      float xsum = 0.0F;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const float v = xb[j];
        dot += static_cast<float>(sub.q[j]) * v;
        xsum += v;
      }
      acc[t] += (sub.scale * dot) - (sub.offset * xsum);
    }
  }
}

/// Q8_0 multi-token dot of one row.
__device__ __forceinline__ void FlintDotRowQ8_0(
    const char* __restrict__ row, int lane, const float* __restrict__ x,
    int x_stride, const int* __restrict__ tokens, int tn, int k, float* acc) {
  const int num_blocks = static_cast<int>(k / kQ8_0BlockSize);
  const auto* blocks = reinterpret_cast<const Q8_0Block*>(row);
  for (int b = lane; b < num_blocks; b += 32) {
    const Q8_0Block& wblk = blocks[b];
    const float d_w = __half2float(wblk.d);
    for (int t = 0; t < tn; ++t) {
      const float* xb =
          x + static_cast<std::size_t>(tokens[t]) * x_stride + b * kQ8_0BlockSize;
      float dot = 0.0F;
#pragma unroll
      for (int i = 0; i < kQ8_0BlockSize; ++i) {
        dot += static_cast<float>(wblk.qs[i]) * xb[i];
      }
      acc[t] += d_w * dot;
    }
  }
}

__device__ __forceinline__ void FlintDotRow(
    core::GgmlType type, const char* __restrict__ row, int lane,
    const float* __restrict__ x, int x_stride,
    const int* __restrict__ tokens, int tn, int k, float* acc) {
  if (type == core::GgmlType::kQ8_0 && (k % kQ8_0BlockSize) == 0) {
    FlintDotRowQ8_0(row, lane, x, x_stride, tokens, tn, k, acc);
    return;
  }
  FlintDotRowSub16(type, row, lane, x, x_stride, tokens, tn, k, acc);
}

__global__ void __launch_bounds__(kFlintBatchWarps * 32, 4)
    FlintMoeBatchSwiGLUKernel(const void* __restrict__ gate_stack,
                              const void* __restrict__ up_stack,
                              core::GgmlType type, const float* __restrict__ x,
                              float* __restrict__ act,
                              const int* __restrict__ offsets,
                              const int* __restrict__ slot_tokens,
                              int expert_ff, int hidden,
                              std::size_t slab_bytes) {
  const int e = static_cast<int>(blockIdx.y);
  const int begin = offsets[e];
  const int end = offsets[e + 1];
  if (begin >= end) {
    return;
  }
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int r0 =
      (static_cast<int>(blockIdx.x) * kFlintBatchWarps + warp) *
      kFlintBatchRowsPerWarp;
  if (r0 >= expert_ff) {
    return;
  }
  const bool has_r1 = (r0 + 1) < expert_ff;

  const char* gbase = static_cast<const char*>(gate_stack) +
                      static_cast<std::size_t>(e) * slab_bytes;
  const char* ubase = static_cast<const char*>(up_stack) +
                      static_cast<std::size_t>(e) * slab_bytes;
  const std::size_t row_bytes = QuantRowBytes(type, static_cast<std::size_t>(hidden));
  const char* grow0 = gbase + static_cast<std::size_t>(r0) * row_bytes;
  const char* urow0 = ubase + static_cast<std::size_t>(r0) * row_bytes;
  const char* grow1 = has_r1
                          ? gbase + static_cast<std::size_t>(r0 + 1) * row_bytes
                          : nullptr;
  const char* urow1 = has_r1
                          ? ubase + static_cast<std::size_t>(r0 + 1) * row_bytes
                          : nullptr;

  for (int tb = begin; tb < end; tb += kFlintBatchTokens) {
    const int tn = min(kFlintBatchTokens, end - tb);
    const int* toks = slot_tokens + tb;
    float ag0[kFlintBatchTokens];
    float au0[kFlintBatchTokens];
    float ag1[kFlintBatchTokens];
    float au1[kFlintBatchTokens];
#pragma unroll
    for (int t = 0; t < kFlintBatchTokens; ++t) {
      ag0[t] = 0.0F;
      au0[t] = 0.0F;
      ag1[t] = 0.0F;
      au1[t] = 0.0F;
    }
    FlintDotRow(type, grow0, lane, x, hidden, toks, tn, hidden, ag0);
    FlintDotRow(type, urow0, lane, x, hidden, toks, tn, hidden, au0);
    if (grow1 != nullptr) {
      FlintDotRow(type, grow1, lane, x, hidden, toks, tn, hidden, ag1);
      FlintDotRow(type, urow1, lane, x, hidden, toks, tn, hidden, au1);
    }
    FlintReduceTokens(ag0, tn);
    FlintReduceTokens(au0, tn);
    if (grow1 != nullptr) {
      FlintReduceTokens(ag1, tn);
      FlintReduceTokens(au1, tn);
    }
    if (lane == 0) {
      for (int t = 0; t < tn; ++t) {
        const std::size_t slot =
            static_cast<std::size_t>(tb + t) * expert_ff;
        const float g0 = ag0[t];
        act[slot + r0] = (g0 / (1.0F + expf(-g0))) * au0[t];
        if (grow1 != nullptr) {
          const float g1 = ag1[t];
          act[slot + r0 + 1] = (g1 / (1.0F + expf(-g1))) * au1[t];
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Expert-grouped down projection: down_out[slot, r] = w(slot) * down_e[r] .
// act[slot]. Compact slot rows keep the combine deterministic.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(kFlintBatchWarps * 32, 4)
    FlintMoeBatchDownKernel(const void* __restrict__ down_stack,
                            core::GgmlType type, const float* __restrict__ act,
                            float* __restrict__ down_out,
                            const int* __restrict__ offsets,
                            const float* __restrict__ slot_weights,
                            int expert_ff, int hidden,
                            std::size_t slab_bytes) {
  const int e = static_cast<int>(blockIdx.y);
  const int begin = offsets[e];
  const int end = offsets[e + 1];
  if (begin >= end) {
    return;
  }
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int r0 =
      (static_cast<int>(blockIdx.x) * kFlintBatchWarps + warp) *
      kFlintBatchRowsPerWarp;
  if (r0 >= hidden) {
    return;
  }
  const bool has_r1 = (r0 + 1) < hidden;

  const char* base = static_cast<const char*>(down_stack) +
                     static_cast<std::size_t>(e) * slab_bytes;
  const std::size_t row_bytes =
      QuantRowBytes(type, static_cast<std::size_t>(expert_ff));
  const char* row0 = base + static_cast<std::size_t>(r0) * row_bytes;
  const char* row1 =
      has_r1 ? base + static_cast<std::size_t>(r0 + 1) * row_bytes : nullptr;

  for (int tb = begin; tb < end; tb += kFlintBatchTokens) {
    const int tn = min(kFlintBatchTokens, end - tb);
    float s0[kFlintBatchTokens];
    float s1[kFlintBatchTokens];
#pragma unroll
    for (int t = 0; t < kFlintBatchTokens; ++t) {
      s0[t] = 0.0F;
      s1[t] = 0.0F;
    }
    if (type == core::GgmlType::kQ8_0 && (expert_ff % kQ8_0BlockSize) == 0) {
      const int num_blocks = expert_ff / kQ8_0BlockSize;
      const auto* b0 = reinterpret_cast<const Q8_0Block*>(row0);
      const auto* b1 =
          has_r1 ? reinterpret_cast<const Q8_0Block*>(row1) : nullptr;
      for (int b = lane; b < num_blocks; b += 32) {
        const Q8_0Block& d0 = b0[b];
        const float dd0 = __half2float(d0.d);
        const Q8_0Block* d1 = b1 != nullptr ? &b1[b] : nullptr;
        const float dd1 = d1 != nullptr ? __half2float(d1->d) : 0.0F;
        for (int t = 0; t < tn; ++t) {
          const float* ab = act + static_cast<std::size_t>(tb + t) * expert_ff +
                            b * kQ8_0BlockSize;
          float dot0 = 0.0F;
          float dot1 = 0.0F;
#pragma unroll
          for (int i = 0; i < kQ8_0BlockSize; ++i) {
            const float av = ab[i];
            dot0 += static_cast<float>(d0.qs[i]) * av;
            if (d1 != nullptr) {
              dot1 += static_cast<float>(d1->qs[i]) * av;
            }
          }
          s0[t] += dd0 * dot0;
          if (d1 != nullptr) {
            s1[t] += dd1 * dot1;
          }
        }
      }
    } else {
      // Sub-16 formats: the activation row is a plain f32 vector here, so the
      // weight decode is shared across the bucket the same way.
      const int num_sub = static_cast<int>(expert_ff / 16);
      QuantSub16 sub0;
      QuantSub16 sub1;
      for (int s = lane; s < num_sub; s += 32) {
        DecodeQuantSub16(type, row0, static_cast<std::size_t>(s), sub0);
        if (has_r1) {
          DecodeQuantSub16(type, row1, static_cast<std::size_t>(s), sub1);
        }
        for (int t = 0; t < tn; ++t) {
          const float* ab = act + static_cast<std::size_t>(tb + t) * expert_ff +
                            s * 16;
          float dot0 = 0.0F;
          float xsum0 = 0.0F;
          float dot1 = 0.0F;
          float xsum1 = 0.0F;
#pragma unroll
          for (int j = 0; j < 16; ++j) {
            const float v = ab[j];
            dot0 += static_cast<float>(sub0.q[j]) * v;
            xsum0 += v;
            if (has_r1) {
              dot1 += static_cast<float>(sub1.q[j]) * v;
              xsum1 += v;
            }
          }
          s0[t] += (sub0.scale * dot0) - (sub0.offset * xsum0);
          if (has_r1) {
            s1[t] += (sub1.scale * dot1) - (sub1.offset * xsum1);
          }
        }
      }
    }
    FlintReduceTokens(s0, tn);
    if (has_r1) {
      FlintReduceTokens(s1, tn);
    }
    if (lane == 0) {
      for (int t = 0; t < tn; ++t) {
        const float w = slot_weights[tb + t];
        const std::size_t out =
            static_cast<std::size_t>(tb + t) * hidden;
        down_out[out + r0] = w * s0[t];
        if (has_r1) {
          down_out[out + r0 + 1] = w * s1[t];
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Deterministic combine: mlp[t] = shared[t] + sum_j down_out[token_slots[t,j]]
// in selection order (the slot rows are already router-weighted).
// ---------------------------------------------------------------------------

__global__ void FlintMoeCombineKernel(const float* __restrict__ shared_out,
                                      const float* __restrict__ down_out,
                                      const int* __restrict__ token_slots,
                                      float* __restrict__ mlp_out, int used,
                                      int hidden) {
  const int t = static_cast<int>(blockIdx.x);
  const int* slots = token_slots + static_cast<std::size_t>(t) * used;
  for (int h = static_cast<int>(threadIdx.x); h < hidden;
       h += static_cast<int>(blockDim.x)) {
    float acc = shared_out[static_cast<std::size_t>(t) * hidden + h];
    for (int j = 0; j < used; ++j) {
      acc += down_out[static_cast<std::size_t>(slots[j]) * hidden + h];
    }
    mlp_out[static_cast<std::size_t>(t) * hidden + h] = acc;
  }
}

}  // namespace

void LaunchFlintRouterBatch(const float* x, std::size_t n_tokens,
                            const void* w, core::GgmlType type,
                            std::size_t expert_count, std::size_t hidden,
                            float* logits, hipStream_t stream) {
  if (n_tokens == 0 || expert_count == 0 || hidden == 0) {
    return;
  }
  const dim3 grid(
      static_cast<unsigned int>((expert_count + kFlintRouterWarps - 1) /
                                kFlintRouterWarps),
      static_cast<unsigned int>((n_tokens + kFlintRouterTokens - 1) /
                                kFlintRouterTokens));
  const dim3 block(kFlintRouterWarps * 32);
  if (type == core::GgmlType::kBF16) {
    FlintRouterBatchKernel<hip_bfloat16><<<grid, block, 0, stream>>>(
        x, static_cast<int>(n_tokens), static_cast<const hip_bfloat16*>(w),
        static_cast<int>(expert_count), static_cast<int>(hidden), logits);
    return;
  }
  if (type == core::GgmlType::kF16) {
    FlintRouterBatchKernel<__half><<<grid, block, 0, stream>>>(
        x, static_cast<int>(n_tokens), static_cast<const __half*>(w),
        static_cast<int>(expert_count), static_cast<int>(hidden), logits);
    return;
  }
  if (type == core::GgmlType::kF32) {
    FlintRouterBatchKernel<float><<<grid, block, 0, stream>>>(
        x, static_cast<int>(n_tokens), static_cast<const float*>(w),
        static_cast<int>(expert_count), static_cast<int>(hidden), logits);
    return;
  }
  throw std::invalid_argument(
      "flint batched router requires an f32, f16 or bf16 router tensor");
}

void LaunchFlintRouterTopKBatch(const float* logits, std::size_t n_tokens,
                                std::size_t expert_count, std::uint32_t used,
                                bool norm, float scale,
                                std::uint32_t* selected, float* weights,
                                hipStream_t stream) {
  if (n_tokens == 0 || used == 0) {
    return;
  }
  FlintRouterTopKBatchKernel<<<static_cast<unsigned int>(n_tokens), 256, 0,
                               stream>>>(
      logits, static_cast<int>(expert_count), static_cast<int>(used),
      norm ? 1 : 0, scale, selected, weights);
}

void LaunchFlintMoeSlots(const std::uint32_t* selected,
                         const float* sel_weights, std::size_t n_tokens,
                         std::size_t used, std::size_t expert_count,
                         int* counts, int* offsets, int* cursors,
                         int* slot_tokens, float* slot_weights,
                         int* token_slots, hipStream_t stream) {
  const int total = static_cast<int>(n_tokens * used);
  const unsigned int blocks =
      static_cast<unsigned int>((total + 255) / 256);
  HIP_CHECK(hipMemsetAsync(counts, 0, expert_count * sizeof(int), stream));
  FlintMoeCountKernel<<<blocks, 256, 0, stream>>>(selected, total, counts);
  FlintMoeOffsetsKernel<<<1, 32, 0, stream>>>(counts, offsets, cursors,
                                              static_cast<int>(expert_count));
  FlintMoeScatterKernel<<<blocks, 256, 0, stream>>>(
      selected, sel_weights, cursors, slot_tokens, slot_weights, token_slots,
      total, static_cast<int>(used));
}

void LaunchFlintMoeBatchSwiGLU(const void* gate_stack, const void* up_stack,
                               core::GgmlType type, const float* x, float* act,
                               const int* offsets, const int* slot_tokens,
                               std::size_t expert_count, std::size_t expert_ff,
                               std::size_t hidden, std::size_t slab_bytes,
                               hipStream_t stream) {
  if (expert_count == 0 || expert_ff == 0) {
    return;
  }
  const dim3 grid(
      static_cast<unsigned int>((expert_ff + kFlintBatchRowsPerWarp *
                                                 kFlintBatchWarps - 1) /
                                (kFlintBatchRowsPerWarp * kFlintBatchWarps)),
      static_cast<unsigned int>(expert_count));
  FlintMoeBatchSwiGLUKernel<<<grid, kFlintBatchWarps * 32, 0, stream>>>(
      gate_stack, up_stack, type, x, act, offsets, slot_tokens,
      static_cast<int>(expert_ff), static_cast<int>(hidden), slab_bytes);
}

void LaunchFlintMoeBatchDown(const void* down_stack, core::GgmlType type,
                             const float* act, float* down_out,
                             const int* offsets, const float* slot_weights,
                             std::size_t expert_count, std::size_t expert_ff,
                             std::size_t hidden, std::size_t slab_bytes,
                             hipStream_t stream) {
  if (expert_count == 0 || hidden == 0) {
    return;
  }
  const dim3 grid(
      static_cast<unsigned int>((hidden + kFlintBatchRowsPerWarp *
                                              kFlintBatchWarps - 1) /
                                (kFlintBatchRowsPerWarp * kFlintBatchWarps)),
      static_cast<unsigned int>(expert_count));
  FlintMoeBatchDownKernel<<<grid, kFlintBatchWarps * 32, 0, stream>>>(
      down_stack, type, act, down_out, offsets, slot_weights,
      static_cast<int>(expert_ff), static_cast<int>(hidden), slab_bytes);
}

void LaunchFlintMoeCombine(const float* shared_out, const float* down_out,
                           const int* token_slots, float* mlp_out,
                           std::size_t n_tokens, std::size_t used,
                           std::size_t hidden, hipStream_t stream) {
  if (n_tokens == 0 || hidden == 0) {
    return;
  }
  FlintMoeCombineKernel<<<static_cast<unsigned int>(n_tokens), 256, 0,
                          stream>>>(shared_out, down_out, token_slots, mlp_out,
                                    static_cast<int>(used),
                                    static_cast<int>(hidden));
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
