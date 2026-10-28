#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/models/qwen/hip/detail/flint_keycount.hpp"
#include "src/models/qwen/hip/detail/flint_kv.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"

namespace gufo::hip {

GUFO_FLINT_KEYCOUNT_TU(graph)

using detail::FlintKvLoad;
using detail::FlintKvOffset;

__global__ void WriteKVCachePtrKernel(
    const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    std::size_t layer_base, std::uint32_t rows,
    const std::uint32_t* __restrict__ d_pos, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::uint32_t kv_h = blockIdx.x;
  if (kv_h >= num_kv_heads)
    return;
  const std::size_t tid = threadIdx.x;
  if (tid >= head_dim)
    return;

  const std::uint32_t pos = *d_pos;
  const std::size_t kv_offset = FlintKvOffset<float>(
      layer_base, kv_h, pos, rows, max_context, num_kv_heads, head_dim);
  const float key = k[static_cast<std::size_t>(kv_h) * head_dim + tid];
  const float val = v[static_cast<std::size_t>(kv_h) * head_dim + tid];
  if (k_cache != nullptr) {
    k_cache[kv_offset + tid] = key;
  }
  if (v_cache != nullptr) {
    v_cache[kv_offset + tid] = val;
  }

  if (k_cache_f16 != nullptr || v_cache_f16 != nullptr) {
    const std::size_t f16_offset = FlintKvOffset<__half>(
        layer_base, kv_h, pos, rows, max_context, num_kv_heads, head_dim);
    if (k_cache_f16 != nullptr) {
      k_cache_f16[f16_offset] = __float2half_rn(key);
    }
    if (v_cache_f16 != nullptr) {
      v_cache_f16[f16_offset] = __float2half_rn(val);
    }
  }
}

__device__ __forceinline__ float4 LoadGraphHalf4(const __half* source) {
  const auto* pairs = reinterpret_cast<const __half2*>(source);
  const float2 lower = __half22float2(pairs[0]);
  const float2 upper = __half22float2(pairs[1]);
  return make_float4(lower.x, lower.y, upper.x, upper.y);
}

__launch_bounds__(32) __global__ void QwenDecodeOnlineAttentionPtrKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx,
    const std::uint32_t* __restrict__ d_pos, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads) {
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / 32;

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || lane >= 32) {
    return;
  }

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * head_dim));
  const std::size_t cache_head_offset =
      ((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_head) *
       max_context * head_dim);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float4 query_values[vectors_per_lane];
  float4 output_values[vectors_per_lane] = {};
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    query_values[vector] = query[lane + (vector * 32)];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const std::uint32_t pos = *d_pos;
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  for (std::size_t position = 0; position < sequence_length; ++position) {
    const auto* key = reinterpret_cast<const float4*>(
        k_cache + cache_head_offset + (position * head_dim));
    float score = 0.0F;
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 query_value = query_values[vector];
      const float4 key_value = key[lane + (vector * 32)];
      score += (query_value.x * key_value.x) + (query_value.y * key_value.y) +
               (query_value.z * key_value.z) + (query_value.w * key_value.w);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      score += __shfl_down(score, offset);
    }
    score = __shfl(score, 0) * scale;

    const float new_max = fmaxf(running_max, score);
    const float previous_scale = expf(running_max - new_max);
    const float value_scale = expf(score - new_max);
    running_sum = (running_sum * previous_scale) + value_scale;
    running_max = new_max;

    const auto* value = reinterpret_cast<const float4*>(
        v_cache + cache_head_offset + (position * head_dim));
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 value_data = value[lane + (vector * 32)];
      float4& output = output_values[vector];
      output.x = (output.x * previous_scale) + (value_data.x * value_scale);
      output.y = (output.y * previous_scale) + (value_data.y * value_scale);
      output.z = (output.z * previous_scale) + (value_data.z * value_scale);
      output.w = (output.w * previous_scale) + (value_data.w * value_scale);
    }
  }

  auto* output = reinterpret_cast<float4*>(
      out_context + (static_cast<std::size_t>(head) * head_dim));
  const auto* gate_values =
      gate == nullptr ? nullptr
                      : reinterpret_cast<const float4*>(
                            gate + (static_cast<std::size_t>(head) * head_dim));
  const float inverse_sum = 1.0F / running_sum;
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    float4 result = output_values[vector];
    result.x *= inverse_sum;
    result.y *= inverse_sum;
    result.z *= inverse_sum;
    result.w *= inverse_sum;
    if (gate_values != nullptr) {
      const float4 gate_value = gate_values[lane + (vector * 32)];
      result.x *= 1.0F / (1.0F + expf(-gate_value.x));
      result.y *= 1.0F / (1.0F + expf(-gate_value.y));
      result.z *= 1.0F / (1.0F + expf(-gate_value.z));
      result.w *= 1.0F / (1.0F + expf(-gate_value.w));
    }
    output[lane + (vector * 32)] = result;
  }
}

__launch_bounds__(32) __global__ void QwenDecodeOnlineAttentionHalfPtrKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx,
    const std::uint32_t* __restrict__ d_pos, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads) {
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / 32;

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || lane >= 32) {
    return;
  }

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * head_dim));
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t cache_head_offset =
      (static_cast<std::size_t>(layer_idx) * max_context * kv_width) +
      (static_cast<std::size_t>(kv_head) * head_dim);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float4 query_values[vectors_per_lane];
  float4 output_values[vectors_per_lane] = {};
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    query_values[vector] = query[lane + (vector * 32)];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const std::uint32_t pos = *d_pos;
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  // Load neighboring keys and values together, then retain the scalar
  // position order for every softmax and output update.
  constexpr std::size_t positions_per_load = 2;
  for (std::size_t position = 0; position < sequence_length;
       position += positions_per_load) {
    float scores[positions_per_load]{};
    float4 values[positions_per_load][vectors_per_lane];
#pragma unroll
    for (std::size_t step = 0; step < positions_per_load; ++step) {
      if (position + step >= sequence_length)
        break;
      const auto offset = cache_head_offset + ((position + step) * kv_width);
      const __half* key = k_cache + offset;
      const __half* value = v_cache + offset;
      float score = 0.0F;
#pragma unroll
      for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
        const float4 query_value = query_values[vector];
        const auto element = (lane + (vector * 32)) * 4;
        const float4 key_value = LoadGraphHalf4(key + element);
        values[step][vector] = LoadGraphHalf4(value + element);
        score += (query_value.x * key_value.x) + (query_value.y * key_value.y) +
                 (query_value.z * key_value.z) + (query_value.w * key_value.w);
      }
      // Lane zero keeps the descending reduction tree. Inside each
      // 16-lane row, DPP avoids the LDS path used by shuffle instructions.
      score += __shfl_down(score, 16);
      score += __builtin_bit_cast(
          float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, score),
                                             0x168, 0xF, 0xF, false));
      score += __builtin_bit_cast(
          float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, score),
                                             0x164, 0xF, 0xF, false));
      score += __builtin_bit_cast(
          float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, score),
                                             0x162, 0xF, 0xF, false));
      score += __builtin_bit_cast(
          float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, score),
                                             0x161, 0xF, 0xF, false));
      scores[step] = __shfl(score, 0) * scale;
    }

#pragma unroll
    for (std::size_t step = 0; step < positions_per_load; ++step) {
      if (position + step >= sequence_length)
        break;
      const float new_max = fmaxf(running_max, scores[step]);
      const float previous_scale = expf(running_max - new_max);
      const float value_scale = expf(scores[step] - new_max);
      running_sum = (running_sum * previous_scale) + value_scale;
      running_max = new_max;

#pragma unroll
      for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
        const float4 value_data = values[step][vector];
        float4& output = output_values[vector];
        // Retain the original half-pair kernel's rounded product and FMA.
        output.x = __fmaf_rn(output.x, previous_scale,
                             __fmul_rn(value_data.x, value_scale));
        output.y = __fmaf_rn(output.y, previous_scale,
                             __fmul_rn(value_data.y, value_scale));
        output.z = __fmaf_rn(value_data.z, value_scale,
                             __fmul_rn(output.z, previous_scale));
        output.w = __fmaf_rn(value_data.w, value_scale,
                             __fmul_rn(output.w, previous_scale));
      }
    }
  }

  auto* output = reinterpret_cast<float4*>(
      out_context + (static_cast<std::size_t>(head) * head_dim));
  const auto* gate_values =
      gate == nullptr ? nullptr
                      : reinterpret_cast<const float4*>(
                            gate + (static_cast<std::size_t>(head) * head_dim));
  const float inverse_sum = 1.0F / running_sum;
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    float4 result = output_values[vector];
    result.x *= inverse_sum;
    result.y *= inverse_sum;
    result.z *= inverse_sum;
    result.w *= inverse_sum;
    if (gate_values != nullptr) {
      const float4 gate_value = gate_values[lane + (vector * 32)];
      result.x *= 1.0F / (1.0F + expf(-gate_value.x));
      result.y *= 1.0F / (1.0F + expf(-gate_value.y));
      result.z *= 1.0F / (1.0F + expf(-gate_value.z));
      result.w *= 1.0F / (1.0F + expf(-gate_value.w));
    }
    output[lane + (vector * 32)] = result;
  }
}

/// Fixed-LDS twin of AttentionKernel (attention_decode.hip) for graph
/// capture: identical arithmetic and summation order, but the score row is
/// a compile-time LDS array instead of a seq-sized dynamic allocation —
/// dynamic LDS is baked into the graph node and cannot vary with position.
/// The kernel reads the position from d_pos and only touches s_scores[0,
/// span). kSpan comes from the capture-time worst case (windowed: the
/// sliding window at 513; global: max_context on the 4096/8192 ladder —
/// 8192 floats + the 1 KiB reduce row is the 64 KiB static-LDS ceiling;
/// beyond it the online-tile kernel takes over, as in the legacy path).
template <std::size_t kSpan, typename KVT = float>
__global__ void AttentionKernelGraph(
    const float* __restrict__ q, const float* __restrict__ gate,
    const KVT* __restrict__ k_cache, const KVT* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx,
    std::size_t layer_base, const std::uint32_t* __restrict__ d_pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    detail::FlintKernelPolicy policy) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads) {
    return;
  }

  const std::uint32_t group_size =
      (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const std::uint32_t kv_h = h / group_size;
  const std::size_t tid = threadIdx.x;

  const float* q_h = q + (static_cast<std::size_t>(h) * head_dim);
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dim));
  const std::uint32_t pos = *d_pos;
  const std::size_t seq_len = static_cast<std::size_t>(pos) + 1;
  // F-D2: the window keeps the last `window` keys; scores use relative
  // indexing so the LDS row spans only the visible range.
  const std::size_t p_begin =
      (policy.window != 0 && seq_len > policy.window)
          ? seq_len - policy.window
          : 0;
  const std::size_t span = seq_len - p_begin;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    detail::FlintKeyCountAdd(layer_idx, static_cast<std::uint32_t>(span));
  }
  // F-D1: the sink joins max and denominator (F-D6 order: scale -> sink ->
  // softmax -> weighted sum) and contributes no value.
  const bool has_sink = policy.sinks != nullptr;
  const float sink = has_sink ? policy.sinks[h] : 0.0F;

  __shared__ float s_scores[kSpan];
  __shared__ float s_scratch[256];

  // Step 1: Compute scores for each position p = p_begin..seq_len-1
  for (std::size_t p = p_begin; p < seq_len; ++p) {
    const std::size_t p_offset = FlintKvOffset<KVT>(
        layer_base, kv_h, p, policy.kv_rows, max_context, num_kv_heads,
        head_dim);
    const KVT* k_p = k_cache + p_offset;

    float dot = 0.0F;
    for (std::size_t d = tid; d < head_dim; d += blockDim.x) {
      dot += q_h[d] * FlintKvLoad(k_p + d);
    }
    s_scratch[tid] = dot;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (tid < stride) {
        s_scratch[tid] += s_scratch[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      s_scores[p - p_begin] = s_scratch[0] * scale;
    }
    __syncthreads();
  }

  // Step 2: Softmax across s_scores[0..span-1]
  float local_max = -1e30F;
  for (std::size_t p = tid; p < span; p += blockDim.x) {
    local_max = fmaxf(local_max, s_scores[p]);
  }
  s_scratch[tid] = local_max;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_scratch[tid] = fmaxf(s_scratch[tid], s_scratch[tid + stride]);
    }
    __syncthreads();
  }
  float max_score = s_scratch[0];
  if (has_sink) {
    max_score = fmaxf(max_score, sink);
  }
  __syncthreads();

  float local_sum = 0.0F;
  for (std::size_t p = tid; p < span; p += blockDim.x) {
    const float e = expf(s_scores[p] - max_score);
    s_scores[p] = e;
    local_sum += e;
  }
  if (has_sink && tid == 0) {
    local_sum += expf(sink - max_score);
  }
  s_scratch[tid] = local_sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_scratch[tid] += s_scratch[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum_exp = 1.0F / (s_scratch[0] + 1e-9F);
  __syncthreads();

  for (std::size_t p = tid; p < span; p += blockDim.x) {
    s_scores[p] *= inv_sum_exp;
  }
  __syncthreads();

  // Step 3: Weighted sum of V
  if (tid < head_dim) {
    float ctx = 0.0F;
    for (std::size_t p = p_begin; p < seq_len; ++p) {
      const std::size_t p_offset = FlintKvOffset<KVT>(
          layer_base, kv_h, p, policy.kv_rows, max_context, num_kv_heads,
          head_dim);
      const KVT* v_p = v_cache + p_offset;
      ctx += s_scores[p - p_begin] * FlintKvLoad(v_p + tid);
    }

    if (gate != nullptr) {
      const float g = gate[static_cast<std::size_t>(h) * head_dim + tid];
      const float sig = 1.0F / (1.0F + expf(-g));
      ctx *= sig;
    }
    out_context[static_cast<std::size_t>(h) * head_dim + tid] = ctx;
  }
}

/// d_pos twin of FlintDecodeOnlineAttentionKernel (attention_decode.hip):
/// identical arithmetic and tile order — the kernel only reads the position
/// from device memory so a captured graph serves every position. Fixed grid
/// (num_heads blocks, 256 threads) and fixed LDS (two 256-float rows) — the
/// graph-capture requirements the baseline's seq-sized score row breaks.
template <typename KVT = float>
__global__ void FlintDecodeOnlineAttentionPtrKernel(
    const float* __restrict__ q, const KVT* __restrict__ k_cache,
    const KVT* __restrict__ v_cache, float* __restrict__ out_context,
    std::uint32_t layer_idx, std::size_t layer_base,
    const std::uint32_t* __restrict__ d_pos, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    detail::FlintKernelPolicy policy) {
  constexpr std::size_t kTile = 256;
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads) {
    return;
  }
  const std::uint32_t group_size =
      (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const std::uint32_t kv_h = h / group_size;
  const std::size_t tid = threadIdx.x;

  const float* q_h = q + static_cast<std::size_t>(h) * head_dim;
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dim));
  const std::uint32_t pos = *d_pos;
  const std::size_t seq_len = static_cast<std::size_t>(pos) + 1;
  const std::size_t p_begin =
      (policy.window != 0 && seq_len > policy.window)
          ? seq_len - policy.window
          : 0;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    detail::FlintKeyCountAdd(layer_idx,
                             static_cast<std::uint32_t>(seq_len - p_begin));
  }
  const bool has_sink = policy.sinks != nullptr;
  const float sink = has_sink ? policy.sinks[h] : 0.0F;

  __shared__ float s_scores[kTile];
  __shared__ float s_reduce[kTile];
  __shared__ float s_m;     // running max
  __shared__ float s_l;     // running denominator
  __shared__ float s_alpha; // this tile's O rescale

  float o = 0.0F; // thread-private output dim (tid < head_dim)

  if (tid == 0) {
    s_m = has_sink ? sink : -INFINITY;
    s_l = has_sink ? 1.0F : 0.0F;
  }
  __syncthreads();

  for (std::size_t tile = p_begin; tile < seq_len; tile += kTile) {
    const std::size_t tile_end = std::min(tile + kTile, seq_len);
    const std::size_t n = tile_end - tile;
    // 1. Scores for this tile (thread-strided over keys, full dot per key).
    for (std::size_t i = tid; i < n; i += kTile) {
      const std::size_t p = tile + i;
      const std::size_t p_offset = FlintKvOffset<KVT>(
          layer_base, kv_h, p, policy.kv_rows, max_context, num_kv_heads,
          head_dim);
      const KVT* k_p = k_cache + p_offset;
      float dot = 0.0F;
      for (std::size_t d = 0; d < head_dim; ++d) {
        dot += q_h[d] * FlintKvLoad(k_p + d);
      }
      s_scores[i] = dot * scale;
    }
    __syncthreads();
    // 2. Tile max -> running max + rescale.
    float local_max = -INFINITY;
    for (std::size_t i = tid; i < n; i += kTile) {
      local_max = fmaxf(local_max, s_scores[i]);
    }
    s_reduce[tid] = local_max;
    __syncthreads();
    for (std::size_t stride = kTile / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + stride]);
      }
      __syncthreads();
    }
    if (tid == 0) {
      const float m_new = fmaxf(s_m, s_reduce[0]);
      s_alpha = expf(s_m - m_new);
      s_m = m_new;
    }
    __syncthreads();
    const float alpha = s_alpha;
    // 3. Exponentiates + tile sum -> running denominator.
    float local_sum = 0.0F;
    for (std::size_t i = tid; i < n; i += kTile) {
      const float e = expf(s_scores[i] - s_m);
      s_scores[i] = e;
      local_sum += e;
    }
    s_reduce[tid] = local_sum;
    __syncthreads();
    for (std::size_t stride = kTile / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        s_reduce[tid] += s_reduce[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      s_l = s_l * alpha + s_reduce[0];
    }
    // 4. O accumulation: rescale then add this tile's weighted V row.
    if (tid < head_dim) {
      o *= alpha;
      for (std::size_t i = 0; i < n; ++i) {
        const std::size_t v_offset = FlintKvOffset<KVT>(
            layer_base, kv_h, tile + i, policy.kv_rows, max_context, num_kv_heads,
            head_dim);
        o += s_scores[i] * FlintKvLoad(v_cache + v_offset + tid);
      }
    }
    __syncthreads();
  }
  if (tid < head_dim) {
    out_context[static_cast<std::size_t>(h) * head_dim + tid] = o / s_l;
  }
}

/// Split-KV GQA-sharing partial for flint global-layer decode (see
/// LaunchFlintDecodeSplitKvAttention). Block (kv_h, chunk) covers one KV head
/// and one kChunk-key slice, and serves all eight query heads of the group:
/// thread (hg, lane) owns query head hg of the group and dims lane + 32k, so
/// every K/V row is staged in LDS once per block instead of once per head.
///
/// Scratch layout (fp32): record r = chunk * num_heads + h, each record is
/// (head_dim + 2) floats: [0] running max m, [1] denominator l, then
/// [2, head_dim + 2) the unnormalized output o relative to m. The sink joins
/// chunk 0 only (m = sink, l = 1), so it counts once in the combine. A chunk
/// that starts past seq_len writes the neutral record (m = -inf, l = 0, o = 0).
template <typename KVT = float>
__global__ void __launch_bounds__(256) FlintDecodeSplitKvGqaKernel(
    const float* __restrict__ q, const KVT* __restrict__ k_cache,
    const KVT* __restrict__ v_cache, float* __restrict__ scratch,
    std::uint32_t layer_idx, std::size_t layer_base,
    const std::uint32_t* __restrict__ d_pos, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    detail::FlintKernelPolicy policy) {
  constexpr std::uint32_t kDim = 128;             // head_dim (launcher-checked)
  constexpr std::uint32_t kGroup = 8;             // query heads per KV head
  constexpr std::uint32_t kLanes = 32;            // lanes per query head
  constexpr std::uint32_t kPerLane = kDim / kLanes;
  constexpr std::uint32_t kTile = 32;             // keys per LDS sub-tile
  constexpr std::uint32_t kThreads = kGroup * kLanes;

  const std::uint32_t kv_h = blockIdx.x;
  const std::uint32_t chunk = blockIdx.y;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t hg = tid / kLanes;
  const std::uint32_t lane = tid % kLanes;
  const std::uint32_t h = kv_h * kGroup + hg;

  const std::uint32_t pos = *d_pos;
  const std::size_t seq_len = static_cast<std::size_t>(pos) + 1;
  const std::size_t p_begin =
      (policy.window != 0 && seq_len > policy.window)
          ? seq_len - policy.window
          : 0;
  if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
    detail::FlintKeyCountAdd(layer_idx,
                             static_cast<std::uint32_t>(seq_len - p_begin));
  }

  constexpr std::size_t kRecord = kDim + 2;
  float* record = scratch + ((static_cast<std::size_t>(chunk) * num_heads) + h) *
                                kRecord;
  const std::size_t chunk_begin =
      p_begin + static_cast<std::size_t>(chunk) * kFlintSplitKvChunk;
  if (chunk_begin >= seq_len) {
    if (lane == 0) {
      record[0] = -INFINITY;
      record[1] = 0.0F;
    }
#pragma unroll
    for (std::uint32_t k = 0; k < kPerLane; ++k) {
      record[2 + lane + (kLanes * k)] = 0.0F;
    }
    return;
  }
  const std::size_t chunk_end =
      std::min(chunk_begin + kFlintSplitKvChunk, seq_len);

  const bool has_sink = policy.sinks != nullptr;
  const float sink = has_sink ? policy.sinks[h] : 0.0F;
  const bool owns_sink = has_sink && chunk == 0;
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dim));

  __shared__ float s_k[kTile * kDim];
  __shared__ float s_v[kTile * kDim];
  __shared__ float s_score[kGroup][kTile];
  __shared__ float s_w[kGroup][kTile];
  __shared__ float s_m[kGroup];
  __shared__ float s_l[kGroup];
  __shared__ float s_alpha[kGroup];

  const float* q_h = q + (static_cast<std::size_t>(h) * kDim);
  float q_reg[kPerLane];
  float o_reg[kPerLane] = {};
#pragma unroll
  for (std::uint32_t k = 0; k < kPerLane; ++k) {
    q_reg[k] = q_h[lane + (kLanes * k)];
  }
  if (lane == 0) {
    s_m[hg] = owns_sink ? sink : -INFINITY;
    s_l[hg] = owns_sink ? 1.0F : 0.0F;
  }

  for (std::size_t t = chunk_begin; t < chunk_end; t += kTile) {
    const std::size_t n = std::min<std::size_t>(kTile, chunk_end - t);

    // S1: stage the K and V sub-tiles once for all eight query heads.
    for (std::size_t i = tid; i < n * kDim; i += kThreads) {
      const std::size_t j = i / kDim;
      const std::size_t d = i % kDim;
      const std::size_t offset =
          FlintKvOffset<KVT>(layer_base, kv_h, t + j, policy.kv_rows,
                             max_context, num_kv_heads, head_dim) +
          d;
      s_k[i] = FlintKvLoad(k_cache + offset);
      s_v[i] = FlintKvLoad(v_cache + offset);
    }
    __syncthreads();

    // Scores: each 32-lane group reduces its head's dot products with xor
    // shuffles (offsets < 32 stay inside the lane group on wave32 and wave64).
    for (std::size_t j = 0; j < n; ++j) {
      float dot = 0.0F;
#pragma unroll
      for (std::uint32_t k = 0; k < kPerLane; ++k) {
        dot += q_reg[k] * s_k[(j * kDim) + lane + (kLanes * k)];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_xor(dot, offset);
      }
      if (lane == 0) {
        s_score[hg][j] = dot * scale;
      }
    }
    __syncthreads();

    // S3 prep: lane 0 of each group runs the online-softmax update for its
    // head. Its (m, l) and the tile's rescale are shared through LDS.
    if (lane == 0) {
      float tile_max = -INFINITY;
      for (std::size_t j = 0; j < n; ++j) {
        tile_max = fmaxf(tile_max, s_score[hg][j]);
      }
      const float m_old = s_m[hg];
      const float m_new = fmaxf(m_old, tile_max);
      const float alpha = expf(m_old - m_new);
      float tile_sum = 0.0F;
      for (std::size_t j = 0; j < n; ++j) {
        const float p = expf(s_score[hg][j] - m_new);
        s_w[hg][j] = p;
        tile_sum += p;
      }
      s_alpha[hg] = alpha;
      s_m[hg] = m_new;
      s_l[hg] = (s_l[hg] * alpha) + tile_sum;
    }
    __syncthreads();

    // O accumulation: rescale, then add this tile's weighted V rows from LDS.
    const float alpha = s_alpha[hg];
#pragma unroll
    for (std::uint32_t k = 0; k < kPerLane; ++k) {
      o_reg[k] *= alpha;
    }
    for (std::size_t j = 0; j < n; ++j) {
      const float w = s_w[hg][j];
#pragma unroll
      for (std::uint32_t k = 0; k < kPerLane; ++k) {
        o_reg[k] += w * s_v[(j * kDim) + lane + (kLanes * k)];
      }
    }
    // S4: the next tile overwrites s_k / s_v.
    __syncthreads();
  }

#pragma unroll
  for (std::uint32_t k = 0; k < kPerLane; ++k) {
    record[2 + lane + (kLanes * k)] = o_reg[k];
  }
  if (lane == 0) {
    record[0] = s_m[hg];
    record[1] = s_l[hg];
  }
}

/// Fixed-order LSE combine of the split-KV partials (no atomics). One block
/// per query head, one thread per output dim; chunks are folded in index
/// order. Neutral records (l == 0) are skipped.
__launch_bounds__(128) __global__ void FlintDecodeSplitKvCombine(
    const float* __restrict__ scratch, float* __restrict__ out_context,
    std::uint32_t num_heads, std::uint32_t num_chunks,
    std::uint32_t head_dim) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t d = threadIdx.x;
  if (h >= num_heads || d >= head_dim) {
    return;
  }
  const std::size_t record_size = static_cast<std::size_t>(head_dim) + 2;
  float running_max = -INFINITY;
  float running_sum = 0.0F;
  float running_out = 0.0F;
  for (std::uint32_t c = 0; c < num_chunks; ++c) {
    const float* record =
        scratch + ((static_cast<std::size_t>(c) * num_heads) + h) * record_size;
    const float chunk_max = record[0];
    const float chunk_sum = record[1];
    if (chunk_sum <= 0.0F) {
      continue;
    }
    const float new_max = fmaxf(running_max, chunk_max);
    const float previous_scale = expf(running_max - new_max);
    const float chunk_scale = expf(chunk_max - new_max);
    running_sum = (running_sum * previous_scale) + (chunk_sum * chunk_scale);
    running_out = (running_out * previous_scale) +
                  (record[2 + d] * chunk_scale);
    running_max = new_max;
  }
  out_context[(static_cast<std::size_t>(h) * head_dim) + d] =
      running_sum > 0.0F ? running_out / running_sum : 0.0F;
}

bool LaunchFlintDecodeSplitKvAttention(
    const float* q, float* k_cache, float* v_cache, void* k_cache_f16,
    void* v_cache_f16, float* out_context, std::uint32_t layer_idx,
    std::size_t layer_base, const std::uint32_t* d_pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, float* scratch,
    hipStream_t stream, const detail::FlintKernelPolicy* policy) {
  if (q == nullptr || out_context == nullptr || d_pos == nullptr ||
      scratch == nullptr || max_context == 0 || num_kv_heads == 0 ||
      num_heads != kFlintSplitKvGroupSize * num_kv_heads || head_dim != 128) {
    return false;
  }
  const bool use_f16 = k_cache_f16 != nullptr && v_cache_f16 != nullptr;
  const bool use_f32 = !use_f16 && k_cache != nullptr && v_cache != nullptr;
  if (!use_f16 && !use_f32) {
    return false;
  }
  const detail::FlintKernelPolicy kernel_policy =
      policy != nullptr ? *policy : detail::FlintKernelPolicy{};
  const std::uint32_t num_chunks =
      (max_context + kFlintSplitKvChunk - 1) / kFlintSplitKvChunk;
  const dim3 grid(num_kv_heads, num_chunks);
  if (use_f16) {
    FlintDecodeSplitKvGqaKernel<__half><<<grid, 256, 0, stream>>>(
        q, static_cast<const __half*>(k_cache_f16),
        static_cast<const __half*>(v_cache_f16), scratch, layer_idx,
        layer_base, d_pos, max_context, num_heads, num_kv_heads, head_dim,
        kernel_policy);
  } else {
    FlintDecodeSplitKvGqaKernel<float><<<grid, 256, 0, stream>>>(
        q, k_cache, v_cache, scratch, layer_idx, layer_base, d_pos,
        max_context, num_heads, num_kv_heads, head_dim, kernel_policy);
  }
  FlintDecodeSplitKvCombine<<<num_heads, 128, 0, stream>>>(
      scratch, out_context, num_heads, num_chunks, head_dim);
  return hipGetLastError() == hipSuccess;
}

void LaunchAttention(const float* q, const float* k, const float* v,
                     const float* gate, float* k_cache, float* v_cache,
                     void* k_cache_f16, void* v_cache_f16, float* out_context,
                     std::uint32_t layer_idx, const std::uint32_t* d_pos,
                     std::uint32_t max_context, std::uint32_t num_heads,
                     std::uint32_t num_kv_heads, std::uint32_t head_dim,
                     hipStream_t stream, bool skip_kv_write,
                     const detail::FlintKernelPolicy* policy,
                     std::size_t layer_base) {
  const std::size_t base =
      layer_base == kUniformKvBase
          ? static_cast<std::size_t>(layer_idx) * num_kv_heads * max_context *
                head_dim
          : layer_base;
  const std::uint32_t rows = policy != nullptr ? policy->kv_rows : 0U;
  // 7.4a (GUFO_FLINT_ATTN_WINDOW_ONLINE, default ON): route the windowed
  // layers to the online-tile kernel. The fixed-LDS ladder costs 0.46 ms per
  // windowed layer per token, flat in depth (7.1 attribution: 16.81 ms/token,
  // 49% of a 512-token step) because its score loop walks one key at a time
  // with a full 256-thread tree reduction per key. The online-tile kernel
  // carries the same window policy, the same fixed grid and fixed LDS, so
  // nothing about graph capture changes; only the softmax accumulation order
  // does (tile-wise online vs per-key tree). UT/G3/GX/G4/G8 all pass with the
  // route on, so it is the default; set GUFO_FLINT_ATTN_WINDOW_ONLINE=0 to
  // fall back to the fixed-LDS ladder.
  static const bool window_online = [] {
    const char* value = std::getenv("GUFO_FLINT_ATTN_WINDOW_ONLINE");
    return value == nullptr || value[0] != '0';
  }();
  if (!skip_kv_write) {
    WriteKVCachePtrKernel<<<num_kv_heads, 256, 0, stream>>>(
        k, v, k_cache, v_cache, static_cast<__half*>(k_cache_f16),
        static_cast<__half*>(v_cache_f16), base, rows, d_pos, max_context,
        num_kv_heads, head_dim);
  }

  // F-D1/F-D2 graph route: the flint policy (sinks + window, fp32 cache
  // plane) takes the fixed-LDS baseline clone — identical arithmetic to the
  // legacy baseline, LDS sized at capture time from the worst-case span
  // (windowed: the sliding window; global: max_context on the ladder).
  // Past the 64 KiB LDS budget (span > 16384) the online-tile kernel takes
  // over, exactly like the legacy path past 16128.
  if (policy != nullptr) {
    const std::size_t span_needed =
        policy->window != 0
            ? std::min<std::size_t>(policy->window, max_context)
            : max_context;
    if (k_cache_f16 != nullptr && v_cache_f16 != nullptr) {
      // f16 KV plane (GUFO_FLINT_KV_F16): the same ladder with f16 loads —
      // half the KV bytes per global-layer decode step, the long-context
      // lever. Arithmetic otherwise identical (f16 values are the rn-rounded
      // fp32 plane values; prefill's WMMA fragments already consume exactly
      // these).
      const auto* k16 = static_cast<const __half*>(k_cache_f16);
      const auto* v16 = static_cast<const __half*>(v_cache_f16);
      if (window_online && span_needed <= 513) {
        detail::EmitAttentionDispatch("decode_online_flint_window_f16", "");
        FlintDecodeOnlineAttentionPtrKernel<__half><<<num_heads, 256, 0, stream>>>(
            q, k16, v16, out_context, layer_idx, base, d_pos, max_context,
            num_heads, num_kv_heads, head_dim, *policy);
        return;
      }
      if (span_needed <= 513) {
        detail::EmitAttentionDispatch("decode_baseline_flint_graph_f16", "");
        AttentionKernelGraph<513, __half><<<num_heads, 256, 0, stream>>>(
            q, gate, k16, v16, out_context, layer_idx, base, d_pos, max_context,
            num_heads, num_kv_heads, head_dim, *policy);
        return;
      }
      if (span_needed <= 4096) {
        detail::EmitAttentionDispatch("decode_baseline_flint_graph_f16", "");
        AttentionKernelGraph<4096, __half><<<num_heads, 256, 0, stream>>>(
            q, gate, k16, v16, out_context, layer_idx, base, d_pos, max_context,
            num_heads, num_kv_heads, head_dim, *policy);
        return;
      }
      // 3.7: the fixed-LDS <8192> tier measured 2.4x SLOWER than the
      // online-tile at 6.6K spans (night-4: doc decode 8.2 vs 19.8 t/s came
      // from -c selecting this tier). Route every span above the <4096>
      // tier to the online-tile so the fast path is the default, not a
      // -c accident.
      detail::EmitAttentionDispatch("decode_online_flint_graph_f16", "");
      FlintDecodeOnlineAttentionPtrKernel<__half><<<num_heads, 256, 0, stream>>>(
          q, k16, v16, out_context, layer_idx, base, d_pos, max_context,
          num_heads, num_kv_heads, head_dim, *policy);
      return;
    }
    if (k_cache != nullptr && v_cache != nullptr) {
      if (window_online && span_needed <= 513) {
        detail::EmitAttentionDispatch("decode_online_flint_window", "");
        FlintDecodeOnlineAttentionPtrKernel<<<num_heads, 256, 0, stream>>>(
            q, k_cache, v_cache, out_context, layer_idx, base, d_pos,
            max_context, num_heads, num_kv_heads, head_dim, *policy);
        return;
      }
      if (span_needed <= 513) {
        detail::EmitAttentionDispatch("decode_baseline_flint_graph", "");
        AttentionKernelGraph<513><<<num_heads, 256, 0, stream>>>(
            q, gate, k_cache, v_cache, out_context, layer_idx, base, d_pos,
            max_context, num_heads, num_kv_heads, head_dim, *policy);
        return;
      }
      if (span_needed <= 4096) {
        detail::EmitAttentionDispatch("decode_baseline_flint_graph", "");
        AttentionKernelGraph<4096><<<num_heads, 256, 0, stream>>>(
            q, gate, k_cache, v_cache, out_context, layer_idx, base, d_pos,
            max_context, num_heads, num_kv_heads, head_dim, *policy);
        return;
      }
      // 3.7: same routing as the f16 branch — the <8192> fixed-LDS tier is
      // 2.4x slower than online-tile at 6.6K spans; online-tile takes over
      // above the <4096> tier.
      // 8192 floats + the 1 KiB reduce row is the static-LDS ceiling (the
      // 16384 tier would exceed the 64 KiB block budget); beyond it the
      // online-tile kernel takes over, as in the legacy path past 16128.
      detail::EmitAttentionDispatch("decode_online_flint_graph", "");
      FlintDecodeOnlineAttentionPtrKernel<<<num_heads, 256, 0, stream>>>(
          q, k_cache, v_cache, out_context, layer_idx, base, d_pos, max_context,
          num_heads, num_kv_heads, head_dim, *policy);
      return;
    }
    detail::EmitAttentionDispatch(
        "none", "decode_online_flint_graph: requires_cache");
    return;
  }

  if (head_dim == 256 && num_kv_heads > 0 && (num_heads % num_kv_heads) == 0) {
    if (k_cache != nullptr && v_cache != nullptr) {
      detail::EmitAttentionDispatch("decode_online_fp32_graph", "");
      QwenDecodeOnlineAttentionPtrKernel<<<num_heads, 32, 0, stream>>>(
          q, gate, k_cache, v_cache, out_context, layer_idx, d_pos, max_context,
          num_heads, num_kv_heads);
    } else {
      detail::EmitAttentionDispatch("decode_online_fp16_graph", "");
      QwenDecodeOnlineAttentionHalfPtrKernel<<<num_heads, 32, 0, stream>>>(
          q, gate, static_cast<const __half*>(k_cache_f16),
          static_cast<const __half*>(v_cache_f16), out_context, layer_idx,
          d_pos, max_context, num_heads, num_kv_heads);
    }
  } else {
    detail::EmitAttentionDispatch(
        "none", "decode_online_fp32_graph: unsupported_shape");
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
