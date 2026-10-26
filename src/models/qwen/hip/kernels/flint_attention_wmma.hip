// Flint-shaped masked prefill attention on the WMMA matrix cores.
//
// Adapted from kernels/attention_wmma.hip (qwen27b 24/4/256) for the flint
// geometry (32/4/128) with the flint policy built in: per-head attention
// sinks (F-D1) and the sliding window (F-D2). The qwen kernel keeps its own
// shape-tuned constants; unifying the two waits until both shapes are
// validated through the same harness (F-K3 records the decision).
//
// Flint's KV plane is fp32 (F-L4 landed it that way; the f16 canonical
// plane is a later bandwidth item). The kernel converts K/V to f16 at the
// fragment load — the same quantize-to-f16 trade the qwen WMMA path makes
// with its Q operands (S = f16(q·scale)·f16(k), tensor-core f32
// accumulate). Deterministic conversion keeps G6 split-exactness; the
// parity class is the usual near-tie envelope (findings 18/19).
//
// Wave/fragment layout is the qwen kernel's, re-derived for head_dim 128:
// 8 waves = 4 S tiles x 2 K-step halves; each wave covers 4 of the 8
// 16-dim K steps for QK and exactly one 16-dim O tile (wave index) for
// every row block. Fragment element mapping (verified against the qwen
// CPU double reference and re-verified here): A holds row L%16 and 16
// contiguous k, B holds column L%16 and 16 contiguous k (transposed tile
// fed as B), C element i is row 2i + L/16, column L%16.
//
// Sink semantics (F-D1): the sink seeds each row's running max and the
// denominator (sum starts at 1), contributes no value. Window (F-D2): the
// block's key sweep starts at the first row's window edge; rows deeper in
// the block mask their own low edge through the softmax `valid` predicate,
// which yields exactly zero probability — skipped and masked tiles are
// arithmetic no-ops, so the sweep bounds never change results.
//
// Chunk-boundary invariance (G6): keys are visited in ascending 16-key
// tiles at a fixed tile width; every row's max/sum/output accumulation
// order depends only on its own visible span, never on the chunk split.
// The flint smoke asserts the split bit-exactness directly.
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/detail/flint_keycount.hpp"

namespace gufo::hip {

GUFO_FLINT_KEYCOUNT_TU(wmma)

namespace {

constexpr std::uint32_t kFHeadDim = 128;
constexpr std::uint32_t kFQueryHeads = 32;
constexpr std::uint32_t kFKvHeads = 4;
constexpr std::uint32_t kFGqa = kFQueryHeads / kFKvHeads;  // 8
constexpr std::uint32_t kFAttnWidth = kFQueryHeads * kFHeadDim;
constexpr std::uint32_t kFKvWidth = kFKvHeads * kFHeadDim;

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f FWmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h FLoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

constexpr std::uint32_t kFWmmaHeads = 2;    // query heads per block
constexpr std::uint32_t kFQueryRows = 32;   // query rows per block
constexpr std::uint32_t kFKeys = 16;        // key tile width
constexpr std::uint32_t kFRowBlocks = (kFQueryRows / 16) * kFWmmaHeads;
constexpr std::uint32_t kFKeyBlocks = kFKeys / 16;
constexpr std::uint32_t kFSTiles = kFRowBlocks * kFKeyBlocks;
constexpr std::uint32_t kFKSteps = kFHeadDim / 16;               // 8
constexpr std::uint32_t kFKStepsPerWave = kFKSteps / 2;          // 4
constexpr std::uint32_t kFDimTilesPerWave = kFKSteps / 8;        // 1
constexpr std::uint32_t kFSoftmaxLanes = 256 / (kFRowBlocks * 16);
constexpr std::uint32_t kFKStride = kFHeadDim + 8;    // 136, 16B-aligned
constexpr std::uint32_t kFVtStride = kFKeys + 8;      // 24
// 1 / sqrt(128): std::sqrt is not constexpr on the device toolchain, so
// square the target and take the reciprocal root as a rational literal.
constexpr float kFAttentionScale = 0.08838834764831845F;

static_assert(kFSTiles == 4, "eight waves cover four S tiles in two halves");
static_assert(kFDimTilesPerWave == 1, "128-dim heads: one O tile per wave");
static_assert(kFKStepsPerWave * 2 == kFKSteps, "k split over two wave halves");
static_assert(kFSoftmaxLanes * (kFKeys / kFSoftmaxLanes) == kFKeys,
              "softmax segmentation");
static_assert(kFKStride % 8 == 0 && kFVtStride % 8 == 0,
              "fragment rows must start on a 16-byte boundary");
static_assert(kFGqa % kFWmmaHeads == 0, "head pairs nest inside GQA groups");

constexpr std::uint32_t kFVRegs = (kFKeys * kFHeadDim) / (256 * 8);
constexpr std::uint32_t kFKRegs = (kFKeys * (kFHeadDim / 8)) / 256;
static_assert(kFKRegs * 256 == kFKeys * (kFHeadDim / 8), "K stages evenly");
static_assert(kFVRegs * 256 * 8 == kFKeys * kFHeadDim, "V stages evenly");

constexpr std::uint32_t kFKvLdsHalves =
    2 * (kFKeys * kFKStride) + kFHeadDim * kFVtStride;
// Layout: [0, 16*136) K hi, [16*136, 2*16*136) K lo, then V^T. The K lo
// plane carries the f16 residual (x - f16(x)): QK runs two WMMA
// accumulations so the score error stays at the Q-only quantization
// level (see FPack8Lo).
constexpr std::uint32_t kFKHiHalves = kFKeys * kFKStride;

// 8 halves packed from 8 fp32 source values (2 x float4). K and V arrive
// from flint's fp32 KV plane; the fragment/LDS staging stays f16.
// FPack8HiLo additionally produces the f16 residual (x - f16(x)): the
// two-term K decomposition keeps the WMMA score error at the Q-only
// quantization level instead of Q+K. The attention output error feeds 49
// layers of router near-ties — the G4 cjk fork (rank #3 at 5.5 nats) was
// traced to the single-f16 K quantization.
__device__ __forceinline__ uint4 FPack8(const float* p) {
  union {
    uint4 u;
    __half h[8];
  } cvt;
  const float4 f0 = *reinterpret_cast<const float4*>(p);
  const float4 f1 = *reinterpret_cast<const float4*>(p + 4);
  cvt.h[0] = __float2half_rn(f0.x);
  cvt.h[1] = __float2half_rn(f0.y);
  cvt.h[2] = __float2half_rn(f0.z);
  cvt.h[3] = __float2half_rn(f0.w);
  cvt.h[4] = __float2half_rn(f1.x);
  cvt.h[5] = __float2half_rn(f1.y);
  cvt.h[6] = __float2half_rn(f1.z);
  cvt.h[7] = __float2half_rn(f1.w);
  return cvt.u;
}

__device__ __forceinline__ uint4 FPack8Lo(const float* p) {
  union {
    uint4 u;
    __half h[8];
  } cvt;
  const float4 f0 = *reinterpret_cast<const float4*>(p);
  const float4 f1 = *reinterpret_cast<const float4*>(p + 4);
  const float vals[8] = {f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w};
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    const __half hi = __float2half_rn(vals[i]);
    cvt.h[i] = __float2half_rn(vals[i] - __half2float(hi));
  }
  return cvt.u;
}

}  // namespace

__global__ void __launch_bounds__(256, 2) FlintWmmaCausalAttention(
    const float* __restrict__ q, const float* __restrict__ k_cache,
    const float* __restrict__ v_cache, float* __restrict__ out_context,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, detail::FlintKernelPolicy policy) {
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  const std::uint32_t query_start = blockIdx.x * kFQueryRows;
  const std::uint32_t head_pair = blockIdx.y;
  const std::uint32_t kv_head = head_pair / (kFGqa / kFWmmaHeads);
  const std::uint32_t pair_in_group = head_pair % (kFGqa / kFWmmaHeads);
  const std::uint32_t first_query_head =
      (kv_head * kFGqa) + (pair_in_group * kFWmmaHeads);

  const auto row_block_head = [&](std::uint32_t rb) {
    return first_query_head + (rb % kFWmmaHeads);
  };
  const auto row_block_offset = [&](std::uint32_t rb) {
    return (rb / kFWmmaHeads) * 16;
  };

  __shared__ __half kv_lds[kFKvLdsHalves];
  __shared__ float s_lds[2][kFSTiles][16][17];
  __shared__ __half p_lds[kFRowBlocks * 16][kFKeys + 8];
  __shared__ float row_max[kFRowBlocks * 16];
  __shared__ float row_sum[kFRowBlocks * 16];
  __shared__ float row_scale[kFRowBlocks * 16];

  const std::uint32_t s_tile = wave % kFSTiles;
  const std::uint32_t s_kh = wave / kFSTiles;
  const std::uint32_t s_rb = s_tile % kFRowBlocks;
  const std::uint32_t s_kb = s_tile / kFRowBlocks;

  v16h q_frag[kFKStepsPerWave];
  {
    const std::uint32_t local_query =
        query_start + row_block_offset(s_rb) + sub;
    const bool live = local_query < batch_size;
    const float* q_row =
        q + (static_cast<std::size_t>(local_query) * kFAttnWidth) +
        (static_cast<std::size_t>(row_block_head(s_rb)) * kFHeadDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kFKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kFKStepsPerWave) + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][(v * 4) + 0] =
            static_cast<_Float16>(f.x * kFAttentionScale);
        q_frag[ks][(v * 4) + 1] =
            static_cast<_Float16>(f.y * kFAttentionScale);
        q_frag[ks][(v * 4) + 2] =
            static_cast<_Float16>(f.z * kFAttentionScale);
        q_frag[ks][(v * 4) + 3] =
            static_cast<_Float16>(f.w * kFAttentionScale);
      }
    }
  }

  v8f o_acc[kFRowBlocks][kFDimTilesPerWave] = {};
  // The sink seeds the max and the denominator (F-D1): sum starts at 1 for
  // exp(sink - sink). Every thread carries its softmax row's stats; the
  // lanes of a row keep them identical via the shuffle reductions.
  const std::uint32_t init_rg = tid / kFSoftmaxLanes;
  const std::uint32_t init_rb = init_rg / 16;
  float running_max = -INFINITY;
  float running_sum = 0.0F;
  if (policy.sinks != nullptr) {
    running_max = policy.sinks[row_block_head(init_rb)];
    running_sum = 1.0F;
  }

  const std::uint32_t context_end =
      start_pos + static_cast<std::uint32_t>(batch_size);
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kFQueryRows);
  // Window: the block's earliest row edge is the first query row's
  // p_begin; deeper rows mask through the softmax predicate. The sweep
  // start is FLOORED to the 16-key tile grid: off-grid starts would shift
  // every tile's key grouping and change f32 summation order (the G6
  // multi-split test caught exactly that). Leading keys below the first
  // row's edge are fully masked — exact no-ops — so flooring never moves
  // a result.
  const std::uint32_t sweep_begin =
      policy.window != 0
          ? ((start_pos + query_start + 1 > policy.window
                  ? start_pos + query_start + 1 - policy.window
                  : 0) &
             ~(kFKeys - 1u))
          : 0;
  if (blockIdx.y == 0 && threadIdx.x == 0) {
    detail::FlintKeyCountAdd(layer_idx, max_visible - sweep_begin);
  }

  constexpr std::uint32_t kVRegs = kFVRegs;
  const std::uint32_t v_key = lane % kFKeys;
  const std::uint32_t v_slice = (tid / kFKeys) * (kVRegs * 8);
  // Flint's fp32 KV plane is head-major: [layer][kv_head][pos][dim] —
  // positions are contiguous within a (layer, kv_head) plane.
  constexpr std::size_t kv_stride = kFHeadDim;
  const std::size_t head_offset =
      (static_cast<std::size_t>(layer_idx) * kFKvHeads + kv_head) *
      static_cast<std::size_t>(max_context) * kFHeadDim;
  const auto* v_base = v_cache + head_offset + v_slice;
  const auto* k_base = k_cache + head_offset;

  const auto load_v = [&](std::uint32_t key_start, uint4* dst) {
    const std::uint32_t key_position = key_start + v_key;
    const auto* src =
        v_base + (static_cast<std::size_t>(key_position) * kv_stride);
    const bool live = key_position < context_end;
#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j) {
      dst[j] = live ? FPack8(src + (j * 8)) : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  const auto load_k = [&](std::uint32_t key_start, uint4* dst,
                          uint4* dst_lo) {
#pragma unroll
    for (std::uint32_t n = 0; n < kFKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_position = key_start + (idx / (kFHeadDim / 8));
      const std::uint32_t d8 = (idx % (kFHeadDim / 8)) * 8;
      const bool live = key_position < context_end;
      dst[n] = live ? FPack8(k_base +
                             (static_cast<std::size_t>(key_position) *
                              kv_stride) +
                             d8)
                    : make_uint4(0u, 0u, 0u, 0u);
      dst_lo[n] = live ? FPack8Lo(k_base +
                                  (static_cast<std::size_t>(key_position) *
                                   kv_stride) +
                                  d8)
                       : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  uint4 k_cur[kFKRegs];
  uint4 v_cur[kVRegs];
  uint4 k_lo_cur[kFKRegs];
  uint4 k_pre[kFKRegs] = {};
  uint4 k_lo_pre[kFKRegs] = {};
  uint4 v_pre[kVRegs] = {};
  load_k(sweep_begin, k_cur, k_lo_cur);
  load_v(sweep_begin, v_cur);

  for (std::uint32_t key_start = sweep_begin; key_start < max_visible;
       key_start += kFKeys) {
    // --- stage K hi/lo from the registers the previous iteration prefetched
    __syncthreads();
#pragma unroll
    for (std::uint32_t n = 0; n < kFKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_row = idx / (kFHeadDim / 8);
      const std::uint32_t d8 = (idx % (kFHeadDim / 8)) * 8;
      *reinterpret_cast<uint4*>(&kv_lds[(key_row * kFKStride) + d8]) =
          k_cur[n];
      *reinterpret_cast<uint4*>(
          &kv_lds[kFKHiHalves + (key_row * kFKStride) + d8]) = k_lo_cur[n];
    }
    __syncthreads();

    // --- prefetch the next key tile
    const std::uint32_t next_start = key_start + kFKeys;
    if (next_start < max_visible) {
      load_k(next_start, k_pre, k_lo_pre);
      load_v(next_start, v_pre);
    }

    // --- S = Q (K_hi + K_lo)^T — two accumulations keep the score error
    // at the Q-only f16 quantization level
    {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kFKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kFKStepsPerWave) + ks) * 16;
        const v16h k_hi = FLoadFrag(
            &kv_lds[(((s_kb * 16) + sub) * kFKStride) + d0]);
        s_acc = FWmma(q_frag[ks], k_hi, s_acc);
        const v16h k_lo = FLoadFrag(
            &kv_lds[kFKHiHalves + (((s_kb * 16) + sub) * kFKStride) + d0]);
        s_acc = FWmma(q_frag[ks], k_lo, s_acc);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
      }
    }
    __syncthreads();

    // --- online softmax: kFSoftmaxLanes threads per query row
    {
      const std::uint32_t rg = tid / kFSoftmaxLanes;
      const std::uint32_t seg = tid % kFSoftmaxLanes;
      constexpr std::uint32_t kPerLane = kFKeys / kFSoftmaxLanes;
      const std::uint32_t rb = rg / 16;
      const std::uint32_t row = rg % 16;
      const std::uint32_t local_query =
          query_start + row_block_offset(rb) + row;
      const std::uint32_t absolute_query = start_pos + local_query;
      const std::uint32_t window_edge =
          policy.window != 0
              ? (absolute_query + 1 > policy.window
                     ? absolute_query + 1 - policy.window
                     : 0)
              : 0;
      float part_max = -INFINITY;
      float vals[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = (seg * kPerLane) + m;
        const std::uint32_t key_position = key_start + col;
        const bool valid = local_query < batch_size &&
                           key_position <= absolute_query &&
                           key_position < context_end &&
                           key_position >= window_edge;
        const std::uint32_t tile = ((col / 16) * kFRowBlocks) + rb;
        vals[m] = valid ? (s_lds[0][tile][row][col % 16] +
                           s_lds[1][tile][row][col % 16])
                        : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kFSoftmaxLanes; off <<= 1) {
        part_max = fmaxf(part_max, __shfl_xor(part_max, off));
      }
      const float prev_max = running_max;
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        p_lds[rg][(seg * kPerLane) + m] = static_cast<__half>(w);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kFSoftmaxLanes; off <<= 1) {
        part_sum += __shfl_xor(part_sum, off);
      }
      running_max = next_max;
      running_sum = (running_sum * prior_scale) + part_sum;
      if (seg == 0) {
        row_scale[rg] = prior_scale;
      }
    }
    __syncthreads();

    // --- rescale O, stage V transposed, O += P V
#pragma unroll
    for (std::uint32_t rb = 0; rb < kFRowBlocks; ++rb) {
      float scale[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        scale[i] = row_scale[(rb * 16) + (2 * i) + half_id];
      }
#pragma unroll
      for (std::uint32_t t = 0; t < kFDimTilesPerWave; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[rb][t][i] *= scale[i];
        }
      }
    }

#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j) {
      const auto* packed = reinterpret_cast<const __half*>(&v_cur[j]);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        kv_lds[((v_slice + (j * 8) + i) * kFVtStride) + v_key] = packed[i];
      }
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t rb = 0; rb < kFRowBlocks; ++rb) {
#pragma unroll
      for (std::uint32_t t = 0; t < kFDimTilesPerWave; ++t) {
        const std::uint32_t dim_tile = wave + (t * 8);
        v16h v_frag[kFKeyBlocks];
#pragma unroll
        for (std::uint32_t kb = 0; kb < kFKeyBlocks; ++kb) {
          v_frag[kb] =
              FLoadFrag(&kv_lds[(dim_tile * 16 + sub) * kFVtStride + kb * 16]);
        }
#pragma unroll
        for (std::uint32_t kb = 0; kb < kFKeyBlocks; ++kb) {
          const v16h p_frag = FLoadFrag(&p_lds[(rb * 16) + sub][kb * 16]);
          v8f next = FWmma(p_frag, v_frag[kb], o_acc[rb][t]);
          const auto first_key = key_start + kb * 16;
          // Final-partial-tile correction (as in the qwen kernel): sum a
          // query's last partial tile over visible keys only — WMMA can
          // round differently when a zero probability meets a populated
          // value. Window-masked keys need no counterpart: their stored
          // probability is exactly 0, an exact +0 contribution.
          if (first_key + 15 > start_pos + query_start) {
            v8f tail = o_acc[rb][t];
            // Keep the position difference uniform across the wave; only the
            // even/odd row owner varies between its two half-waves.
            const int relative_key = static_cast<int>(first_key) -
                                     static_cast<int>(start_pos + query_start +
                                                      row_block_offset(rb));
#pragma unroll
            for (std::uint32_t key = 0; key < 16; ++key) {
              const float value = static_cast<float>(v_frag[kb][key]);
#pragma unroll
              for (std::uint32_t i = 0; i < 8; ++i) {
                if (static_cast<int>(half_id) <
                    relative_key + 15 - static_cast<int>(2 * i)) {
                  tail[i] = fmaf(__half2float(
                                     p_lds[rb * 16 + (2 * i) + half_id]
                                          [kb * 16 + key]),
                                 value, tail[i]);
                }
              }
            }
#pragma unroll
            for (std::uint32_t i = 0; i < 8; ++i) {
              if (static_cast<int>(half_id) <
                  relative_key + 15 - static_cast<int>(2 * i)) {
                next[i] = tail[i];
              }
            }
          }
          o_acc[rb][t] = next;
        }
      }
    }

#pragma unroll
    for (std::uint32_t n = 0; n < kFKRegs; ++n) {
      k_cur[n] = k_pre[n];
      k_lo_cur[n] = k_lo_pre[n];
    }
#pragma unroll
    for (std::uint32_t n = 0; n < kVRegs; ++n) {
      v_cur[n] = v_pre[n];
    }
  }
  if (tid % kFSoftmaxLanes == 0) {
    const std::uint32_t rg = tid / kFSoftmaxLanes;
    row_max[rg] = running_max;
    row_sum[rg] = running_sum;
  }
  __syncthreads();

  // --- epilogue: each wave writes its own 16-dim slice of every row block
#pragma unroll
  for (std::uint32_t rb = 0; rb < kFRowBlocks; ++rb) {
    const std::uint32_t query_head = row_block_head(rb);
    const std::uint32_t row_offset = row_block_offset(rb);
#pragma unroll
    for (std::uint32_t t = 0; t < kFDimTilesPerWave; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        const std::uint32_t local_query = query_start + row_offset + row;
        if (local_query >= batch_size) {
          continue;
        }
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(local_query) * kFAttnWidth) +
            (static_cast<std::size_t>(query_head) * kFHeadDim) +
            (dim_tile * 16) + sub;
        const float value =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
        out_context[offset] = value;
      }
    }
  }
}

bool LaunchFlintWmmaAttention(
    const float* q, const float* k_cache, const float* v_cache,
    float* out_context, std::uint32_t layer_idx, std::uint32_t start_pos,
    std::size_t batch_size, std::uint32_t max_context, hipStream_t stream,
    const detail::FlintKernelPolicy* policy) {
  if (batch_size == 0 || k_cache == nullptr || v_cache == nullptr) {
    return false;
  }
  if (policy == nullptr) {
    return false;  // flint always carries sinks/window through the policy
  }
  if (static_cast<std::size_t>(start_pos) + batch_size > max_context) {
    return false;
  }
  detail::EmitAttentionDispatch("flint_prefill_wmma", "");
  const dim3 grid((static_cast<unsigned>(batch_size) + kFQueryRows - 1) /
                      kFQueryRows,
                  kFKvHeads * (kFGqa / kFWmmaHeads));
  FlintWmmaCausalAttention<<<grid, 256, 0, stream>>>(
      q, k_cache, v_cache, out_context, layer_idx, start_pos, batch_size,
      max_context, *policy);
  return true;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
