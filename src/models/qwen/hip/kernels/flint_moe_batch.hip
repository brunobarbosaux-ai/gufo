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
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {
namespace {

constexpr int kFlintBatchWarps = 4;        // warps per expert block
constexpr int kFlintBatchRowsPerWarp = 2;  // weight rows per warp
constexpr int kFlintBatchTokens = 8;       // tokens sharing one weight read
constexpr int kFlintRouterWarps = 8;       // experts per router block
constexpr int kFlintRouterTokens = 8;      // tokens per router block

/// WMMA token tile: expert buckets are padded to a multiple of this so one
/// activation tile never straddles two experts (F-L4 grouped WMMA route).
constexpr int kFlintSlotTile = 16;

/// Tiles processed per grouped-WMMA block (7.5d). Expert buckets are emitted
/// as runs of 16-slot tiles; a block claims a run of up to this many tiles so
/// one decoded weight block feeds that many activation tiles, cutting the
/// per-expert weight re-read (the routed MoE is DRAM-bound at chunk sizes).
constexpr int kFlintGroupTiles = 2;

// ---------------------------------------------------------------------------
// Router: logits[N, E] = x[N, H] * W[E, H]^T for the whole chunk.
// ---------------------------------------------------------------------------

/// Tiled router GEMM (Phase 2 item: router on the fast path). The warp-per-
/// expert scalar kernel below runs ~1.7 ms per layer at 512 tokens (~13% of
/// the chunk) because each warp issues nine global loads per eight FMAs with
/// no reuse. This kernel stages 32-token x and 32-expert w tiles in LDS with
/// a 2x2 register tile per thread, turning the nine global loads into four
/// broadcast LDS reads and reusing every staged value 32 times.
///
/// Numerics match the decode GEMV's operand semantics exactly (fp32 products
/// of the stored weight and activation, fp32 accumulation) and each output
/// accumulates over k in one fixed sequential order that does not depend on
/// the chunk size, so split and single-chunk runs stay bit-identical (G6).
///
/// Out-of-range rows load as zero and masked stores keep partial tiles safe;
/// the launcher only takes hidden % 64 == 0 (the staged float4 loads) and
/// falls back to the scalar kernel otherwise. The LDS tiles are k-major
/// [k][row] with pitch 34: every thread's pair of values is one aligned
/// float2 read (conflict-free; the staged stores pay at most a 4-way).
template <typename T>
__global__ void __launch_bounds__(256, 3)
    FlintRouterTiledKernel(const float* __restrict__ x, int n_tokens,
                           const T* __restrict__ w, int expert_count,
                           int hidden, float* __restrict__ logits) {
  constexpr int kTile = 32;          // experts and tokens per block tile
  constexpr int kBK = 64;            // k elements per LDS stage
  constexpr int kPitch = kTile + 2;  // float2-aligned, bank-spreading pitch
  __shared__ float s_x[kBK][kPitch];  // [k][token]
  __shared__ float s_w[kBK][kPitch];  // [k][expert]

  const int tid = static_cast<int>(threadIdx.x);
  const int tx = tid & 15;  // thread tile column: expert pair of 2
  const int ty = tid >> 4;  // thread tile row: token pair of 2
  const int e0 = static_cast<int>(blockIdx.x) * kTile;
  const int t0 = static_cast<int>(blockIdx.y) * kTile;

  float acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      acc[i][j] = 0.0F;
    }
  }

  for (int k0 = 0; k0 < hidden; k0 += kBK) {
    __syncthreads();
    // Stage x: 32 tokens x 64 fp32 as float4. Threads walk rows of 16
    // float4 (coalesced); the stores transpose into the k-major tile.
    // (The launcher only takes hidden % 64 == 0, so tiles are never
    // partial in k.)
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int idx = tid + p * 256;  // float4 index in the tile
      const int t = idx >> 4;
      const int c4 = (idx & 15) << 2;
      float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      if (t0 + t < n_tokens) {
        v = *reinterpret_cast<const float4*>(&
            x[static_cast<std::size_t>(t0 + t) * hidden + k0 + c4]);
      }
      s_x[c4 + 0][t] = v.x;
      s_x[c4 + 1][t] = v.y;
      s_x[c4 + 2][t] = v.z;
      s_x[c4 + 3][t] = v.w;
    }
    // Stage w: 32 experts x 64 weight row elements, converted to fp32 once
    // here so the compute loop has no conversions.
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int idx = tid + p * 256;  // weight-x4 index in the tile
      const int e = idx >> 4;
      const int c4 = (idx & 15) << 2;
      float v[4] = {0.0F, 0.0F, 0.0F, 0.0F};
      if (e0 + e < expert_count) {
        const T* row = w + static_cast<std::size_t>(e0 + e) * hidden + k0 + c4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          v[j] = static_cast<float>(row[j]);
        }
      }
      s_w[c4 + 0][e] = v[0];
      s_w[c4 + 1][e] = v[1];
      s_w[c4 + 2][e] = v[2];
      s_w[c4 + 3][e] = v[3];
    }
    __syncthreads();

    for (int k = 0; k < kBK; ++k) {
      // One aligned float2 per operand per k: the w pair broadcasts across
      // the lanes that share tx, the x pair across the lanes that share ty.
      const float2 wv = *reinterpret_cast<const float2*>(&s_w[k][2 * tx]);
      const float2 xv = *reinterpret_cast<const float2*>(&s_x[k][2 * ty]);
      acc[0][0] += wv.x * xv.x;
      acc[1][0] += wv.y * xv.x;
      acc[0][1] += wv.x * xv.y;
      acc[1][1] += wv.y * xv.y;
    }
  }

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int e = e0 + 2 * tx + i;
    if (e >= expert_count) {
      continue;
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int t = t0 + 2 * ty + j;
      if (t < n_tokens) {
        logits[static_cast<std::size_t>(t) * expert_count + e] = acc[i][j];
      }
    }
  }
}

/// Scalar fallback for shapes the tiled kernel cannot take (a hidden size
/// that is not a multiple of 64). Warp per expert x 8 tokens.
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

// Bucket starts are padded to the 16-row WMMA token tile (F-L4 tuning) so a
// tile never straddles two experts: the grouped WMMA kernels address their
// activation tiles directly by `slot0 / 16`. The same single-threaded pass
// emits the tile table the grouped kernels index (tile -> expert, slot base)
// and zeroes the padding rows of the slot tables so the kernels that still
// walk buckets by rows (the scalar grouped pair) read row 0 instead of an
// uninitialized index.
__global__ void FlintMoeOffsetsKernel(const int* __restrict__ counts,
                                      int* __restrict__ offsets,
                                      int* __restrict__ cursors,
                                      int* __restrict__ slot_tokens,
                                      float* __restrict__ slot_weights,
                                      int* __restrict__ tile_expert,
                                      int* __restrict__ tile_slot0,
                                      int* __restrict__ tile_count,
                                      int expert_count) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int running = 0;
  int tiles = 0;
  for (int e = 0; e < expert_count; ++e) {
    const int c = counts[e];
    const int padded = (c + kFlintSlotTile - 1) & ~(kFlintSlotTile - 1);
    offsets[e] = running;
    cursors[e] = running;
    for (int s = running + c; s < running + padded; ++s) {
      slot_tokens[s] = 0;
      slot_weights[s] = 0.0F;
    }
    for (int lt = 0; lt < padded; lt += kFlintSlotTile) {
      tile_expert[tiles] = e;
      tile_slot0[tiles] = running + lt;
      ++tiles;
    }
    running += padded;
  }
  offsets[expert_count] = running;
  tile_count[0] = tiles;
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
                              core::GgmlType type, core::GgmlType up_type,
                              const float* __restrict__ x,
                              float* __restrict__ act,
                              const int* __restrict__ offsets,
                              const int* __restrict__ slot_tokens,
                              int expert_ff, int hidden,
                              std::size_t slab_bytes, std::size_t slab_up) {
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
                      static_cast<std::size_t>(e) * slab_up;
  const std::size_t row_bytes = QuantRowBytes(type, static_cast<std::size_t>(hidden));
  const std::size_t row_bytes_u =
      QuantRowBytes(up_type, static_cast<std::size_t>(hidden));
  const char* grow0 = gbase + static_cast<std::size_t>(r0) * row_bytes;
  const char* urow0 = ubase + static_cast<std::size_t>(r0) * row_bytes_u;
  const char* grow1 = has_r1
                          ? gbase + static_cast<std::size_t>(r0 + 1) * row_bytes
                          : nullptr;
  const char* urow1 = has_r1
                          ? ubase + static_cast<std::size_t>(r0 + 1) * row_bytes_u
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
    FlintDotRow(up_type, urow0, lane, x, hidden, toks, tn, hidden, au0);
    if (grow1 != nullptr) {
      FlintDotRow(type, grow1, lane, x, hidden, toks, tn, hidden, ag1);
      FlintDotRow(up_type, urow1, lane, x, hidden, toks, tn, hidden, au1);
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

// ---------------------------------------------------------------------------
// Grouped WMMA route (F-L4 tuning).
//
// The scalar grouped kernels above decode each weight row with scalar FMAs,
// which held the routed MoE at ~2 TFLOP/s while every projection in the same
// chunk already runs on int8 tensor cores. These kernels move the expert
// buckets onto the same WMMA arithmetic as WKQuantA8BlockedWmmaGEMMKernel,
// specialized to ONE 16-token activation tile (BN = 16): bucket starts are
// padded to the tile, so a block owns (tile, weight rows, expert) and reads
// its activation tile contiguously. The tile table comes from the compaction
// pass; blocks past the device-side tile count exit.
//
// Pad rows of a bucket are never zeroed in the Q8_1 staging (they hold stale
// rows from the previous layer); their outputs are never read — the combine
// walks token_slots, which only references real slots.
// ---------------------------------------------------------------------------

/// Copies the tiled Q8_1 rows of `x_q8` (token-major) into slot order:
/// dst tile (slot/16, kb) row (slot%16) <- src tile (token/16, kb) row
/// (token%16). One thread per (slot, k-block).
__global__ void FlintGatherQ8ActKernel(const int* __restrict__ slot_tokens,
                                       const int* __restrict__ offsets,
                                       int expert_count,
                                       const std::int8_t* __restrict__ src,
                                       std::int8_t* __restrict__ dst,
                                       int num_blocks) {
  const int s = static_cast<int>(blockIdx.x);
  if (s >= offsets[expert_count]) {
    return;
  }
  const int tok = slot_tokens[s];
  const int kb = static_cast<int>(threadIdx.x);
  if (kb >= num_blocks) {
    return;
  }
  const std::size_t stt = static_cast<std::size_t>(tok) >> 4;
  const std::size_t stl = static_cast<std::size_t>(tok) & 15;
  const std::size_t dtt = static_cast<std::size_t>(s) >> 4;
  const std::size_t dtl = static_cast<std::size_t>(s) & 15;
  const std::int8_t* s_tile = src + ((stt * num_blocks + kb) * kQ8ActTileBytes);
  std::int8_t* d_tile = dst + ((dtt * num_blocks + kb) * kQ8ActTileBytes);
  reinterpret_cast<int32x4_t*>(d_tile + dtl * 16)[0] =
      reinterpret_cast<const int32x4_t*>(s_tile + stl * 16)[0];
  reinterpret_cast<int32x4_t*>(d_tile + 256 + dtl * 16)[0] =
      reinterpret_cast<const int32x4_t*>(s_tile + 256 + stl * 16)[0];
  reinterpret_cast<float*>(d_tile + kQ8ActScaleOffset)[dtl] =
      reinterpret_cast<const float*>(s_tile + kQ8ActScaleOffset)[stl];
}

/// Grouped gate/up GEMM over the expert buckets on WMMA int8 tensor cores:
/// gate_out[slot, r] = gate_e[r] . x[slot], up likewise, for the 16 slots of
/// the block's tile. Mirrors WKQuantA8BlockedWmmaGEMMKernel's stage/accumulate
/// structure with BM=128 rows x BN=16 slots, 8 waves (one 16-row tile each),
/// BK 32-element K blocks per stage.
template <core::GgmlType WType>
__launch_bounds__(256, 2) __global__ void FlintMoeGroupedGateUpWmmaKernel(
    const void* __restrict__ gate_stack, const void* __restrict__ up_stack,
    const void* __restrict__ slot_q8, float* __restrict__ gate_out,
    float* __restrict__ up_out, const int* __restrict__ offsets,
    const int* __restrict__ tile_expert, const int* __restrict__ tile_slot0,
    const int* __restrict__ tile_count, int expert_ff, int hidden,
    std::size_t slab_bytes) {
  constexpr int BM = 128;
  constexpr int BK = 2;
  constexpr int G = kFlintGroupTiles;
  constexpr int kRowTiles = BM / 16;
  constexpr bool PerHalfScale =
      WType == core::GgmlType::kQ6_K || WType == core::GgmlType::kQ3_K;
  constexpr int kScaleHalves = PerHalfScale ? 2 : 1;

  const int t = static_cast<int>(blockIdx.x);
  if (t >= tile_count[0]) {
    return;
  }
  const int e = tile_expert[t];
  // Tiles are expert-sorted. A block claims a group of up to G consecutive
  // tiles of one expert: the group leader is the tile whose offset within the
  // expert's run is a multiple of G. One decoded weight block then feeds G
  // activation tiles, cutting the per-expert weight re-read.
  int run_start = t;
  while (run_start > 0 && tile_expert[run_start - 1] == e) {
    --run_start;
  }
  if (((t - run_start) % G) != 0) {
    return;
  }
  int ntiles = 1;
#pragma unroll
  for (int j = 1; j < G; ++j) {
    if (t + j < tile_count[0] && tile_expert[t + j] == e) {
      ++ntiles;
    }
  }
  const std::size_t slot0 = static_cast<std::size_t>(tile_slot0[t]);
  const std::size_t r_block = static_cast<std::size_t>(blockIdx.y) * BM;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;

  const std::size_t num_blocks = static_cast<std::size_t>(hidden) / 32;
  const std::size_t row_bytes = QuantRowBytes(WType, hidden);
  const std::uint8_t* grow = static_cast<const std::uint8_t*>(gate_stack) +
                             static_cast<std::size_t>(e) * slab_bytes;
  const std::uint8_t* urow = static_cast<const std::uint8_t*>(up_stack) +
                             static_cast<std::size_t>(e) * slab_bytes;

  __shared__ int32x4_t s_g[BK][kRowTiles][32];
  __shared__ int32x4_t s_u[BK][kRowTiles][32];
  __shared__ float s_dwg[BK][kScaleHalves][kRowTiles][16];
  __shared__ float s_dwu[BK][kScaleHalves][kRowTiles][16];
  __shared__ int32x4_t s_b[BK][G][32];
  __shared__ float s_dx[BK][G][16];

  // One (row, kb) weight unit per thread: BM * BK == 256 threads.
  const int rr = tid / BK;
  const int kk = tid % BK;
  const int rl = rr % 16;
  const int scale_slot = (rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2));
  const std::uint8_t* gline =
      grow + (r_block + static_cast<std::size_t>(rr)) * row_bytes;
  const std::uint8_t* uline =
      urow + (r_block + static_cast<std::size_t>(rr)) * row_bytes;

  // Wave 0 stages the group's token tiles (32 lanes cover both k-half runs).
  const std::int8_t* b_base =
      static_cast<const std::int8_t*>(slot_q8) +
      ((slot0 >> 4) * num_blocks * kQ8ActTileBytes);

  float acc_g[G][8];
  float acc_u[G][8];
#pragma unroll
  for (int j = 0; j < G; ++j) {
#pragma unroll
    for (int l = 0; l < 8; ++l) {
      acc_g[j][l] = 0.0F;
      acc_u[j][l] = 0.0F;
    }
  }

  int32x4_t r_qg0, r_qg1, r_qu0, r_qu1;
  float r_dwg0, r_dwg1, r_dwu0, r_dwu1;
  int32x4_t r_b[G][BK];
  float r_dx[G][BK];

  const auto fetch_stage = [&](int kb0) {
    const int kb = kb0 + kk;
    QuantSub16 lo, hi;
    DecodeQuantSub16(WType, gline, static_cast<std::size_t>(kb) * 2, lo);
    DecodeQuantSub16(WType, gline, static_cast<std::size_t>(kb) * 2 + 1, hi);
    __builtin_memcpy(&r_qg0, lo.q, 16);
    __builtin_memcpy(&r_qg1, hi.q, 16);
    r_dwg0 = lo.scale;
    r_dwg1 = hi.scale;
    DecodeQuantSub16(WType, uline, static_cast<std::size_t>(kb) * 2, lo);
    DecodeQuantSub16(WType, uline, static_cast<std::size_t>(kb) * 2 + 1, hi);
    __builtin_memcpy(&r_qu0, lo.q, 16);
    __builtin_memcpy(&r_qu1, hi.q, 16);
    r_dwu0 = lo.scale;
    r_dwu1 = hi.scale;
    if (wave_id == 0) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
#pragma unroll
        for (int j = 0; j < G; ++j) {
          if (j >= ntiles) {
            continue;
          }
          const std::int8_t* tile =
              b_base + ((static_cast<std::size_t>(j) * num_blocks) + kb0 + i) *
                           kQ8ActTileBytes;
          r_b[j][i] = reinterpret_cast<const int32x4_t*>(tile)[lane_id];
          r_dx[j][i] = reinterpret_cast<const float*>(
              tile + kQ8ActScaleOffset)[lane_id & 15];
        }
      }
    }
  };

  const auto commit_stage = [&]() {
    s_g[kk][rr / 16][rl] = r_qg0;
    s_g[kk][rr / 16][16 + rl] = r_qg1;
    s_u[kk][rr / 16][rl] = r_qu0;
    s_u[kk][rr / 16][16 + rl] = r_qu1;
    s_dwg[kk][0][rr / 16][scale_slot] = r_dwg0;
    s_dwu[kk][0][rr / 16][scale_slot] = r_dwu0;
    if constexpr (PerHalfScale) {
      s_dwg[kk][1][rr / 16][scale_slot] = r_dwg1;
      s_dwu[kk][1][rr / 16][scale_slot] = r_dwu1;
    }
    if (wave_id == 0) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
#pragma unroll
        for (int j = 0; j < G; ++j) {
          if (j >= ntiles) {
            continue;
          }
          s_b[i][j][lane_id] = r_b[j][i];
          if (lane_id < 16) {
            s_dx[i][j][lane_id] = r_dx[j][i];
          }
        }
      }
    }
  };

  const int num_kb = static_cast<int>(num_blocks);
  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      const int rs = wave_id;
      const int32x4_t ag0 = s_g[kb][rs][sub_lane];
      const int32x4_t ag1 = s_g[kb][rs][16 + sub_lane];
      const int32x4_t au0 = s_u[kb][rs][sub_lane];
      const int32x4_t au1 = s_u[kb][rs][16 + sub_lane];
      float dwg0[8], dwu0[8], dwg1[PerHalfScale ? 8 : 1],
          dwu1[PerHalfScale ? 8 : 1];
      const float4 g_lo =
          *reinterpret_cast<const float4*>(&s_dwg[kb][0][rs][half_id * 8]);
      const float4 g_up =
          *reinterpret_cast<const float4*>(&s_dwg[kb][0][rs][(half_id * 8) + 4]);
      const float4 u_lo =
          *reinterpret_cast<const float4*>(&s_dwu[kb][0][rs][half_id * 8]);
      const float4 u_up =
          *reinterpret_cast<const float4*>(&s_dwu[kb][0][rs][(half_id * 8) + 4]);
      dwg0[0] = g_lo.x; dwg0[1] = g_lo.y; dwg0[2] = g_lo.z; dwg0[3] = g_lo.w;
      dwg0[4] = g_up.x; dwg0[5] = g_up.y; dwg0[6] = g_up.z; dwg0[7] = g_up.w;
      dwu0[0] = u_lo.x; dwu0[1] = u_lo.y; dwu0[2] = u_lo.z; dwu0[3] = u_lo.w;
      dwu0[4] = u_up.x; dwu0[5] = u_up.y; dwu0[6] = u_up.z; dwu0[7] = u_up.w;
      if constexpr (PerHalfScale) {
        const float4 h_glo =
            *reinterpret_cast<const float4*>(&s_dwg[kb][1][rs][half_id * 8]);
        const float4 h_gup =
            *reinterpret_cast<const float4*>(&s_dwg[kb][1][rs][(half_id * 8) + 4]);
        const float4 h_ulo =
            *reinterpret_cast<const float4*>(&s_dwu[kb][1][rs][half_id * 8]);
        const float4 h_uup =
            *reinterpret_cast<const float4*>(&s_dwu[kb][1][rs][(half_id * 8) + 4]);
        dwg1[0] = h_glo.x; dwg1[1] = h_glo.y; dwg1[2] = h_glo.z; dwg1[3] = h_glo.w;
        dwg1[4] = h_gup.x; dwg1[5] = h_gup.y; dwg1[6] = h_gup.z; dwg1[7] = h_gup.w;
        dwu1[0] = h_ulo.x; dwu1[1] = h_ulo.y; dwu1[2] = h_ulo.z; dwu1[3] = h_ulo.w;
        dwu1[4] = h_uup.x; dwu1[5] = h_uup.y; dwu1[6] = h_uup.z; dwu1[7] = h_uup.w;
      }
#pragma unroll
      for (int j = 0; j < G; ++j) {
        if (j >= ntiles) {
          continue;
        }
        const int32x4_t b0 = s_b[kb][j][sub_lane];
        const int32x4_t b1 = s_b[kb][j][16 + sub_lane];
        const float dx = s_dx[kb][j][sub_lane];
        if constexpr (PerHalfScale) {
          int32x8_t cg0 = {0, 0, 0, 0, 0, 0, 0, 0};
          int32x8_t cg1 = {0, 0, 0, 0, 0, 0, 0, 0};
          int32x8_t cu0 = {0, 0, 0, 0, 0, 0, 0, 0};
          int32x8_t cu1 = {0, 0, 0, 0, 0, 0, 0, 0};
          cg0 = WmmaQuant<32>(ag0, b0, cg0);
          cg1 = WmmaQuant<32>(ag1, b1, cg1);
          cu0 = WmmaQuant<32>(au0, b0, cu0);
          cu1 = WmmaQuant<32>(au1, b1, cu1);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc_g[j][l] += (dwg0[l] * dx) * static_cast<float>(cg0[l]) +
                           (dwg1[l] * dx) * static_cast<float>(cg1[l]);
            acc_u[j][l] += (dwu0[l] * dx) * static_cast<float>(cu0[l]) +
                           (dwu1[l] * dx) * static_cast<float>(cu1[l]);
          }
        } else {
          int32x8_t cg = {0, 0, 0, 0, 0, 0, 0, 0};
          int32x8_t cu = {0, 0, 0, 0, 0, 0, 0, 0};
          cg = WmmaQuant<32>(ag0, b0, cg);
          cg = WmmaQuant<32>(ag1, b1, cg);
          cu = WmmaQuant<32>(au0, b0, cu);
          cu = WmmaQuant<32>(au1, b1, cu);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc_g[j][l] += (dwg0[l] * dx) * static_cast<float>(cg[l]);
            acc_u[j][l] += (dwu0[l] * dx) * static_cast<float>(cu[l]);
          }
        }
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
  }

  // Transpose each 16x16 tile through LDS (one scratch per accumulator set)
  // and write f32 rows indexed by global slot. All 16 rows of the tile are
  // written; pad rows are never read downstream.
  __syncthreads();
  float* sc_g = reinterpret_cast<float*>(&s_g[0][0][0]) + (wave_id * 256);
  float* sc_u = reinterpret_cast<float*>(&s_u[0][0][0]) + (wave_id * 256);
  const std::size_t r0 = r_block + static_cast<std::size_t>(wave_id * 16);
#pragma unroll
  for (int j = 0; j < G; ++j) {
    if (j >= ntiles) {
      continue;
    }
#pragma unroll
    for (int l = 0; l < 8; ++l) {
      sc_g[(sub_lane * 16) + (2 * l) + half_id] = acc_g[j][l];
      sc_u[(sub_lane * 16) + (2 * l) + half_id] = acc_u[j][l];
    }
    __builtin_amdgcn_wave_barrier();
    const std::size_t sj = slot0 + static_cast<std::size_t>(j) * 16;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int flat = (s * 32) + lane_id;
      const std::size_t out = (sj + static_cast<std::size_t>(flat >> 4)) *
                                  static_cast<std::size_t>(expert_ff) +
                              r0 + static_cast<std::size_t>(flat & 15);
      gate_out[out] = sc_g[flat];
      up_out[out] = sc_u[flat];
    }
    __builtin_amdgcn_wave_barrier();
  }
}

/// Grouped down projection over the expert stacks on WMMA int8 tensor cores:
/// down_out[slot, r] = slot_weight * (down_e[r] . act[slot]). Templated on the
/// down weight format: Q8_0 (the UD-Q6_K_XL stacks), Q6_K (per-16 scales) and
/// IQ4_XS (per-32 scales, the bartowski int4 artifact) all decode through
/// DecodeQuantSub16; Q8_0's decode is the same 32-byte block read the original
/// kernel performed, so its outputs stay bit-identical. The epilogue applies
/// the slot's router weight.
template <core::GgmlType WType>
__launch_bounds__(256, 2) __global__ void FlintMoeGroupedDownWmmaKernel(
    const void* __restrict__ down_stack, const void* __restrict__ act_q8,
    float* __restrict__ down_out, const int* __restrict__ offsets,
    const float* __restrict__ slot_weights,
    const int* __restrict__ tile_expert, const int* __restrict__ tile_slot0,
    const int* __restrict__ tile_count, int expert_ff, int hidden,
    std::size_t slab_bytes) {
  constexpr int BM = 128;
  constexpr int BK = 2;
  constexpr int G = kFlintGroupTiles;
  constexpr int kRowTiles = BM / 16;
  constexpr bool PerHalfScale =
      WType == core::GgmlType::kQ6_K || WType == core::GgmlType::kQ3_K;
  constexpr int kScaleHalves = PerHalfScale ? 2 : 1;

  const int t = static_cast<int>(blockIdx.x);
  if (t >= tile_count[0]) {
    return;
  }
  const int e = tile_expert[t];
  // Tiles are expert-sorted. A block claims a group of up to G consecutive
  // tiles of one expert: the group leader is the tile whose offset within the
  // expert's run is a multiple of G. One decoded weight block then feeds G
  // activation tiles, cutting the per-expert weight re-read.
  int run_start = t;
  while (run_start > 0 && tile_expert[run_start - 1] == e) {
    --run_start;
  }
  if (((t - run_start) % G) != 0) {
    return;
  }
  int ntiles = 1;
#pragma unroll
  for (int j = 1; j < G; ++j) {
    if (t + j < tile_count[0] && tile_expert[t + j] == e) {
      ++ntiles;
    }
  }
  const std::size_t slot0 = static_cast<std::size_t>(tile_slot0[t]);
  const std::size_t r_block = static_cast<std::size_t>(blockIdx.y) * BM;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;

  const std::size_t num_blocks = static_cast<std::size_t>(expert_ff) / 32;
  const std::size_t row_bytes = QuantRowBytes(WType, expert_ff);
  const std::uint8_t* wbase = static_cast<const std::uint8_t*>(down_stack) +
                              static_cast<std::size_t>(e) * slab_bytes;

  __shared__ int32x4_t s_a[BK][kRowTiles][32];
  __shared__ float s_dw[BK][kScaleHalves][kRowTiles][16];
  __shared__ int32x4_t s_b[BK][G][32];
  __shared__ float s_dx[BK][G][16];

  const int rr = tid / BK;
  const int kk = tid % BK;
  const int rl = rr % 16;
  const int scale_slot = (rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2));
  const std::uint8_t* wline =
      wbase + (r_block + static_cast<std::size_t>(rr)) * row_bytes;

  const std::int8_t* b_base =
      static_cast<const std::int8_t*>(act_q8) +
      ((slot0 >> 4) * num_blocks * kQ8ActTileBytes);

  float acc[G][8];
#pragma unroll
  for (int j = 0; j < G; ++j) {
#pragma unroll
    for (int l = 0; l < 8; ++l) {
      acc[j][l] = 0.0F;
    }
  }

  int32x4_t r_q0, r_q1;
  float r_dw0, r_dw1;
  int32x4_t r_b[G][BK];
  float r_dx[G][BK];

  const auto fetch_stage = [&](int kb0) {
    const int kb = kb0 + kk;
    QuantSub16 lo, hi;
    DecodeQuantSub16(WType, wline, static_cast<std::size_t>(kb) * 2, lo);
    DecodeQuantSub16(WType, wline, static_cast<std::size_t>(kb) * 2 + 1, hi);
    __builtin_memcpy(&r_q0, lo.q, 16);
    __builtin_memcpy(&r_q1, hi.q, 16);
    r_dw0 = lo.scale;
    r_dw1 = hi.scale;
    if (wave_id == 0) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
#pragma unroll
        for (int j = 0; j < G; ++j) {
          if (j >= ntiles) {
            continue;
          }
          const std::int8_t* tile =
              b_base + ((static_cast<std::size_t>(j) * num_blocks) + kb0 + i) *
                           kQ8ActTileBytes;
          r_b[j][i] = reinterpret_cast<const int32x4_t*>(tile)[lane_id];
          r_dx[j][i] = reinterpret_cast<const float*>(
              tile + kQ8ActScaleOffset)[lane_id & 15];
        }
      }
    }
  };

  const auto commit_stage = [&]() {
    s_a[kk][rr / 16][rl] = r_q0;
    s_a[kk][rr / 16][16 + rl] = r_q1;
    s_dw[kk][0][rr / 16][scale_slot] = r_dw0;
    if constexpr (PerHalfScale) {
      s_dw[kk][1][rr / 16][scale_slot] = r_dw1;
    }
    if (wave_id == 0) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
#pragma unroll
        for (int j = 0; j < G; ++j) {
          if (j >= ntiles) {
            continue;
          }
          s_b[i][j][lane_id] = r_b[j][i];
          if (lane_id < 16) {
            s_dx[i][j][lane_id] = r_dx[j][i];
          }
        }
      }
    }
  };

  const int num_kb = static_cast<int>(num_blocks);
  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      const int rs = wave_id;
      const int32x4_t a0 = s_a[kb][rs][sub_lane];
      const int32x4_t a1 = s_a[kb][rs][16 + sub_lane];
      float dw0[8], dw1[PerHalfScale ? 8 : 1];
      const float4 lo =
          *reinterpret_cast<const float4*>(&s_dw[kb][0][rs][half_id * 8]);
      const float4 up =
          *reinterpret_cast<const float4*>(&s_dw[kb][0][rs][(half_id * 8) + 4]);
      dw0[0] = lo.x; dw0[1] = lo.y; dw0[2] = lo.z; dw0[3] = lo.w;
      dw0[4] = up.x; dw0[5] = up.y; dw0[6] = up.z; dw0[7] = up.w;
      if constexpr (PerHalfScale) {
        const float4 h_lo =
            *reinterpret_cast<const float4*>(&s_dw[kb][1][rs][half_id * 8]);
        const float4 h_up =
            *reinterpret_cast<const float4*>(&s_dw[kb][1][rs][(half_id * 8) + 4]);
        dw1[0] = h_lo.x; dw1[1] = h_lo.y; dw1[2] = h_lo.z; dw1[3] = h_lo.w;
        dw1[4] = h_up.x; dw1[5] = h_up.y; dw1[6] = h_up.z; dw1[7] = h_up.w;
      }
#pragma unroll
      for (int j = 0; j < G; ++j) {
        if (j >= ntiles) {
          continue;
        }
        const int32x4_t b0 = s_b[kb][j][sub_lane];
        const int32x4_t b1 = s_b[kb][j][16 + sub_lane];
        const float dx = s_dx[kb][j][sub_lane];
        if constexpr (PerHalfScale) {
          int32x8_t c0 = {0, 0, 0, 0, 0, 0, 0, 0};
          int32x8_t c1 = {0, 0, 0, 0, 0, 0, 0, 0};
          c0 = WmmaQuant<32>(a0, b0, c0);
          c1 = WmmaQuant<32>(a1, b1, c1);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc[j][l] += (dw0[l] * dx) * static_cast<float>(c0[l]) +
                         (dw1[l] * dx) * static_cast<float>(c1[l]);
          }
        } else {
          int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
          c = WmmaQuant<32>(a0, b0, c);
          c = WmmaQuant<32>(a1, b1, c);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc[j][l] += (dw0[l] * dx) * static_cast<float>(c[l]);
          }
        }
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
  }

  __syncthreads();
  float* scratch = reinterpret_cast<float*>(&s_a[0][0][0]) + (wave_id * 256);
  const std::size_t r0 = r_block + static_cast<std::size_t>(wave_id * 16);
#pragma unroll
  for (int j = 0; j < G; ++j) {
    if (j >= ntiles) {
      continue;
    }
#pragma unroll
    for (int l = 0; l < 8; ++l) {
      scratch[(sub_lane * 16) + (2 * l) + half_id] = acc[j][l];
    }
    __builtin_amdgcn_wave_barrier();
    const std::size_t sj = slot0 + static_cast<std::size_t>(j) * 16;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int flat = (s * 32) + lane_id;
      const std::size_t slot = sj + static_cast<std::size_t>(flat >> 4);
      down_out[slot * static_cast<std::size_t>(hidden) + r0 +
               static_cast<std::size_t>(flat & 15)] =
          slot_weights[slot] * scratch[flat];
    }
    __builtin_amdgcn_wave_barrier();
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
  // A/B override: the W1 measurements compare tiled vs scalar on one binary.
  static const bool force_scalar =
      std::getenv("GUFO_FLINT_ROUTER_SCALAR") != nullptr;
  if (!force_scalar && hidden % 64 == 0) {
    const dim3 grid(static_cast<unsigned int>((expert_count + 31) / 32),
                    static_cast<unsigned int>((n_tokens + 31) / 32));
    if (type == core::GgmlType::kBF16) {
      FlintRouterTiledKernel<hip_bfloat16><<<grid, 256, 0, stream>>>(
          x, static_cast<int>(n_tokens), static_cast<const hip_bfloat16*>(w),
          static_cast<int>(expert_count), static_cast<int>(hidden), logits);
      return;
    }
    if (type == core::GgmlType::kF16) {
      FlintRouterTiledKernel<__half><<<grid, 256, 0, stream>>>(
          x, static_cast<int>(n_tokens), static_cast<const __half*>(w),
          static_cast<int>(expert_count), static_cast<int>(hidden), logits);
      return;
    }
    if (type == core::GgmlType::kF32) {
      FlintRouterTiledKernel<float><<<grid, 256, 0, stream>>>(
          x, static_cast<int>(n_tokens), static_cast<const float*>(w),
          static_cast<int>(expert_count), static_cast<int>(hidden), logits);
      return;
    }
    throw std::invalid_argument(
        "flint batched router requires an f32, f16 or bf16 router tensor");
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
                         int* token_slots, int* tile_expert, int* tile_slot0,
                         int* tile_count, hipStream_t stream) {
  const int total = static_cast<int>(n_tokens * used);
  const unsigned int blocks =
      static_cast<unsigned int>((total + 255) / 256);
  HIP_CHECK(hipMemsetAsync(counts, 0, expert_count * sizeof(int), stream));
  FlintMoeCountKernel<<<blocks, 256, 0, stream>>>(selected, total, counts);
  FlintMoeOffsetsKernel<<<1, 32, 0, stream>>>(counts, offsets, cursors,
                                              slot_tokens, slot_weights,
                                              tile_expert, tile_slot0,
                                              tile_count,
                                              static_cast<int>(expert_count));
  FlintMoeScatterKernel<<<blocks, 256, 0, stream>>>(
      selected, sel_weights, cursors, slot_tokens, slot_weights, token_slots,
      total, static_cast<int>(used));
}

void LaunchFlintMoeBatchSwiGLU(const void* gate_stack, const void* up_stack,
                               core::GgmlType type, core::GgmlType up_type,
                               const float* x, float* act,
                               const int* offsets, const int* slot_tokens,
                               std::size_t expert_count, std::size_t expert_ff,
                               std::size_t hidden, std::size_t slab_bytes,
                               std::size_t slab_up,
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
      gate_stack, up_stack, type, up_type, x, act, offsets, slot_tokens,
      static_cast<int>(expert_ff), static_cast<int>(hidden), slab_bytes,
      slab_up);
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

// ---------------------------------------------------------------------------
// Grouped WMMA route (F-L4 tuning). `tile_bound` is the host-side upper bound
// on the device-built tile count: ceil(slots / 16) + expert_count (each expert
// wastes at most one tile to padding).
// ---------------------------------------------------------------------------

void LaunchFlintGatherQ8Act(const void* x_q8, void* slot_q8,
                            const int* slot_tokens, const int* offsets,
                            std::size_t expert_count, std::size_t slot_bound,
                            std::size_t hidden, hipStream_t stream) {
  if (slot_bound == 0 || hidden == 0) {
    return;
  }
  FlintGatherQ8ActKernel<<<static_cast<unsigned int>(slot_bound),
                           static_cast<unsigned int>(hidden / 32), 0, stream>>>(
      slot_tokens, offsets, static_cast<int>(expert_count),
      static_cast<const std::int8_t*>(x_q8),
      static_cast<std::int8_t*>(slot_q8), static_cast<int>(hidden / 32));
}

bool FlintGroupedWmmaSupported(core::GgmlType gate_up_type,
                               core::GgmlType up_type,
                               core::GgmlType down_type, std::size_t expert_ff,
                               std::size_t hidden) noexcept {
  // The grouped gate/up kernel mirrors WKQuantA8's no-offset formats; the
  // fused kernel decodes gate and up rows with the SAME type, so a layer with
  // gate != up (bartowski IQ4_XS blk.2) must fall back to the scalar pair.
  // The down kernel covers Q8_0 / Q6_K / IQ4_XS stacks.
  const bool gate_up_ok =
      gate_up_type == up_type &&
      (gate_up_type == core::GgmlType::kQ6_K ||
       gate_up_type == core::GgmlType::kQ3_K ||
       gate_up_type == core::GgmlType::kIQ4_NL ||
       gate_up_type == core::GgmlType::kIQ4_XS ||
       gate_up_type == core::GgmlType::kIQ3_S);
  const bool down_ok = down_type == core::GgmlType::kQ8_0 ||
                       down_type == core::GgmlType::kQ6_K ||
                       down_type == core::GgmlType::kIQ4_XS;
  return gate_up_ok && down_ok && expert_ff % 128 == 0 && hidden % 128 == 0 &&
         hidden % 64 == 0 && expert_ff % 64 == 0;
}

void LaunchFlintMoeGroupedGateUpWmma(const void* gate_stack,
                                     const void* up_stack,
                                     core::GgmlType type, const void* slot_q8,
                                     float* gate_out, float* up_out,
                                     const int* offsets,
                                     const int* tile_expert,
                                     const int* tile_slot0,
                                     const int* tile_count,
                                     std::size_t tile_bound,
                                     std::size_t expert_count,
                                     std::size_t expert_ff, std::size_t hidden,
                                     std::size_t slab_bytes,
                                     hipStream_t stream) {
  if (tile_bound == 0 || expert_ff == 0) {
    return;
  }
  const dim3 grid(static_cast<unsigned int>(tile_bound),
                  static_cast<unsigned int>(expert_ff / 128));
  const auto launch = [&](auto type_tag) {
    constexpr core::GgmlType kType = decltype(type_tag)::value;
    FlintMoeGroupedGateUpWmmaKernel<kType><<<grid, 256, 0, stream>>>(
        gate_stack, up_stack, slot_q8, gate_out, up_out, offsets, tile_expert,
        tile_slot0, tile_count, static_cast<int>(expert_ff),
        static_cast<int>(hidden), slab_bytes);
  };
  switch (type) {
    case core::GgmlType::kQ6_K:
      launch(std::integral_constant<core::GgmlType, core::GgmlType::kQ6_K>{});
      break;
    case core::GgmlType::kQ3_K:
      launch(std::integral_constant<core::GgmlType, core::GgmlType::kQ3_K>{});
      break;
    case core::GgmlType::kIQ4_NL:
      launch(
          std::integral_constant<core::GgmlType, core::GgmlType::kIQ4_NL>{});
      break;
    case core::GgmlType::kIQ4_XS:
      launch(
          std::integral_constant<core::GgmlType, core::GgmlType::kIQ4_XS>{});
      break;
    case core::GgmlType::kIQ3_S:
      launch(std::integral_constant<core::GgmlType, core::GgmlType::kIQ3_S>{});
      break;
    default:
      throw std::invalid_argument(
          "flint grouped WMMA gate/up does not support the routed type");
  }
}

void LaunchFlintMoeGroupedDownWmma(const void* down_stack,
                                   core::GgmlType type, const void* act_q8,
                                   float* down_out,
                                   const int* offsets,
                                   const float* slot_weights,
                                   const int* tile_expert,
                                   const int* tile_slot0,
                                   const int* tile_count,
                                   std::size_t tile_bound,
                                   std::size_t expert_count,
                                   std::size_t expert_ff, std::size_t hidden,
                                   std::size_t slab_bytes,
                                   hipStream_t stream) {
  if (tile_bound == 0 || hidden == 0) {
    return;
  }
  const dim3 grid(static_cast<unsigned int>(tile_bound),
                  static_cast<unsigned int>(hidden / 128));
  const auto launch = [&](auto type_tag) {
    constexpr core::GgmlType kType = decltype(type_tag)::value;
    FlintMoeGroupedDownWmmaKernel<kType><<<grid, 256, 0, stream>>>(
        down_stack, act_q8, down_out, offsets, slot_weights, tile_expert,
        tile_slot0, tile_count, static_cast<int>(expert_ff),
        static_cast<int>(hidden), slab_bytes);
  };
  switch (type) {
    case core::GgmlType::kQ8_0:
      launch(std::integral_constant<core::GgmlType, core::GgmlType::kQ8_0>{});
      break;
    case core::GgmlType::kQ6_K:
      launch(std::integral_constant<core::GgmlType, core::GgmlType::kQ6_K>{});
      break;
    case core::GgmlType::kIQ4_XS:
      launch(std::integral_constant<core::GgmlType, core::GgmlType::kIQ4_XS>{});
      break;
    default:
      throw std::invalid_argument(
          "flint grouped WMMA down does not support the routed type");
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
