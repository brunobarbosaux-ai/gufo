// Flint MoE device kernels (F-E1/F-E3): sigmoid router top-k selection and
// weighted expert accumulation. Selection matches the CPU reference exactly:
// sigmoid over the full expert set, top-`used` with ties resolved to the
// lowest index, renormalisation over the selected probabilities, then the
// expert weights scale.
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {

namespace {

constexpr int kMaxExperts = 1024;
constexpr int kMaxSelected = 8;

__global__ void FlintRouterTopKKernel(const float* __restrict__ logits,
                                      int expert_count, int used, int norm,
                                      float scale,
                                      std::uint32_t* __restrict__ selected,
                                      float* __restrict__ weights,
                                      float* __restrict__ weights_full,
                                      float shared_weight,
                                      float* __restrict__ zero_out,
                                      std::size_t zero_n,
                                      int* __restrict__ job_list) {
  __shared__ float s_val[kMaxExperts];
  __shared__ float s_warp_best[32];
  __shared__ unsigned int s_warp_idx[32];
  __shared__ float s_winner_value[kMaxSelected];
  __shared__ unsigned int s_winner_idx[kMaxSelected];
  __shared__ float s_sum;

  const int tid = static_cast<int>(threadIdx.x);
  if (tid == 0) {
    s_sum = 0.0F;
  }
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
      if (selected != nullptr) {
        selected[k] = index;
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

  if (weights != nullptr && tid < used) {
    float weight = s_winner_value[tid];
    if (norm && s_sum > 0.0F) {
      weight /= s_sum;
    }
    weights[tid] = weight * scale;
  }

  // Fused-dispatch epilogue (F-J1): the full weight array lets the expert
  // kernels self-gate on device, removing the host readback sync. The
  // shared expert (ungated, unit weight, F-E3) occupies slot expert_count.
  if (weights_full != nullptr) {
    if (tid < expert_count) {
      float value = 0.0F;
      for (int k = 0; k < used; ++k) {
        if (tid == static_cast<int>(s_winner_idx[k])) {
          value = s_winner_value[k];
        }
      }
      if (norm && s_sum > 0.0F) {
        value /= s_sum;
      }
      weights_full[tid] = value * scale;
    }
    if (tid == 0) {
      weights_full[expert_count] = shared_weight;
    }
  }
  // Job compaction (F-J1): the fused expert kernels index their grids
  // through this list instead of spanning the whole expert stack — ~257x
  // fewer block dispatches per launch, the dominant short-context cost.
  // Order is the top-k order (descending sigmoid, ties to the lowest
  // index) with the shared expert last, matching the legacy accumulation
  // order; -1 pads the unused slots.
  if (job_list != nullptr) {
    if (tid <= used) {
      job_list[tid] = (tid == used)
                          ? expert_count
                          : static_cast<int>(s_winner_idx[tid]);
    } else if (tid < static_cast<int>(kFlintMoeMaxJobs)) {
      job_list[tid] = -1;
    }
  }
  if (zero_out != nullptr) {
    for (std::size_t i = tid; i < zero_n; i += blockDim.x) {
      zero_out[i] = 0.0F;
    }
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

// Fused MoE branch (F-J1): the grid spans only real jobs when the router's
// job list is provided (no host readback sync, no per-expert launches);
// without it the grid spans the whole expert stack and blocks self-gate on
// weights_full (synthetic tests). Tiling mirrors the validated legacy
// kernels: 128 threads = 4 warps, each warp owns TWO CONSECUTIVE rows so
// the block-dot chains interleave and the warp reduces once per row pair.
// opt-r7's per-block-butterfly shape (one row per warp pass, 8 sequential
// rows) measured 2.5-3x slower per layer on gfx1151 — see probe stage
// timings. Q8_0 pairs (layer 14, every shared expert) take the dedicated
// register-accumulation loop from swiglu.hip / gemv_quant.hip.

constexpr int kMoEWavesPerBlock = 4;                     // warps per block
constexpr int kMoERowsPerBlock = kMoEWavesPerBlock * 2;  // 2 rows per warp

// Device copy of the IQ4_NL codebook (kValuesIq4Nl): the fused MoE fast
// paths stage it into LDS once per block for divergent lane lookups.
__device__ constexpr std::int8_t kIq4NlDeviceLut[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// Q6_K lane decode (7.4b): lane `lane` owns elements {lane, lane+32, lane+64,
// lane+96} of each 128-half of one 256-wide block, so a warp covers the whole
// block with each lane reading exactly 6 bytes (4 ql + 2 qh) instead of the
// generic sub16 route's 32 bytes per 16 elements. The per-16-element int8
// scales are applied directly (four per lane per half). `xb` points at the
// block's 256 x values; the block's fp16 `d` is applied by the caller so it
// can be shared across the gate/up pair. Mirrors ggml's dequantize_row_q6_K
// element order exactly.
__device__ inline float Q6KBlockDotLane(const Q6KBlock& blk,
                                        const float* __restrict__ xb,
                                        std::size_t lane) noexcept {
  const std::size_t is = lane / 16;
  float acc = 0.0F;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const std::uint8_t* ql = blk.ql + (h * 64);
    const std::uint8_t* qh = blk.qh + (h * 32);
    const std::int8_t* sc = blk.scales + (h * 8);
    const float* xh = xb + (h * 128);
    const std::uint8_t qhb = qh[lane];
    const int q1 = static_cast<int>((ql[lane] & 0x0FU) |
                                    (((qhb >> 0U) & 3U) << 4U)) -
                   32;
    const int q2 = static_cast<int>((ql[lane + 32] & 0x0FU) |
                                    (((qhb >> 2U) & 3U) << 4U)) -
                   32;
    const int q3 = static_cast<int>((ql[lane] >> 4U) |
                                    (((qhb >> 4U) & 3U) << 4U)) -
                   32;
    const int q4 = static_cast<int>((ql[lane + 32] >> 4U) |
                                    (((qhb >> 6U) & 3U) << 4U)) -
                   32;
    acc += static_cast<float>(sc[is + 0]) * static_cast<float>(q1) * xh[lane];
    acc +=
        static_cast<float>(sc[is + 2]) * static_cast<float>(q2) * xh[lane + 32];
    acc +=
        static_cast<float>(sc[is + 4]) * static_cast<float>(q3) * xh[lane + 64];
    acc +=
        static_cast<float>(sc[is + 6]) * static_cast<float>(q4) * xh[lane + 96];
  }
  return acc;
}

__global__ void __launch_bounds__(kMoEWavesPerBlock * 32, 4)
    FlintFusedMoESwiGLUKernel(
        const void* __restrict__ gate_stack, const void* __restrict__ up_stack,
        const void* __restrict__ gate_shared, const void* __restrict__ up_shared,
        core::GgmlType gate_type, core::GgmlType up_type,
        core::GgmlType shared_gate_type,
        core::GgmlType shared_up_type, const float* __restrict__ x,
        float* __restrict__ act, const float* __restrict__ weights_full,
        int expert_count, std::size_t expert_ff, std::size_t shared_ff,
        std::size_t hidden, std::size_t gate_slab_bytes,
        std::size_t up_slab_bytes,
        const int* __restrict__ job_list) {
  const int e = job_list != nullptr
                    ? job_list[blockIdx.x]
                    : static_cast<int>(blockIdx.x);
  if (e < 0) {
    return;
  }
  const float w = weights_full[e];
  if (w == 0.0F) {
    return;
  }
  // IQ4_XS fast paths read the codebook from LDS; the stage is uniform per
  // block (both early returns above are too), so the barrier is safe.
  __shared__ float s_iq4_lut[16];
  if (static_cast<int>(threadIdx.x) < 16) {
    s_iq4_lut[threadIdx.x] = static_cast<float>(kIq4NlDeviceLut[threadIdx.x]);
  }
  __syncthreads();
  const bool is_shared = (e == expert_count);
  const std::size_t ff = is_shared ? shared_ff : expert_ff;
  const void* gate = is_shared
                         ? gate_shared
                         : static_cast<const char*>(gate_stack) +
                               static_cast<std::size_t>(e) * gate_slab_bytes;
  const void* up = is_shared
                       ? up_shared
                       : static_cast<const char*>(up_stack) +
                             static_cast<std::size_t>(e) * up_slab_bytes;
  // UD checkpoints quantise the shared expert independently of the routed
  // stacks (Flint UD-Q6_K_XL: gate/up Q6_K routed, everything shared Q8_0),
  // so each matrix decodes with its own type. The routed stacks may ALSO
  // differ from each other (bartowski IQ4_XS blk.2: gate IQ4_XS + up Q5_K),
  // each with its own slab stride.
  const core::GgmlType gt = is_shared ? shared_gate_type : gate_type;
  const core::GgmlType ut = is_shared ? shared_up_type : up_type;
  float* act_e =
      act + (is_shared ? static_cast<std::size_t>(expert_count) * expert_ff
                       : static_cast<std::size_t>(e) * expert_ff);

  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp = threadIdx.x >> 5u;
  const std::size_t r0 =
      (static_cast<std::size_t>(blockIdx.y) * kMoEWavesPerBlock + warp) * 2;
  if (r0 >= ff) {
    return;
  }
  const bool has_r1 = (r0 + 1) < ff;

  if (gt == core::GgmlType::kQ8_0 && ut == core::GgmlType::kQ8_0 &&
      hidden % kQ8_0BlockSize == 0) {
    // Dedicated 2-row Q8_0 loop: per-lane register accumulation over
    // lane-strided blocks, one butterfly per row pair at the end.
    const std::size_t num_blocks = hidden / kQ8_0BlockSize;
    const auto* g0 =
        static_cast<const Q8_0Block*>(gate) + (r0 * num_blocks);
    const auto* u0 = static_cast<const Q8_0Block*>(up) + (r0 * num_blocks);
    const auto* g1 =
        has_r1 ? static_cast<const Q8_0Block*>(gate) + ((r0 + 1) * num_blocks)
               : nullptr;
    const auto* u1 =
        has_r1 ? static_cast<const Q8_0Block*>(up) + ((r0 + 1) * num_blocks)
               : nullptr;
    float sum_g0 = 0.0F, sum_u0 = 0.0F, sum_g1 = 0.0F, sum_u1 = 0.0F;
    for (std::size_t b = lane; b < num_blocks; b += 32) {
      const float* xb = x + (b * kQ8_0BlockSize);
      const Q8_0Block& gb0 = g0[b];
      const Q8_0Block& ub0 = u0[b];
      const float dg0 = __half2float(gb0.d);
      const float du0 = __half2float(ub0.d);
      float blk_g0 = 0.0F, blk_u0 = 0.0F;
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        blk_g0 += static_cast<float>(gb0.qs[i]) * xb[i];
        blk_u0 += static_cast<float>(ub0.qs[i]) * xb[i];
      }
      sum_g0 += dg0 * blk_g0;
      sum_u0 += du0 * blk_u0;
      if (g1 != nullptr) {
        const Q8_0Block& gb1 = g1[b];
        const Q8_0Block& ub1 = u1[b];
        const float dg1 = __half2float(gb1.d);
        const float du1 = __half2float(ub1.d);
        float blk_g1 = 0.0F, blk_u1 = 0.0F;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          blk_g1 += static_cast<float>(gb1.qs[i]) * xb[i];
          blk_u1 += static_cast<float>(ub1.qs[i]) * xb[i];
        }
        sum_g1 += dg1 * blk_g1;
        sum_u1 += du1 * blk_u1;
      }
    }
    for (int off = 16; off > 0; off >>= 1) {
      sum_g0 += __shfl_xor(sum_g0, off);
      sum_u0 += __shfl_xor(sum_u0, off);
      sum_g1 += __shfl_xor(sum_g1, off);
      sum_u1 += __shfl_xor(sum_u1, off);
    }
    if (lane == 0) {
      // SwiGLU = silu(gate) * up, matching FlintSwiglu on the CPU.
      act_e[r0] = (sum_g0 / (1.0F + expf(-sum_g0))) * sum_u0;
      if (has_r1) {
        act_e[r0 + 1] = (sum_g1 / (1.0F + expf(-sum_g1))) * sum_u1;
      }
    }
    return;
  }

  if (gt == core::GgmlType::kIQ4_XS && ut == core::GgmlType::kIQ4_XS &&
      hidden % 256 == 0) {
    // IQ4_XS 2-row loop: 32 lanes cover one 256-value super-block (8
    // values each, one sub-block per lane); x is loaded once and shared
    // by both rows and both matrices. One butterfly per row-matrix total
    // (QuantWarpBlockDot pays one per 32-value sub-block — the FFN
    // stage's latency is exactly that reduce chain).
    const std::size_t supers = hidden / 256;
    const auto* g_blocks = static_cast<const IQ4XSBlock*>(gate);
    const auto* u_blocks = static_cast<const IQ4XSBlock*>(up);
    const std::uint32_t quad = lane % 4u;
    const std::uint32_t byte_base = (quad % 2u) * 8u;
    const std::uint32_t nib_shift = (quad / 2u) * 4u;
    const std::uint32_t sub = lane / 4u;
    float sum_g0 = 0.0F, sum_u0 = 0.0F, sum_g1 = 0.0F, sum_u1 = 0.0F;
    for (std::size_t sb = 0; sb < supers; ++sb) {
      const float* xb = x + (sb * 256) + (lane * 8);
      const float4 xa = *reinterpret_cast<const float4*>(xb);
      const float4 xq = *reinterpret_cast<const float4*>(xb + 4);
      const float xs[8] = {xa.x, xa.y, xa.z, xa.w, xq.x, xq.y, xq.z, xq.w};
      const IQ4XSBlock& g0b = g_blocks[(r0 * supers) + sb];
      const IQ4XSBlock& u0b = u_blocks[(r0 * supers) + sb];
      const std::uint32_t ls_sel = (g0b.scales_l[sub / 2] >> (4u * (sub % 2u))) & 0xFu;
      const std::uint32_t ls0 =
          ls_sel | (((g0b.scales_h >> (2u * sub)) & 0x3u) << 4u);
      const float gscale0 =
          __half2float(g0b.d) * static_cast<float>(static_cast<int>(ls0) - 32);
      const std::uint32_t ls_sel_u =
          (u0b.scales_l[sub / 2] >> (4u * (sub % 2u))) & 0xFu;
      const std::uint32_t ls1 =
          ls_sel_u | (((u0b.scales_h >> (2u * sub)) & 0x3u) << 4u);
      const float uscale0 =
          __half2float(u0b.d) * static_cast<float>(static_cast<int>(ls1) - 32);
      float acc_g0 = 0.0F, acc_u0 = 0.0F;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const std::uint32_t gc =
            (static_cast<std::uint32_t>(g0b.qs[sub * 16 + byte_base + j]) >> nib_shift) &
            0xFu;
        const std::uint32_t uc =
            (static_cast<std::uint32_t>(u0b.qs[sub * 16 + byte_base + j]) >> nib_shift) &
            0xFu;
        acc_g0 += s_iq4_lut[gc] * xs[j];
        acc_u0 += s_iq4_lut[uc] * xs[j];
      }
      sum_g0 += gscale0 * acc_g0;
      sum_u0 += uscale0 * acc_u0;
      if (has_r1) {
        const IQ4XSBlock& g1b = g_blocks[((r0 + 1) * supers) + sb];
        const IQ4XSBlock& u1b = u_blocks[((r0 + 1) * supers) + sb];
        const std::uint32_t ls_sel_1 =
            (g1b.scales_l[sub / 2] >> (4u * (sub % 2u))) & 0xFu;
        const std::uint32_t lsg1 =
            ls_sel_1 | (((g1b.scales_h >> (2u * sub)) & 0x3u) << 4u);
        const float gscale1 = __half2float(g1b.d) *
                              static_cast<float>(static_cast<int>(lsg1) - 32);
        const std::uint32_t ls_sel_u1 =
            (u1b.scales_l[sub / 2] >> (4u * (sub % 2u))) & 0xFu;
        const std::uint32_t lsu1 =
            ls_sel_u1 | (((u1b.scales_h >> (2u * sub)) & 0x3u) << 4u);
        const float uscale1 = __half2float(u1b.d) *
                              static_cast<float>(static_cast<int>(lsu1) - 32);
        float acc_g1 = 0.0F, acc_u1 = 0.0F;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const std::uint32_t gc =
              (static_cast<std::uint32_t>(g1b.qs[sub * 16 + byte_base + j]) >> nib_shift) &
              0xFu;
          const std::uint32_t uc =
              (static_cast<std::uint32_t>(u1b.qs[sub * 16 + byte_base + j]) >> nib_shift) &
              0xFu;
          acc_g1 += s_iq4_lut[gc] * xs[j];
          acc_u1 += s_iq4_lut[uc] * xs[j];
        }
        sum_g1 += gscale1 * acc_g1;
        sum_u1 += uscale1 * acc_u1;
      }
    }
    for (int off = 16; off > 0; off >>= 1) {
      sum_g0 += __shfl_xor(sum_g0, off);
      sum_u0 += __shfl_xor(sum_u0, off);
      sum_g1 += __shfl_xor(sum_g1, off);
      sum_u1 += __shfl_xor(sum_u1, off);
    }
    if (lane == 0) {
      act_e[r0] = (sum_g0 / (1.0F + expf(-sum_g0))) * sum_u0;
      if (has_r1) {
        act_e[r0 + 1] = (sum_g1 / (1.0F + expf(-sum_g1))) * sum_u1;
      }
    }
    return;
  }

  if (gt == core::GgmlType::kQ6_K && ut == core::GgmlType::kQ6_K &&
      hidden % kQ6KBlockSize == 0) {
    // Q6_K 2-row loop (7.4b): one warp covers a whole 256-wide block per
    // pass, each lane owning 8 elements, so the block's ql/qh bytes are read
    // once instead of the generic sub16 route's ~2.4x amplification. x is
    // loaded once and shared by both rows and both matrices; one butterfly
    // per row-matrix at the end.
    const std::size_t blocks = hidden / kQ6KBlockSize;
    const auto* g_blocks = static_cast<const Q6KBlock*>(gate);
    const auto* u_blocks = static_cast<const Q6KBlock*>(up);
    float sum_g0 = 0.0F, sum_u0 = 0.0F, sum_g1 = 0.0F, sum_u1 = 0.0F;
    for (std::size_t b = 0; b < blocks; ++b) {
      const float* xb = x + (b * kQ6KBlockSize);
      const Q6KBlock& gb0 = g_blocks[(r0 * blocks) + b];
      const Q6KBlock& ub0 = u_blocks[(r0 * blocks) + b];
      sum_g0 += __half2float(gb0.d) * Q6KBlockDotLane(gb0, xb, lane);
      sum_u0 += __half2float(ub0.d) * Q6KBlockDotLane(ub0, xb, lane);
      if (has_r1) {
        const Q6KBlock& gb1 = g_blocks[((r0 + 1) * blocks) + b];
        const Q6KBlock& ub1 = u_blocks[((r0 + 1) * blocks) + b];
        sum_g1 += __half2float(gb1.d) * Q6KBlockDotLane(gb1, xb, lane);
        sum_u1 += __half2float(ub1.d) * Q6KBlockDotLane(ub1, xb, lane);
      }
    }
    for (int off = 16; off > 0; off >>= 1) {
      sum_g0 += __shfl_xor(sum_g0, off);
      sum_u0 += __shfl_xor(sum_u0, off);
      sum_g1 += __shfl_xor(sum_g1, off);
      sum_u1 += __shfl_xor(sum_u1, off);
    }
    if (lane == 0) {
      act_e[r0] = (sum_g0 / (1.0F + expf(-sum_g0))) * sum_u0;
      if (has_r1) {
        act_e[r0 + 1] = (sum_g1 / (1.0F + expf(-sum_g1))) * sum_u1;
      }
    }
    return;
  }

  const float g0 = QuantWarpBlockDot(gt, gate, r0, x, hidden, lane, 0, 1);
  const float u0 = QuantWarpBlockDot(ut, up, r0, x, hidden, lane, 0, 1);
  float g1 = 0.0F, u1 = 0.0F;
  if (has_r1) {
    g1 = QuantWarpBlockDot(gt, gate, r0 + 1, x, hidden, lane, 0, 1);
    u1 = QuantWarpBlockDot(ut, up, r0 + 1, x, hidden, lane, 0, 1);
  }
  if (lane == 0) {
    act_e[r0] = (g0 / (1.0F + expf(-g0))) * u0;
    if (has_r1) {
      act_e[r0 + 1] = (g1 / (1.0F + expf(-g1))) * u1;
    }
  }
}

__global__ void __launch_bounds__(kMoEWavesPerBlock * 32, 4)
    FlintFusedMoEDownKernel(
        const void* __restrict__ down_stack,
        const void* __restrict__ down_shared, core::GgmlType type,
        core::GgmlType shared_type, const float* __restrict__ act,
        float* __restrict__ partials, const float* __restrict__ weights_full,
        int expert_count, std::size_t expert_ff, std::size_t shared_ff,
        std::size_t hidden, std::size_t slab_bytes,
        const int* __restrict__ job_list) {
  const int e = job_list != nullptr
                    ? job_list[blockIdx.x]
                    : static_cast<int>(blockIdx.x);
  if (e < 0) {
    return;
  }
  const float w = weights_full[e];
  if (w == 0.0F) {
    return;
  }
  __shared__ float s_iq4_lut_d[16];
  if (static_cast<int>(threadIdx.x) < 16) {
    s_iq4_lut_d[threadIdx.x] = static_cast<float>(kIq4NlDeviceLut[threadIdx.x]);
  }
  __syncthreads();
  const bool is_shared = (e == expert_count);
  const std::size_t ff = is_shared ? shared_ff : expert_ff;
  // Deterministic accumulation (3.6): each job slot owns a disjoint
  // partials[job_slot * hidden + row] row — written once, no float atomics.
  // FlintFusedMoEReduceKernel sums the slots in fixed job order.
  const std::size_t job_slot = blockIdx.x;
  float* out_rows = partials + job_slot * hidden;
  const void* down = is_shared
                         ? down_shared
                         : static_cast<const char*>(down_stack) +
                               static_cast<std::size_t>(e) * slab_bytes;
  const core::GgmlType dt = is_shared ? shared_type : type;
  const float* act_e =
      act + (is_shared ? static_cast<std::size_t>(expert_count) * expert_ff
                       : static_cast<std::size_t>(e) * expert_ff);

  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp = threadIdx.x >> 5u;
  const std::size_t r0 =
      (static_cast<std::size_t>(blockIdx.y) * kMoEWavesPerBlock + warp) * 2;
  if (r0 >= hidden) {
    return;
  }
  const bool has_r1 = (r0 + 1) < hidden;

  if (dt == core::GgmlType::kQ8_0 && ff % kQ8_0BlockSize == 0) {
    const std::size_t num_blocks = ff / kQ8_0BlockSize;
    const auto* d0 = static_cast<const Q8_0Block*>(down) + (r0 * num_blocks);
    const auto* d1 =
        has_r1 ? static_cast<const Q8_0Block*>(down) + ((r0 + 1) * num_blocks)
               : nullptr;
    float sum0 = 0.0F, sum1 = 0.0F;
    for (std::size_t b = lane; b < num_blocks; b += 32) {
      const float* ab = act_e + (b * kQ8_0BlockSize);
      const Q8_0Block& db0 = d0[b];
      const float dd0 = __half2float(db0.d);
      float blk0 = 0.0F;
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        blk0 += static_cast<float>(db0.qs[i]) * ab[i];
      }
      sum0 += dd0 * blk0;
      if (d1 != nullptr) {
        const Q8_0Block& db1 = d1[b];
        const float dd1 = __half2float(db1.d);
        float blk1 = 0.0F;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          blk1 += static_cast<float>(db1.qs[i]) * ab[i];
        }
        sum1 += dd1 * blk1;
      }
    }
    for (int off = 16; off > 0; off >>= 1) {
      sum0 += __shfl_xor(sum0, off);
      sum1 += __shfl_xor(sum1, off);
    }
    if (lane == 0) {
      out_rows[r0] = w * sum0;
      if (has_r1) {
        out_rows[r0 + 1] = w * sum1;
      }
    }
    return;
  }

  if (false && dt == core::GgmlType::kIQ4_XS && ff % 256 == 0) {
    // IQ4_XS 2-row down loop: same shape as the SwiGLU path — one
    // butterfly per row, act_e shared across the lane's two rows.
    const std::size_t supers = ff / 256;
    const auto* d_blocks = static_cast<const IQ4XSBlock*>(down);
    const std::uint32_t quad = lane % 4u;
    const std::uint32_t byte_base = (quad % 2u) * 8u;
    const std::uint32_t nib_shift = (quad / 2u) * 4u;
    const std::uint32_t sub = lane / 4u;
    float sum0 = 0.0F, sum1 = 0.0F;
    for (std::size_t sb = 0; sb < supers; ++sb) {
      const float* ab = act_e + (sb * 256) + (lane * 8);
      const float4 aa = *reinterpret_cast<const float4*>(ab);
      const float4 aq = *reinterpret_cast<const float4*>(ab + 4);
      const float as[8] = {aa.x, aa.y, aa.z, aa.w, aq.x, aq.y, aq.z, aq.w};
      const IQ4XSBlock& d0b = d_blocks[(r0 * supers) + sb];
      const std::uint32_t ls_lo =
          (d0b.scales_l[sub / 2] >> (4u * (sub % 2u))) & 0xFu;
      const std::uint32_t ls0 =
          ls_lo | (((d0b.scales_h >> (2u * sub)) & 0x3u) << 4u);
      const float dscale0 =
          __half2float(d0b.d) * static_cast<float>(static_cast<int>(ls0) - 32);
      float acc0 = 0.0F;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const std::uint32_t code =
            (static_cast<std::uint32_t>(d0b.qs[sub * 16 + byte_base + j]) >> nib_shift) &
            0xFu;
        acc0 += s_iq4_lut_d[code] * as[j];
      }
      sum0 += dscale0 * acc0;
      if (has_r1) {
        const IQ4XSBlock& d1b = d_blocks[((r0 + 1) * supers) + sb];
        const std::uint32_t ls_lo1 =
            (d1b.scales_l[sub / 2] >> (4u * (sub % 2u))) & 0xFu;
        const std::uint32_t ls1 =
            ls_lo1 | (((d1b.scales_h >> (2u * sub)) & 0x3u) << 4u);
        const float dscale1 = __half2float(d1b.d) *
                              static_cast<float>(static_cast<int>(ls1) - 32);
        float acc1 = 0.0F;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const std::uint32_t code =
              (static_cast<std::uint32_t>(d1b.qs[sub * 16 + byte_base + j]) >>
               nib_shift) &
              0xFu;
          acc1 += s_iq4_lut_d[code] * as[j];
        }
        sum1 += dscale1 * acc1;
      }
    }
    for (int off = 16; off > 0; off >>= 1) {
      sum0 += __shfl_xor(sum0, off);
      sum1 += __shfl_xor(sum1, off);
    }
    if (lane == 0) {
      out_rows[r0] = w * sum0;
      if (has_r1) {
        out_rows[r0 + 1] = w * sum1;
      }
    }
    return;
  }

  const float s0 = QuantWarpBlockDot(dt, down, r0, act_e, ff, lane, 0, 1);
  float s1 = 0.0F;
  if (has_r1) {
    s1 = QuantWarpBlockDot(dt, down, r0 + 1, act_e, ff, lane, 0, 1);
  }
  if (lane == 0) {
    out_rows[r0] = w * s0;
    if (has_r1) {
      out_rows[r0 + 1] = w * s1;
    }
  }
}

}  // namespace

/// Deterministic MoE accumulation (3.6): the down kernel writes one
/// disjoint partials row per job slot; this kernel sums the slots in fixed
/// job-list order and writes the final mlp_out row. Replaces the previous
/// per-row atomicAdd — float atomic order is wave-scheduling dependent,
/// which surfaced as run-to-run logit noise (the serve "wander" bug).
__global__ void FlintFusedMoEReduceKernel(
    const float* __restrict__ partials, const int* __restrict__ job_list,
    const float* __restrict__ weights_full, float* __restrict__ mlp_out,
    float* __restrict__ residual_hidden, std::size_t hidden, int jobs) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= hidden) {
    return;
  }
  float acc = 0.0F;
  for (int j = 0; j < jobs; ++j) {
    const int e = job_list[j];
    if (e < 0) {
      continue;
    }
    if (weights_full[e] == 0.0F) {
      continue;  // the down kernel left this job's slot unwritten
    }
    acc += partials[static_cast<std::size_t>(j) * hidden + row];
  }
  if (residual_hidden != nullptr) {
    // Epilogue fusion (3.3): the FFN residual-add folds into the reduce —
    // hidden += MoE output — and mlp_out is not written at all.
    residual_hidden[row] += acc;
    return;
  }
  mlp_out[row] = acc;
}

void LaunchFlintRouterTopK(const float* logits, std::size_t expert_count,
                           std::uint32_t used, bool norm, float scale,
                           std::uint32_t* selected, float* weights,
                           hipStream_t stream) {
  FlintRouterTopKKernel<<<1, 256, 0, stream>>>(
      logits, static_cast<int>(expert_count), static_cast<int>(used),
      norm ? 1 : 0, scale, selected, weights, nullptr, 0.0F, nullptr, 0,
      nullptr);
}

void LaunchFlintRouterTopKFull(const float* logits, std::size_t expert_count,
                               std::uint32_t used, bool norm, float scale,
                               float* weights_full, float shared_weight,
                               float* zero_out, std::size_t zero_n,
                               int* job_list, hipStream_t stream) {
  FlintRouterTopKKernel<<<1, 256, 0, stream>>>(
      logits, static_cast<int>(expert_count), static_cast<int>(used),
      norm ? 1 : 0, scale, nullptr, nullptr, weights_full, shared_weight,
      zero_out, zero_n, job_list);
}

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
                               std::size_t up_slab_bytes, hipStream_t stream) {
  const std::size_t rows = std::max(expert_ff, shared_ff);
  const std::size_t jobs =
      job_list != nullptr ? max_jobs : expert_count + 1;
  const dim3 grid(jobs,
                  (rows + kMoERowsPerBlock - 1) / kMoERowsPerBlock);
  FlintFusedMoESwiGLUKernel<<<grid, kMoEWavesPerBlock * 32, 0, stream>>>(
      gate_stack, up_stack, gate_shared, up_shared, gate_type, up_type,
      shared_gate_type, shared_up_type, x, act, weights_full,
      static_cast<int>(expert_count), expert_ff, shared_ff, hidden,
      gate_slab_bytes, up_slab_bytes, job_list);
}

void LaunchFlintFusedMoEDown(const void* down_stack,
                             const void* down_shared, core::GgmlType type,
                             core::GgmlType shared_type, const float* act,
                             float* partials, float* mlp_out,
                             float* residual_hidden,
                             const float* weights_full,
                             const int* job_list, std::size_t max_jobs,
                             std::size_t expert_count, std::size_t expert_ff,
                             std::size_t shared_ff, std::size_t hidden,
                             std::size_t slab_bytes, hipStream_t stream) {
  const std::size_t jobs =
      job_list != nullptr ? max_jobs : expert_count + 1;
  const dim3 grid(jobs,
                  (hidden + kMoERowsPerBlock - 1) / kMoERowsPerBlock);
  FlintFusedMoEDownKernel<<<grid, kMoEWavesPerBlock * 32, 0, stream>>>(
      down_stack, down_shared, type, shared_type, act, partials, weights_full,
      static_cast<int>(expert_count), expert_ff, shared_ff, hidden,
      slab_bytes, job_list);
  FlintFusedMoEReduceKernel<<<static_cast<unsigned>((hidden + 255) / 256), 256,
                              0, stream>>>(
      partials, job_list, weights_full, mlp_out, residual_hidden, hidden,
      static_cast<int>(jobs));
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
