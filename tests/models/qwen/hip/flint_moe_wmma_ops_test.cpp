// F-L4 tuning (2.1): the grouped WMMA MoE route — gather -> grouped gate/up
// -> SwiGLU -> quantize -> grouped down -> combine — against a CPU reference
// over dequantized weights and the device's own Q8_1 activation codes. The
// reference re-quantizes on the CPU with the same per-32-block semantics, so
// the comparison isolates the kernels' structure (tile table, expert
// resolution, slot indirection, fragment epilogues) from quantisation noise.
//
// Small shapes by design: this is the fast gate that runs in seconds before
// the 36 GB real-artifact smoke. The flint shapes (E=256, ff=512, H=2048) are
// covered by the chunk mode of flint_real_smoke_test.
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "tests/models/qwen/hip/support/bfloat16.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"

#if defined(ENGINE_ENABLE_HIP)

namespace {

using gufo::core::GgmlType;

#define HIP_CHECK(call)                                                     \
  do {                                                                      \
    const hipError_t status__ = (call);                                     \
    if (status__ != hipSuccess) {                                           \
      std::cerr << "HIP failure at " << __FILE__ << ":" << __LINE__ << ": " \
                << hipGetErrorString(status__) << "\n";                     \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

void Expect(bool condition, const char* msg) {
  if (!condition) {
    std::cerr << "FAIL: " << msg << "\n";
    std::exit(1);
  }
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

float Uniform(std::uint32_t* state, float scale) {
  return scale * static_cast<float>(static_cast<int>(NextRandom(state) &
                                                     0xFFFFU) -
                                    32768) /
         32768.0F;
}

// Full-mantissa, non-power-of-two scales so summation order matters (the
// kquant_gemv rationale): a power of two hides reassociation bugs. Small
// values keep the synthetic SwiGLU outputs from spanning enough dynamic range
// that the down-projection dot cancels catastrophically (f32 rounding of the
// kernel is then the dominant error term, not the kernel's structure).
std::uint16_t TestScaleHalf(std::size_t block) noexcept {
  static constexpr std::uint16_t kScales[4] = {0x2AAAU, 0x28CCU, 0x2B33U,
                                               0x2955U};
  return kScales[block % 4];
}

void FillQ6_K(std::vector<std::uint8_t>& bytes, std::size_t rows,
              std::size_t row_bytes, std::uint32_t* rng) {
  for (std::size_t row = 0; row < rows; ++row) {
    auto* blocks = reinterpret_cast<gufo::quant::block_q6_K*>(
        bytes.data() + row * row_bytes);
    const std::size_t count = row_bytes / sizeof(gufo::quant::block_q6_K);
    for (std::size_t b = 0; b < count; ++b) {
      blocks[b].d = TestScaleHalf(b);
      for (auto& value : blocks[b].ql) {
        value = static_cast<std::uint8_t>(NextRandom(rng) & 0xFFU);
      }
      for (auto& value : blocks[b].qh) {
        value = static_cast<std::uint8_t>(NextRandom(rng) & 0xFFU);
      }
      for (auto& value : blocks[b].scales) {
        value = static_cast<std::int8_t>((NextRandom(rng) % 15U) + 1U);
      }
    }
  }
}

void FillQ8_0(std::vector<std::uint8_t>& bytes, std::size_t rows,
              std::size_t row_bytes, std::uint32_t* rng) {
  for (std::size_t row = 0; row < rows; ++row) {
    auto* blocks = reinterpret_cast<gufo::quant::block_q8_0*>(
        bytes.data() + row * row_bytes);
    const std::size_t count = row_bytes / sizeof(gufo::quant::block_q8_0);
    for (std::size_t b = 0; b < count; ++b) {
      blocks[b].d = TestScaleHalf(b);
      for (auto& value : blocks[b].qs) {
        value = static_cast<std::int8_t>(
            static_cast<int>(NextRandom(rng) % 255U) - 127);
      }
    }
  }
}

void FillIQ4_XS(std::vector<std::uint8_t>& bytes, std::size_t rows,
                std::size_t row_bytes, std::uint32_t* rng) {
  for (std::size_t row = 0; row < rows; ++row) {
    auto* blocks = reinterpret_cast<gufo::quant::block_iq4_xs*>(
        bytes.data() + row * row_bytes);
    const std::size_t count = row_bytes / sizeof(gufo::quant::block_iq4_xs);
    for (std::size_t b = 0; b < count; ++b) {
      blocks[b].d = TestScaleHalf(b);
      blocks[b].scales_h =
          static_cast<std::uint16_t>(NextRandom(rng) & 0xFFFFU);
      for (auto& value : blocks[b].scales_l) {
        value = static_cast<std::uint8_t>(NextRandom(rng) & 0xFFU);
      }
      for (auto& value : blocks[b].qs) {
        value = static_cast<std::uint8_t>(NextRandom(rng) & 0xFFU);
      }
    }
  }
}

void FillStack(std::vector<std::uint8_t>& bytes, std::size_t rows,
               std::size_t row_bytes, GgmlType type, std::uint32_t* rng) {
  switch (type) {
    case GgmlType::kQ6_K:
      FillQ6_K(bytes, rows, row_bytes, rng);
      return;
    case GgmlType::kQ8_0:
      FillQ8_0(bytes, rows, row_bytes, rng);
      return;
    case GgmlType::kIQ4_XS:
      FillIQ4_XS(bytes, rows, row_bytes, rng);
      return;
    default:
      std::cerr << "FillStack: unsupported type\n";
      std::exit(1);
  }
}

void DequantRow(const void* src, float* dst, GgmlType type, std::size_t k) {
  switch (type) {
    case GgmlType::kQ6_K:
      gufo::quant::DequantizeQ6_K(src, dst, k);
      return;
    case GgmlType::kQ8_0:
      gufo::quant::DequantizeQ8_0(src, dst, k);
      return;
    case GgmlType::kIQ4_XS:
      gufo::quant::DequantizeIQ4_XS(src, dst, k);
      return;
    default:
      std::cerr << "DequantRow: unsupported type\n";
      std::exit(1);
  }
}

/// Decodes the tiled Q8_1 buffer (16-token tiles of 576-byte K blocks with
/// inline fp32 scales) into row-major f32.
std::vector<float> DecodeTiledQ8(const std::vector<std::uint8_t>& bytes,
                                 std::size_t rows, std::size_t k) {
  const std::size_t nb = k / 32;
  std::vector<float> out(rows * k, 0.0F);
  for (std::size_t r = 0; r < rows; ++r) {
    const std::size_t tt = r / 16, tl = r % 16;
    for (std::size_t kb = 0; kb < nb; ++kb) {
      const std::uint8_t* tile = bytes.data() + ((tt * nb + kb) * 576);
      float scale;
      std::memcpy(&scale, tile + 512 + tl * 4, sizeof(float));
      for (std::size_t j = 0; j < 16; ++j) {
        out[r * k + kb * 32 + j] =
            scale * static_cast<float>(
                        static_cast<std::int8_t>(tile[tl * 16 + j]));
        out[r * k + kb * 32 + 16 + j] =
            scale * static_cast<float>(
                        static_cast<std::int8_t>(tile[256 + tl * 16 + j]));
      }
    }
  }
  return out;
}

/// CPU mirror of the device quantizer's dequantized value (d = max/127,
/// q = roundf(x/d)) used to build the reference's down-projection input.
std::vector<float> QuantizeDequantCpu(const std::vector<float>& rows,
                                      std::size_t k) {
  std::vector<float> out(rows.size());
  const std::size_t nb = k / 32;
  for (std::size_t r = 0; r * k < rows.size(); ++r) {
    for (std::size_t b = 0; b < nb; ++b) {
      float max_abs = 0.0F;
      for (std::size_t j = 0; j < 32; ++j) {
        max_abs = std::fmax(max_abs, std::fabs(rows[r * k + b * 32 + j]));
      }
      const float d = max_abs / 127.0F;
      const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
      for (std::size_t j = 0; j < 32; ++j) {
        const auto q =
            static_cast<float>(static_cast<std::int8_t>(
                std::roundf(rows[r * k + b * 32 + j] * id)));
        out[r * k + b * 32 + j] = d * q;
      }
    }
  }
  return out;
}

double DotF64(const float* a, const float* b, std::size_t n) {
  double acc = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return acc;
}

int RunCase(std::size_t n, std::size_t used, std::size_t experts,
            std::size_t hidden, std::size_t expert_ff, GgmlType gate_up_type,
            GgmlType down_type, std::uint32_t seed) {
  std::cout << "grouped wmma: n=" << n << " used=" << used
            << " E=" << experts << " H=" << hidden << " ff=" << expert_ff
            << " gate/up=" << gufo::core::ToString(gate_up_type)
            << " down=" << gufo::core::ToString(down_type) << "\n";
  Expect(gufo::hip::FlintGroupedWmmaSupported(gate_up_type, gate_up_type,
                                              down_type, expert_ff, hidden),
         "case shapes must take the grouped WMMA route");
  // IQ4_XS decodes 136-byte super-blocks of 256 elements; a shorter row makes
  // DecodeQuantSub16 read past the slab (and the CPU reference loops over
  // k/256 super-blocks, producing zeros).
  const auto superblock_ok = [](GgmlType t, std::size_t k) {
    return t != GgmlType::kIQ4_XS || k % 256 == 0;
  };
  Expect(superblock_ok(gate_up_type, hidden) && superblock_ok(down_type, expert_ff),
         "IQ4_XS dims must be multiples of 256");

  const std::size_t slots = n * used;
  const std::size_t cap =
      (slots + 15 * experts + 15) & ~static_cast<std::size_t>(15);
  const std::size_t tiles_bound = cap / 16;
  const std::size_t gate_row =
      gufo::quant::QuantizedRowBytes(gate_up_type, hidden);
  const std::size_t down_row =
      gufo::quant::QuantizedRowBytes(down_type, expert_ff);
  const std::size_t slab_gate = expert_ff * gate_row;
  const std::size_t slab_down = hidden * down_row;

  std::uint32_t rng = seed;

  // Activations.
  std::vector<float> h_x(n * hidden);
  for (auto& v : h_x) {
    v = Uniform(&rng, 1.5F);
  }

  // Routing: `used` distinct experts per token.
  std::vector<std::uint32_t> h_selected(slots);
  std::vector<float> h_sel_weights(slots);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t j = 0; j < used; ++j) {
      std::uint32_t e;
      bool dup;
      do {
        e = NextRandom(&rng) % experts;
        dup = false;
        for (std::size_t i = 0; i < j; ++i) {
          dup = dup || (h_selected[t * used + i] == e);
        }
      } while (dup);
      h_selected[t * used + j] = e;
      h_sel_weights[t * used + j] = 0.1F + Uniform(&rng, 0.45F) + 0.45F;
    }
  }

  std::vector<float> h_shared(n * hidden);
  for (auto& v : h_shared) {
    v = Uniform(&rng, 0.3F);
  }

  // Expert stacks: gate/up [expert_ff, hidden] and down [hidden, expert_ff]
  // in the case's quant types.
  std::vector<std::uint8_t> h_gate(experts * slab_gate, 0);
  std::vector<std::uint8_t> h_up(experts * slab_gate, 0);
  std::vector<std::uint8_t> h_down(experts * slab_down, 0);
  FillStack(h_gate, experts * expert_ff, gate_row, gate_up_type, &rng);
  FillStack(h_up, experts * expert_ff, gate_row, gate_up_type, &rng);
  FillStack(h_down, experts * hidden, down_row, down_type, &rng);

  // Dequantized weight images for the reference.
  std::vector<float> w_gate(experts * expert_ff * hidden);
  std::vector<float> w_up(experts * expert_ff * hidden);
  std::vector<float> w_down(experts * hidden * expert_ff);
  for (std::size_t e = 0; e < experts; ++e) {
    for (std::size_t r = 0; r < expert_ff; ++r) {
      DequantRow(h_gate.data() + (e * expert_ff + r) * gate_row,
                 w_gate.data() + (e * expert_ff + r) * hidden, gate_up_type,
                 hidden);
      DequantRow(h_up.data() + (e * expert_ff + r) * gate_row,
                 w_up.data() + (e * expert_ff + r) * hidden, gate_up_type,
                 hidden);
    }
    for (std::size_t r = 0; r < hidden; ++r) {
      DequantRow(h_down.data() + (e * hidden + r) * down_row,
                 w_down.data() + (e * hidden + r) * expert_ff, down_type,
                 expert_ff);
    }
  }

  // --- device ------------------------------------------------------------
  float *d_x = nullptr, *d_shared = nullptr, *d_gate = nullptr, *d_up = nullptr,
        *d_down = nullptr, *d_gate_out = nullptr, *d_up_out = nullptr,
        *d_down_out = nullptr, *d_mlp = nullptr, *d_slot_w = nullptr;
  void *d_xq = nullptr, *d_slotq = nullptr, *d_actq = nullptr;
  std::uint32_t* d_selected = nullptr;
  float* d_sel_w = nullptr;
  int *d_counts = nullptr, *d_offsets = nullptr, *d_cursors = nullptr;
  int *d_slot_tokens = nullptr, *d_token_slots = nullptr;
  int *d_tile_expert = nullptr, *d_tile_slot0 = nullptr, *d_tile_count = nullptr;

  const std::size_t xq_bytes =
      gufo::hip::QuantizedActivationBytes(n, hidden);
  const std::size_t slotq_bytes =
      gufo::hip::QuantizedActivationBytes(cap, hidden);
  const std::size_t actq_bytes =
      gufo::hip::QuantizedActivationBytes(cap, expert_ff);

  HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_shared, h_shared.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, h_gate.size()));
  HIP_CHECK(hipMalloc(&d_up, h_up.size()));
  HIP_CHECK(hipMalloc(&d_down, h_down.size()));
  HIP_CHECK(hipMalloc(&d_gate_out, cap * expert_ff * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_up_out, cap * expert_ff * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_down_out, cap * hidden * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_mlp, n * hidden * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_xq, xq_bytes));
  HIP_CHECK(hipMalloc(&d_slotq, slotq_bytes));
  HIP_CHECK(hipMalloc(&d_actq, actq_bytes));
  HIP_CHECK(hipMalloc(&d_selected, slots * sizeof(std::uint32_t)));
  HIP_CHECK(hipMalloc(&d_sel_w, slots * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_counts, experts * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_offsets, (experts + 1) * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_cursors, experts * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_slot_tokens, cap * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_slot_w, cap * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_token_slots, slots * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_tile_expert, tiles_bound * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_tile_slot0, tiles_bound * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_tile_count, sizeof(int)));

  HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_shared, h_shared.data(), h_shared.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), h_gate.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_up, h_up.data(), h_up.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_down, h_down.data(), h_down.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_selected, h_selected.data(),
                      slots * sizeof(std::uint32_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_sel_w, h_sel_weights.data(), slots * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_gate_out, 0, cap * expert_ff * sizeof(float)));
  HIP_CHECK(hipMemset(d_up_out, 0, cap * expert_ff * sizeof(float)));
  HIP_CHECK(hipMemset(d_down_out, 0, cap * hidden * sizeof(float)));
  HIP_CHECK(hipMemset(d_mlp, 0, n * hidden * sizeof(float)));

  gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(d_x, d_xq, n, hidden);
  gufo::hip::LaunchFlintMoeSlots(d_selected, d_sel_w, n, used, experts,
                                 d_counts, d_offsets, d_cursors, d_slot_tokens,
                                 d_slot_w, d_token_slots, d_tile_expert,
                                 d_tile_slot0, d_tile_count);
  gufo::hip::LaunchFlintGatherQ8Act(d_xq, d_slotq, d_slot_tokens, d_offsets,
                                    experts, cap, hidden);
  gufo::hip::LaunchFlintMoeGroupedGateUpWmma(
      d_gate, d_up, gate_up_type, d_slotq, d_gate_out, d_up_out, d_offsets,
      d_tile_expert, d_tile_slot0, d_tile_count, tiles_bound, experts,
      expert_ff, hidden, slab_gate);
  // Snapshot gate/up before the in-place SwiGLU rewrites d_gate_out.
  std::vector<float> h_gate_out(cap * expert_ff), h_up_out(cap * expert_ff);
  HIP_CHECK(hipMemcpy(h_gate_out.data(), d_gate_out,
                      cap * expert_ff * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_up_out.data(), d_up_out,
                      cap * expert_ff * sizeof(float), hipMemcpyDeviceToHost));
  gufo::hip::LaunchBatchedSwiGLUActivation(d_gate_out, d_up_out, d_gate_out,
                                           nullptr, cap * expert_ff);
  gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(d_gate_out, d_actq, cap,
                                                  expert_ff);
  gufo::hip::LaunchFlintMoeGroupedDownWmma(
      d_down, down_type, d_actq, d_down_out, d_offsets, d_slot_w, d_tile_expert,
      d_tile_slot0, d_tile_count, tiles_bound, experts, expert_ff, hidden,
      slab_down);
  gufo::hip::LaunchFlintMoeCombine(d_shared, d_down_out, d_token_slots, d_mlp,
                                   n, used, hidden);
  HIP_CHECK(hipDeviceSynchronize());

  // --- slot-table invariants ----------------------------------------------
  std::vector<int> counts(experts), offsets(experts + 1);
  std::vector<int> slot_tokens(cap), token_slots(slots);
  std::vector<float> slot_weights(cap);
  std::vector<int> tile_expert(tiles_bound), tile_slot0(tiles_bound),
      tile_count(1);
  HIP_CHECK(hipMemcpy(counts.data(), d_counts, experts * sizeof(int),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(offsets.data(), d_offsets, (experts + 1) * sizeof(int),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(slot_tokens.data(), d_slot_tokens, cap * sizeof(int),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(token_slots.data(), d_token_slots, slots * sizeof(int),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(slot_weights.data(), d_slot_w, cap * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tile_expert.data(), d_tile_expert,
                      tiles_bound * sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tile_slot0.data(), d_tile_slot0,
                      tiles_bound * sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tile_count.data(), d_tile_count, sizeof(int),
                      hipMemcpyDeviceToHost));

  int expected_tiles = 0;
  std::size_t total = 0;
  for (std::size_t e = 0; e < experts; ++e) {
    Expect(offsets[e] % 16 == 0, "bucket starts must be 16-aligned");
    const int padded = (counts[e] + 15) & ~15;
    Expect(static_cast<std::size_t>(offsets[e] + padded) == offsets[e + 1],
           "offsets must be the padded running sum");
    expected_tiles += padded / 16;
    total += static_cast<std::size_t>(counts[e]);
    for (int s = offsets[e] + counts[e]; s < offsets[e + 1]; ++s) {
      Expect(slot_tokens[s] == 0, "pad slot_tokens must be zeroed");
      Expect(slot_weights[s] == 0.0F, "pad slot_weights must be zeroed");
    }
  }
  Expect(total == slots, "counts must sum to n * used");
  Expect(tile_count[0] == expected_tiles, "tile count must match the table");
  Expect(static_cast<std::size_t>(tile_count[0]) <= tiles_bound,
         "tile count must fit the host bound");
  for (int t = 0; t < tile_count[0]; ++t) {
    const int e = tile_expert[t];
    Expect(e >= 0 && static_cast<std::size_t>(e) < experts,
           "tile expert in range");
    Expect(static_cast<std::size_t>(tile_slot0[t]) % 16 == 0 &&
               tile_slot0[t] >= offsets[e] &&
               tile_slot0[t] < offsets[e + 1],
           "tile slot base inside its expert's padded bucket");
  }
  // token_slots -> slot tables consistency (selection order).
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t j = 0; j < used; ++j) {
      const int s = token_slots[t * used + j];
      Expect(static_cast<std::size_t>(s) < cap && slot_tokens[s] ==
                 static_cast<int>(t),
             "token_slots must reference the token's slots");
      Expect(slot_weights[s] == h_sel_weights[t * used + j],
             "slot weights must match the selection");
    }
  }

  // --- numeric reference ---------------------------------------------------
  std::vector<std::uint8_t> h_xq(xq_bytes);
  HIP_CHECK(hipMemcpy(h_xq.data(), d_xq, xq_bytes, hipMemcpyDeviceToHost));
  const std::vector<float> xq = DecodeTiledQ8(h_xq, n, hidden);

  // gate/up per real slot.
  std::vector<double> worst_gate(2, 0.0);
  for (std::size_t e = 0; e < experts; ++e) {
    for (int s = offsets[e]; s < offsets[e] + counts[e]; ++s) {
      const std::size_t t = static_cast<std::size_t>(slot_tokens[s]);
      for (std::size_t r = 0; r < expert_ff; ++r) {
        const double rg =
            DotF64(w_gate.data() + (e * expert_ff + r) * hidden,
                   xq.data() + t * hidden, hidden);
        const double ru =
            DotF64(w_up.data() + (e * expert_ff + r) * hidden,
                   xq.data() + t * hidden, hidden);
        const double eg = std::fabs(rg - h_gate_out[static_cast<std::size_t>(s) *
                                                     expert_ff + r]) /
                          (1.0 + std::fabs(rg));
        const double eu = std::fabs(ru - h_up_out[static_cast<std::size_t>(s) *
                                                   expert_ff + r]) /
                          (1.0 + std::fabs(ru));
        worst_gate[0] = std::fmax(worst_gate[0], eg);
        worst_gate[1] = std::fmax(worst_gate[1], eu);
      }
    }
  }
  std::cout << "  gate/up: max_rel=" << worst_gate[0] << " / "
            << worst_gate[1] << "\n";
  if (!(worst_gate[0] < 5e-3 && worst_gate[1] < 5e-3)) {
    // First mismatch details, to steer debugging fast.
    std::size_t e0 = experts, r0 = 0, t0 = 0;
    double got0 = 0, ref0 = 0;
    for (std::size_t e = 0; e < experts && e0 == experts; ++e) {
      for (int s = offsets[e]; s < offsets[e] + counts[e] && e0 == experts; ++s) {
        const std::size_t t = static_cast<std::size_t>(slot_tokens[s]);
        for (std::size_t r = 0; r < expert_ff && e0 == experts; ++r) {
          const double rg = DotF64(
              w_gate.data() + (e * expert_ff + r) * hidden,
              xq.data() + t * hidden, hidden);
          const double got =
              h_gate_out[static_cast<std::size_t>(s) * expert_ff + r];
          if (std::fabs(got - rg) > 5e-3 * (1.0 + std::fabs(rg))) {
            e0 = e;
            r0 = r;
            t0 = t;
            got0 = got;
            ref0 = rg;
          }
        }
      }
    }
    std::cout << "    first gate mismatch: e=" << e0 << " s-slot r=" << r0
              << " tok=" << t0 << " got=" << got0 << " ref=" << ref0 << "\n";
    Expect(false, "grouped gate/up must match the dequantized reference");
  }

  // SwiGLU + requantize + down. The device SwiGLU ran over all padded rows;
  // reference rows exist only for real slots.
  std::vector<float> ref_act(cap * expert_ff, 0.0F);
  for (std::size_t e = 0; e < experts; ++e) {
    for (int s = offsets[e]; s < offsets[e] + counts[e]; ++s) {
      for (std::size_t r = 0; r < expert_ff; ++r) {
        const float g = h_gate_out[static_cast<std::size_t>(s) * expert_ff + r];
        const float u = h_up_out[static_cast<std::size_t>(s) * expert_ff + r];
        ref_act[static_cast<std::size_t>(s) * expert_ff + r] =
            (g / (1.0F + std::exp(-g))) * u;
      }
    }
  }
  const std::vector<float> ref_actq = QuantizeDequantCpu(ref_act, expert_ff);

  // The device's own quantized activations: the kernel-only down reference
  // (the CPU requantize of the SwiGLU output differs from the device quantize
  // by 1-ulp expf rounding, which flips a code near a .5 boundary and costs
  // ~1 code over the k-dot — quantisation noise, not kernel error).
  std::vector<std::uint8_t> h_actq(actq_bytes);
  HIP_CHECK(hipMemcpy(h_actq.data(), d_actq, actq_bytes, hipMemcpyDeviceToHost));
  const std::vector<float> actq_dev = DecodeTiledQ8(h_actq, cap, expert_ff);

  std::vector<float> h_down_out(cap * hidden);
  HIP_CHECK(hipMemcpy(h_down_out.data(), d_down_out,
                      cap * hidden * sizeof(float), hipMemcpyDeviceToHost));
  // Normalise by the computation's own magnitude (sum of per-block partial
  // magnitudes), not by |ref|: the k-dot can cancel several orders of
  // magnitude on synthetic data, and f32 rounding of the terms then dwarfs
  // |ref|. A structural bug (wrong expert/row/stage/slot) still produces an
  // O(1) error under this metric; f32 rounding sits at ~1e-7.
  double worst_down = 0.0;
  {
    const std::size_t nb = expert_ff / 32;
    for (std::size_t e = 0; e < experts; ++e) {
      for (int s = offsets[e]; s < offsets[e] + counts[e]; ++s) {
        for (std::size_t r = 0; r < hidden; ++r) {
          const float* w = w_down.data() + (e * hidden + r) * expert_ff;
          const float* a =
              actq_dev.data() + static_cast<std::size_t>(s) * expert_ff;
          double ref = 0.0;
          double scale = 0.0;
          for (std::size_t b = 0; b < nb; ++b) {
            const double p = DotF64(w + b * 32, a + b * 32, 32);
            ref += p;
            scale += std::fabs(p);
          }
          ref *= slot_weights[s];
          const double got =
              h_down_out[static_cast<std::size_t>(s) * hidden + r];
          const double ed = std::fabs(got - ref) / (1.0 + scale);
          worst_down = std::fmax(worst_down, ed);
        }
      }
    }
  }
  std::cout << "  down: max_err=" << worst_down << "\n";
  Expect(worst_down < 1e-5,
         "grouped down must match the dequantized reference");
  // Chain check: the device quantize of the device SwiGLU output against the
  // CPU mirror. They may differ by one code per element (expf rounding);
  // more than one code means the SwiGLU or quantize diverged.
  {
    const std::size_t nb = expert_ff / 32;
    std::size_t flips = 0;
    for (std::size_t e = 0; e < experts; ++e) {
      for (int s = offsets[e]; s < offsets[e] + counts[e]; ++s) {
        for (std::size_t b = 0; b < nb; ++b) {
          float d_cpu = 0.0F;
          for (std::size_t j = 0; j < 32; ++j) {
            d_cpu = std::fmax(
                d_cpu, std::fabs(ref_act[static_cast<std::size_t>(s) *
                                              expert_ff +
                                          b * 32 + j]));
          }
          d_cpu /= 127.0F;
          for (std::size_t j = 0; j < 32; ++j) {
            const std::size_t idx =
                static_cast<std::size_t>(s) * expert_ff + b * 32 + j;
            if (std::fabs(actq_dev[idx] - ref_actq[idx]) > 1.5F * d_cpu) {
              ++flips;
            }
          }
        }
      }
    }
    std::cout << "  actq chain: >1-code flips=" << flips << "\n";
    Expect(flips == 0, "SwiGLU+quantize chain must match the CPU mirror");
  }

  // Combine in selection order.
  std::vector<float> h_mlp(n * hidden);
  HIP_CHECK(hipMemcpy(h_mlp.data(), d_mlp, n * hidden * sizeof(float),
                      hipMemcpyDeviceToHost));
  double worst_mlp = 0.0;
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t r = 0; r < hidden; ++r) {
      double acc = h_shared[t * hidden + r];
      for (std::size_t j = 0; j < used; ++j) {
        const int s = token_slots[t * used + j];
        acc += h_down_out[static_cast<std::size_t>(s) * hidden + r];
      }
      const double e =
          std::fabs(acc - h_mlp[t * hidden + r]) / (1.0 + std::fabs(acc));
      worst_mlp = std::fmax(worst_mlp, e);
    }
  }
  std::cout << "  mlp: max_rel=" << worst_mlp << "\n";
  Expect(worst_mlp < 1e-3, "combine must sum the down rows in selection order");

  for (std::size_t i = 0; i < h_mlp.size(); ++i) {
    if (!std::isfinite(h_mlp[i])) {
      Expect(false, "combine output must be finite");
    }
  }

  hipFree(d_x);
  hipFree(d_shared);
  hipFree(d_gate);
  hipFree(d_up);
  hipFree(d_down);
  hipFree(d_gate_out);
  hipFree(d_up_out);
  hipFree(d_down_out);
  hipFree(d_mlp);
  hipFree(d_slot_w);
  hipFree(d_xq);
  hipFree(d_slotq);
  hipFree(d_actq);
  hipFree(d_selected);
  hipFree(d_sel_w);
  hipFree(d_counts);
  hipFree(d_offsets);
  hipFree(d_cursors);
  hipFree(d_slot_tokens);
  hipFree(d_token_slots);
  hipFree(d_tile_expert);
  hipFree(d_tile_slot0);
  hipFree(d_tile_count);
  return 0;
}

}  // namespace

/// Router GEMM gate: tiled logits vs an fp32 CPU reference, and the G6
/// property that a row's logits are bit-identical regardless of how many
/// tokens share the launch (the k-order is fixed per output).
void RunRouterCase(std::size_t n_tokens, std::size_t experts,
                   std::size_t hidden, int w_storage,
                   std::uint32_t seed) {
  // w_storage: 0 = f32, 1 = f16, 2 = bf16 (the artifact's router format).
  std::uint32_t rng = seed;
  std::vector<float> h_x(n_tokens * hidden);
  for (auto& v : h_x) {
    v = Uniform(&rng, 1.0F);
  }
  std::vector<std::uint16_t> h_w16;
  std::vector<float> h_w32;
  const std::size_t w_elems = experts * hidden;
  if (w_storage == 1) {
    h_w16.resize(w_elems);
    for (auto& v : h_w16) {
      const __half h = __float2half(Uniform(&rng, 0.05F));
      std::memcpy(&v, &h, sizeof(v));
    }
  } else if (w_storage == 2) {
    h_w16.resize(w_elems);
    for (auto& v : h_w16) {
      v = gufo::test::FloatToBf16Bits(Uniform(&rng, 0.05F));
    }
  } else {
    h_w32.resize(w_elems);
    for (auto& v : h_w32) {
      v = Uniform(&rng, 0.05F);
    }
  }

  const std::size_t w_elem_bytes = w_storage == 0 ? 4 : 2;
  const std::size_t logits_bytes = n_tokens * experts * sizeof(float);
  float* d_x = nullptr;
  void* d_w = nullptr;
  float* d_logits = nullptr;
  float* d_split = nullptr;
  HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, w_elems * w_elem_bytes));
  HIP_CHECK(hipMalloc(&d_logits, logits_bytes));
  HIP_CHECK(hipMalloc(&d_split, logits_bytes));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w,
                      w_storage == 0 ? (const void*)h_w32.data()
                                     : (const void*)h_w16.data(),
                      w_elems * w_elem_bytes, hipMemcpyHostToDevice));

  const GgmlType wtype = w_storage == 0    ? GgmlType::kF32
                         : w_storage == 1  ? GgmlType::kF16
                                           : GgmlType::kBF16;
  gufo::hip::LaunchFlintRouterBatch(d_x, n_tokens, d_w, wtype, experts, hidden,
                                    d_logits);

  // Split launch: the first half of the tokens alone must give bit-identical
  // logits (fixed per-output k-order).
  const std::size_t n_split = n_tokens / 2;
  gufo::hip::LaunchFlintRouterBatch(d_x, n_split, d_w, wtype, experts, hidden,
                                    d_split);
  std::vector<float> h_logits(n_tokens * experts);
  std::vector<float> h_split(n_split * experts);
  HIP_CHECK(
      hipMemcpy(h_logits.data(), d_logits, logits_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_split.data(), d_split, n_split * experts * sizeof(float),
                      hipMemcpyDeviceToHost));
  for (std::size_t i = 0; i < n_split * experts; ++i) {
    Expect(std::memcmp(&h_logits[i], &h_split[i], sizeof(float)) == 0,
           "router logits must be bit-identical across chunk splits");
  }

  double worst = 0.0;
  for (std::size_t t = 0; t < n_tokens; ++t) {
    for (std::size_t e = 0; e < experts; ++e) {
      double ref = 0.0;
      for (std::size_t k = 0; k < hidden; ++k) {
        double wv;
        if (w_storage == 1) {
          __half h;
          std::memcpy(&h, &h_w16[e * hidden + k], sizeof(h));
          wv = __half2float(h);
        } else if (w_storage == 2) {
          wv = gufo::test::Bf16BitsToFloat(h_w16[e * hidden + k]);
        } else {
          wv = h_w32[e * hidden + k];
        }
        ref += h_x[t * hidden + k] * wv;
      }
      const double got = h_logits[t * experts + e];
      const double rel = std::fabs(got - ref) / (1.0 + std::fabs(ref));
      worst = std::fmax(worst, rel);
      Expect(std::isfinite(got), "router logit must be finite");
    }
  }
  std::cout << "  router n=" << n_tokens << " E=" << experts
            << " H=" << hidden
            << (w_storage == 1 ? " f16" : (w_storage == 2 ? " bf16" : " f32"))
            << ": max_rel=" << worst << "\n";
  Expect(worst < 1e-4, "router logits must match the fp32 reference");

  hipFree(d_x);
  hipFree(d_w);
  hipFree(d_logits);
  hipFree(d_split);
}

int main() {
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cout << "no HIP device: skipping\n";
    return 77;
  }
  // Small case: tiny buckets (empty, partial, and multi-row) at 16-tile
  // granularity. Q6_K gate/up + Q8_0 down is the UD-Q6_K_XL production route.
  RunCase(40, 3, 8, 256, 128, GgmlType::kQ6_K, GgmlType::kQ8_0, 0xC0FFEEU);
  // Larger case: multi-tile buckets and more experts.
  RunCase(100, 6, 16, 512, 256, GgmlType::kQ6_K, GgmlType::kQ8_0, 0xBADC0DEU);
  // IQ4_XS artifact route: uniform IQ4_XS gate/up/down (bartowski layers 3-47).
  // IQ4_XS rows are 136-byte super-blocks of 256, so both k dims must be
  // multiples of 256 — smaller rows decode past the slab (the production
  // shapes are hidden=4096, ff=6144).
  RunCase(40, 3, 8, 256, 256, GgmlType::kIQ4_XS, GgmlType::kIQ4_XS, 0x10415U);
  RunCase(100, 6, 16, 512, 256, GgmlType::kIQ4_XS, GgmlType::kIQ4_XS,
          0x4C5BADU);
  // Mixed: IQ4_XS gate/up with Q6_K down (the fused blk.48 shape minus its
  // Q5_K gate) at a flint-adjacent shape.
  RunCase(64, 4, 8, 512, 256, GgmlType::kIQ4_XS, GgmlType::kQ6_K, 0x06C45U);
  // Router: partial expert tile + partial token tile (37 tokens, 8 experts
  // against 32x32 block tiles), all three weight storages — bf16 is the
  // artifact's actual router format (F-C2); and the flint shape.
  RunRouterCase(37, 8, 256, 1, 0xA11CEU);
  RunRouterCase(37, 8, 256, 2, 0xA11CEU);
  RunRouterCase(37, 8, 256, 0, 0xBEEFU);
  RunRouterCase(512, 256, 2048, 2, 0xF1167U);
  std::cout << "flint grouped wmma ops: PASS\n";
  return 0;
}

#else
int main() {
  return 77;
}
#endif
