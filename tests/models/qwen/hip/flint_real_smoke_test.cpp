// Flint real-artifact GPU smoke test (F-D4 dispatch entry, real shapes).
// Env-gated on GUFO_FLINT_GGUF (skips with 77 when absent): loads the pinned
// flint GGUF once, runs the same greedy token sequence through the CPU
// reference (FlintForwardModel, the D4 oracle) and the GPU decode step
// (ExecuteFlintDecodeStep over the mapped-registration weights of
// FlintGpuModel), and compares logits step by step in the G3 format
// (cosine >= 0.999, max |delta log p| <= 5e-3, argmax identity). The first
// pass warms the UMM pages; the second pass is timed for decode t/s.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/flint_forward.hpp"
#include "src/models/qwen/hip/detail/flint_decode_step.hpp"
#include "src/models/qwen/hip/flint_gpu_model.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

struct StepParity {
  double cosine{0.0};
  double max_logp_diff{0.0};
  std::uint32_t argmax_cpu{0};
  std::uint32_t argmax_gpu{0};
};

/// G3 format: cosine on raw logits, max |delta log p| over the softmax,
/// and the greedy argmax on both sides.
StepParity CompareLogits(const std::vector<float>& cpu,
                         const std::vector<float>& gpu) {
  StepParity parity;
  double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    dot += static_cast<double>(cpu[i]) * gpu[i];
    norm_a += static_cast<double>(cpu[i]) * cpu[i];
    norm_b += static_cast<double>(gpu[i]) * gpu[i];
    if (cpu[i] > cpu[parity.argmax_cpu]) parity.argmax_cpu = static_cast<std::uint32_t>(i);
    if (gpu[i] > gpu[parity.argmax_gpu]) parity.argmax_gpu = static_cast<std::uint32_t>(i);
  }
  parity.cosine = dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-30);

  double max_cpu = -1e30, max_gpu = -1e30;
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    max_cpu = std::max(max_cpu, static_cast<double>(cpu[i]));
    max_gpu = std::max(max_gpu, static_cast<double>(gpu[i]));
  }
  double sum_cpu = 0.0, sum_gpu = 0.0;
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    sum_cpu += std::exp(static_cast<double>(cpu[i]) - max_cpu);
    sum_gpu += std::exp(static_cast<double>(gpu[i]) - max_gpu);
  }
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    const double logp_cpu =
        static_cast<double>(cpu[i]) - max_cpu - std::log(sum_cpu);
    const double logp_gpu =
        static_cast<double>(gpu[i]) - max_gpu - std::log(sum_gpu);
    parity.max_logp_diff =
        std::max(parity.max_logp_diff, std::abs(logp_cpu - logp_gpu));
  }
  return parity;
}

}  // namespace

namespace {

// Synthetic Q8_0 SwiGLU repro at the exact production MoE shapes
// (M=512 rows, K=2048): compares LaunchFusedSwiGLUGEMV (legacy) and the
// F-J1 fused MoE SwiGLU against a CPU float reference. Needs no artifact.
void SynthQ8SwiGLUCheck() {
  using gufo::core::GgmlType;
  constexpr std::size_t kM = 512;
  constexpr std::size_t kK = 2048;
  constexpr std::size_t kBlocks = kK / 32;
  constexpr std::size_t kSlabBytes = kBlocks * 34;

  std::vector<std::uint8_t> h_gate(kM * kSlabBytes), h_up(kM * kSlabBytes);
  auto fill = [&](std::vector<std::uint8_t>& buf, int salt) {
    for (std::size_t m = 0; m < kM; ++m) {
      auto* blocks = reinterpret_cast<gufo::quant::block_q8_0*>(
          buf.data() + m * kSlabBytes);
      for (std::size_t b = 0; b < kBlocks; ++b) {
        // Exact powers of two: fp16 bits are (exp+15)<<10 and the CPU
        // dequant reproduces them bit-exactly.
        const int e =
            -7 + static_cast<int>((m + salt + b / 16) % 3);
        blocks[b].d = static_cast<std::uint16_t>((e + 15) << 10);
        for (int i = 0; i < 32; ++i) {
          const int raw = static_cast<int>((m * 131 + b * 17 + i * 7 + salt) %
                                           256);
          blocks[b].qs[i] = static_cast<std::int8_t>(raw - 128);
        }
      }
    }
  };
  fill(h_gate, 0);
  fill(h_up, 5);

  std::vector<float> h_x(kK);
  for (std::size_t i = 0; i < kK; ++i) h_x[i] = 0.25F * std::sin(0.31F * i);

  // CPU reference: dequant + dot + SwiGLU.
  std::vector<float> act_ref(kM);
  {
    std::vector<float> g_row(kK), u_row(kK);
    for (std::size_t m = 0; m < kM; ++m) {
      gufo::quant::DequantizeQ8_0(h_gate.data() + m * kSlabBytes, g_row.data(),
                                  kK);
      gufo::quant::DequantizeQ8_0(h_up.data() + m * kSlabBytes, u_row.data(),
                                  kK);
      double g = 0.0, u = 0.0;
      for (std::size_t i = 0; i < kK; ++i) {
        g += static_cast<double>(g_row[i]) * h_x[i];
        u += static_cast<double>(u_row[i]) * h_x[i];
      }
      act_ref[m] = static_cast<float>((g / (1.0 + std::exp(-g))) * u);
    }
  }

  void* d_gate = nullptr;
  void* d_up = nullptr;
  float* d_x = nullptr;
  float* d_out_legacy = nullptr;
  float* d_act_moe = nullptr;
  float* d_weights_full = nullptr;
  HIP_CHECK(hipMalloc(&d_gate, h_gate.size()));
  HIP_CHECK(hipMalloc(&d_up, h_up.size()));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_x), kK * 4));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_out_legacy), kM * 4));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_act_moe),
                      (256 + 1) * kM * 4));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_weights_full), 257 * 4));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), h_gate.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_up, h_up.data(), h_up.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), kK * 4, hipMemcpyHostToDevice));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  // Path 1: legacy fused SwiGLU GEMV.
  gufo::hip::LaunchFusedSwiGLUGEMV(d_gate, GgmlType::kQ8_0, d_up,
                                   GgmlType::kQ8_0, d_x, d_out_legacy, kM, kK,
                                   stream);
  // Path 2: fused MoE SwiGLU with expert 0 weighted, everything else zero
  // (shared expert weight 0 so its block returns early).
  {
    std::vector<float> w(257, 0.0F);
    w[0] = 1.0F;
    HIP_CHECK(hipMemcpy(d_weights_full, w.data(), 257 * 4,
                        hipMemcpyHostToDevice));
  }
  gufo::hip::LaunchFlintFusedMoESwiGLU(
      d_gate, d_up, d_gate, d_up, GgmlType::kQ8_0, GgmlType::kQ8_0,
      GgmlType::kQ8_0, d_x, d_act_moe, d_weights_full, nullptr, 0, 256, kM,
      kM, kK, kSlabBytes, stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<float> act_leg(kM), act_fus(kM);
  HIP_CHECK(hipMemcpy(act_leg.data(), d_out_legacy, kM * 4,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(act_fus.data(), d_act_moe, kM * 4,
                      hipMemcpyDeviceToHost));

  double leg_max = 0.0, fus_max = 0.0, ref_scale = 0.0;
  for (std::size_t m = 0; m < kM; ++m) {
    leg_max = std::max(leg_max,
                       std::abs(static_cast<double>(act_leg[m]) - act_ref[m]));
    fus_max = std::max(fus_max,
                       std::abs(static_cast<double>(act_fus[m]) - act_ref[m]));
    ref_scale = std::max(ref_scale, std::abs(static_cast<double>(act_ref[m])));
  }
  std::cout << "synth q8_0 swiglu (M=512, K=2048): ref_scale=" << ref_scale
            << "  legacy err=" << leg_max << "  fused err=" << fus_max << "\n";
  for (int m = 0; m < 4; ++m) {
    std::cout << "  row " << m << ": ref=" << act_ref[m]
              << " legacy=" << act_leg[m] << " fused=" << act_fus[m] << "\n";
  }

  // Bisect: where does the legacy path diverge? Sweep K (block count) with
  // M=512, then M with K=2048. Also cross-check LaunchGEMV on the same gate
  // weights (the down-projection path, believed correct).
  {
    float* d_gdot = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_gdot), kM * 4));
    gufo::hip::LaunchGEMV(d_gate, GgmlType::kQ8_0, d_x, d_gdot, kM, kK,
                          stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<float> gdot(kM);
    HIP_CHECK(hipMemcpy(gdot.data(), d_gdot, kM * 4, hipMemcpyDeviceToHost));
    std::vector<float> g_row(kK);
    double gemv_err = 0.0;
    for (std::size_t m = 0; m < kM; ++m) {
      gufo::quant::DequantizeQ8_0(h_gate.data() + m * kSlabBytes, g_row.data(),
                                  kK);
      double g = 0.0;
      for (std::size_t i = 0; i < kK; ++i) {
        g += static_cast<double>(g_row[i]) * h_x[i];
      }
      gemv_err = std::max(gemv_err, std::abs(g - static_cast<double>(gdot[m])));
    }
    std::cout << "  LaunchGEMV gate-dot err=" << gemv_err << "\n";
    HIP_CHECK(hipFree(d_gdot));
  }
  for (std::size_t kk : {std::size_t{32}, std::size_t{64}, std::size_t{128},
                         std::size_t{256}, std::size_t{512},
                         std::size_t{1024}, std::size_t{2048}}) {
    const std::size_t nb = kk / 32;
    std::vector<std::uint8_t> g2(kM * nb * 34), u2(kM * nb * 34);
    // Reuse the first nb blocks of every synthetic row.
    for (std::size_t m = 0; m < kM; ++m) {
      std::memcpy(g2.data() + m * nb * 34, h_gate.data() + m * kSlabBytes,
                  nb * 34);
      std::memcpy(u2.data() + m * nb * 34, h_up.data() + m * kSlabBytes,
                  nb * 34);
    }
    void* dg2 = nullptr;
    void* du2 = nullptr;
    float* dout2 = nullptr;
    HIP_CHECK(hipMalloc(&dg2, g2.size()));
    HIP_CHECK(hipMalloc(&du2, u2.size()));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dout2), kM * 4));
    HIP_CHECK(hipMemcpy(dg2, g2.data(), g2.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(du2, u2.data(), u2.size(), hipMemcpyHostToDevice));
    gufo::hip::LaunchFusedSwiGLUGEMV(dg2, GgmlType::kQ8_0, du2,
                                     GgmlType::kQ8_0, d_x, dout2, kM, kk,
                                     stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<float> out2(kM);
    HIP_CHECK(hipMemcpy(out2.data(), dout2, kM * 4, hipMemcpyDeviceToHost));
    // CPU reference at this K.
    std::vector<float> gr(kk), ur(kk);
    double err = 0.0;
    for (std::size_t m = 0; m < kM; ++m) {
      gufo::quant::DequantizeQ8_0(g2.data() + m * nb * 34, gr.data(), kk);
      gufo::quant::DequantizeQ8_0(u2.data() + m * nb * 34, ur.data(), kk);
      double g = 0.0, u = 0.0;
      for (std::size_t i = 0; i < kk; ++i) {
        g += static_cast<double>(gr[i]) * h_x[i];
        u += static_cast<double>(ur[i]) * h_x[i];
      }
      const float ref = static_cast<float>((g / (1.0 + std::exp(-g))) * u);
      err = std::max(err, std::abs(ref - static_cast<double>(out2[m])));
    }
    std::cout << "  K=" << kk << " blocks=" << nb
              << " legacy err=" << err << "\n";
    HIP_CHECK(hipFree(dg2));
    HIP_CHECK(hipFree(du2));
    HIP_CHECK(hipFree(dout2));
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_up));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_out_legacy));
  HIP_CHECK(hipFree(d_act_moe));
  HIP_CHECK(hipFree(d_weights_full));
}

}  // namespace

namespace {

// Debug: compare the legacy readback MoE branch against the F-J1 fused
// branch on every real MoE layer with the same O(1) input. On mismatch,
// dumps the full tensor-type inventory, device-vs-host slab bytes, and a
// CPU-dequant ground truth for the act of the first selected expert.
void ProbeMoEBranch(const gufo::models::FlintModelWeights& gpu_weights,
                    const gufo::models::FlintModelWeights& cpu_weights,
                    gufo::hip::FlintGpuArena& arena) {
  using gufo::core::GgmlType;
  using gufo::models::QwenTensorRef;
  const auto& config = gpu_weights.config;
  const std::size_t hidden = config.hidden_size;
  const std::size_t expert_ff = config.expert_feed_forward_length;
  const std::size_t shared_ff = config.expert_shared_feed_forward_length;
  const std::size_t expert_slab = expert_ff * hidden;

  // Device scratch (the kernels read x directly; hipMalloc keeps the
  // hipMemcpy kinds unambiguous).
  const std::size_t scratch_floats =
      hidden * 4 +  // x, expert_out, mlp_legacy, mlp_fused
      config.expert_count +  // router logits
      8 +  // selected weights (top-k readback)
      config.expert_count + 1 +  // weights_full
      expert_ff +  // act
      config.expert_count * expert_ff + shared_ff;  // act_moe
  float* block_start = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&block_start),
                      scratch_floats * sizeof(float)));
  float* d = block_start;
  auto carve = [&](std::size_t n) {
    float* out = d;
    d += n;
    return out;
  };
  float* x = carve(hidden);
  float* router_logits = carve(config.expert_count);
  float* selected = carve(8);
  float* weights_full = carve(config.expert_count + 1);
  float* act = carve(expert_ff);
  float* act_moe = carve(config.expert_count * expert_ff + shared_ff);
  float* expert_out = carve(hidden);
  float* mlp_legacy = carve(hidden);
  float* mlp_fused = carve(hidden);
  // Compacted job list for the fused kernels (router fills it per layer).
  int* job_list = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&job_list),
                      gufo::hip::kFlintMoeMaxJobs * sizeof(int)));
  std::uint32_t* d_selected = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_selected), 8 * 4));

  // O(1) input, post-RMSNorm-like magnitudes.
  std::vector<float> h_x(hidden);
  for (std::size_t i = 0; i < hidden; ++i) h_x[i] = 0.25F * std::sin(0.31F * i);
  HIP_CHECK(hipMemcpy(x, h_x.data(), hidden * 4, hipMemcpyHostToDevice));

  const auto stream = arena.GetBuffers().stream;

  for (std::uint32_t li = 0; li < config.num_layers; ++li) {
    if (gpu_weights.layers[li].attrs.dense_ffn) {
      continue;
    }
    const auto& layer = gpu_weights.layers[li];
    const auto& cpu_layer = cpu_weights.layers[li];
    const auto type = layer.ffn_gate_exps.type;
    const std::size_t slab_bytes =
        gufo::quant::EncodedSizeBytes(type, expert_slab);

  std::vector<std::uint32_t> h_sel(8);
  std::vector<float> h_w(8);
  auto run_legacy = [&](float* mlp) {
    gufo::hip::LaunchGEMV(layer.ffn_gate_inp.data, layer.ffn_gate_inp.type, x,
                          router_logits, config.expert_count, hidden, stream);
    gufo::hip::LaunchFlintRouterTopK(router_logits, config.expert_count,
                                     config.expert_used_count,
                                     config.expert_weights_norm,
                                     config.expert_weights_scale, d_selected,
                                     selected, stream);
    HIP_CHECK(hipMemcpy(h_sel.data(), d_selected, 8 * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_w.data(), selected, 8 * 4, hipMemcpyDeviceToHost));
    gufo::hip::LaunchFlintZero(mlp, hidden, stream);
    for (std::uint32_t i = 0; i < config.expert_used_count; ++i) {
      const std::size_t e = h_sel[i];
      auto slab = [&](const QwenTensorRef& t) {
        auto s = t;
        s.data = static_cast<const std::uint8_t*>(t.data) +
                 gufo::quant::EncodedSizeBytes(t.type, e * expert_slab);
        s.num_elements = expert_slab;
        return s;
      };
      gufo::hip::LaunchFusedSwiGLUGEMV(slab(layer.ffn_gate_exps).data,
                                       layer.ffn_gate_exps.type,
                                       slab(layer.ffn_up_exps).data,
                                       layer.ffn_up_exps.type, x, act,
                                       expert_ff, hidden, stream);
      gufo::hip::LaunchGEMV(
          static_cast<const std::uint8_t*>(layer.ffn_down_exps.data) +
              gufo::quant::EncodedSizeBytes(layer.ffn_down_exps.type,
                                            e * expert_slab),
          layer.ffn_down_exps.type, act, expert_out, hidden, expert_ff,
          stream);
      gufo::hip::LaunchFlintExpertAccumulate(mlp, expert_out, h_w[i], hidden,
                                             stream);
    }
    gufo::hip::LaunchFusedSwiGLUGEMV(layer.ffn_gate_shexp.data,
                                     layer.ffn_gate_shexp.type,
                                     layer.ffn_up_shexp.data,
                                     layer.ffn_up_shexp.type, x, act, shared_ff,
                                     hidden, stream);
    gufo::hip::LaunchGEMV(layer.ffn_down_shexp.data, layer.ffn_down_shexp.type,
                          act, expert_out, hidden, shared_ff, stream);
    gufo::hip::LaunchFlintExpertAccumulate(mlp, expert_out, 1.0F, hidden,
                                           stream);
    HIP_CHECK(hipStreamSynchronize(stream));
  };

  auto run_fused = [&](float* mlp) {
    gufo::hip::LaunchGEMV(layer.ffn_gate_inp.data, layer.ffn_gate_inp.type, x,
                          router_logits, config.expert_count, hidden, stream);
    gufo::hip::LaunchFlintRouterTopKFull(router_logits, config.expert_count,
                                         config.expert_used_count,
                                         config.expert_weights_norm,
                                         config.expert_weights_scale,
                                         weights_full, 1.0F, mlp, hidden,
                                         job_list, stream);
    const std::size_t down_slab_bytes = gufo::quant::EncodedSizeBytes(
        layer.ffn_down_exps.type, expert_slab);
    gufo::hip::LaunchFlintFusedMoESwiGLU(
        layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
        layer.ffn_gate_shexp.data, layer.ffn_up_shexp.data,
        layer.ffn_gate_exps.type, layer.ffn_gate_shexp.type,
        layer.ffn_up_shexp.type, x, act_moe, weights_full, job_list,
        config.expert_used_count + 1, config.expert_count, expert_ff,
        shared_ff, hidden, slab_bytes, stream);
    gufo::hip::LaunchFlintFusedMoEDown(layer.ffn_down_exps.data,
                                       layer.ffn_down_shexp.data,
                                       layer.ffn_down_exps.type,
                                       layer.ffn_down_shexp.type, act_moe, mlp,
                                       weights_full, job_list,
                                       config.expert_used_count + 1,
                                       config.expert_count, expert_ff,
                                       shared_ff, hidden, down_slab_bytes,
                                       stream);
    HIP_CHECK(hipStreamSynchronize(stream));
  };

    run_legacy(mlp_legacy);
    run_fused(mlp_fused);

    // Correctness first (the isolated stage timings below accumulate into
    // the mlp buffers without re-zeroing, so they must not precede this).
    std::vector<float> l(hidden), f(hidden);
    HIP_CHECK(hipMemcpy(l.data(), mlp_legacy, hidden * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(f.data(), mlp_fused, hidden * 4, hipMemcpyDeviceToHost));
    double max_abs = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < hidden; ++i) {
      max_abs = std::max(max_abs, std::abs(static_cast<double>(l[i]) - f[i]));
      scale = std::max(scale, std::abs(static_cast<double>(l[i])));
    }
    const bool ok = max_abs < 1e-4 * std::max(1.0, scale);

    // Perf comparison (debug tool): both paths include their own stream
    // sync, so wall-clock over 10 iterations is directly comparable.
    auto timed = [&](auto&& fn) {
      fn();  // warm-up (allocator, instruction cache)
      fn();
      constexpr int kIters = 10;
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < kIters; ++i) {
        fn();
      }
      const auto t1 = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::micro>(t1 - t0).count() /
             kIters;
    };
    const double legacy_us = timed([&] { run_legacy(mlp_legacy); });
    const double fused_us = timed([&] { run_fused(mlp_fused); });
    const double f_router_us = timed([&] {
      gufo::hip::LaunchGEMV(layer.ffn_gate_inp.data, layer.ffn_gate_inp.type,
                            x, router_logits, config.expert_count, hidden,
                            stream);
      gufo::hip::LaunchFlintRouterTopKFull(
          router_logits, config.expert_count, config.expert_used_count,
          config.expert_weights_norm, config.expert_weights_scale,
          weights_full, 1.0F, mlp_fused, hidden, job_list, stream);
      HIP_CHECK(hipStreamSynchronize(stream));
    });
    const double f_swiglu_us = timed([&] {
      gufo::hip::LaunchFlintFusedMoESwiGLU(
          layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
          layer.ffn_gate_shexp.data, layer.ffn_up_shexp.data,
          layer.ffn_gate_exps.type, layer.ffn_gate_shexp.type,
          layer.ffn_up_shexp.type, x, act_moe, weights_full, job_list,
          config.expert_used_count + 1, config.expert_count, expert_ff,
          shared_ff, hidden, slab_bytes, stream);
      HIP_CHECK(hipStreamSynchronize(stream));
    });
    const double f_down_us = timed([&] {
      gufo::hip::LaunchFlintFusedMoEDown(
          layer.ffn_down_exps.data, layer.ffn_down_shexp.data,
          layer.ffn_down_exps.type, layer.ffn_down_shexp.type, act_moe,
          mlp_fused, weights_full, job_list, config.expert_used_count + 1,
          config.expert_count, expert_ff, shared_ff, hidden,
          gufo::quant::EncodedSizeBytes(layer.ffn_down_exps.type,
                                        expert_slab),
          stream);
      HIP_CHECK(hipStreamSynchronize(stream));
    });
    auto legacy_slab = [&](const QwenTensorRef& t, std::size_t e) {
      auto s = t;
      s.data = static_cast<const std::uint8_t*>(t.data) +
               gufo::quant::EncodedSizeBytes(t.type, e * expert_slab);
      s.num_elements = expert_slab;
      return s;
    };
    const double l_eswiglu_us = timed([&] {
      for (std::uint32_t i = 0; i < config.expert_used_count; ++i) {
        const std::size_t e = h_sel[i];
        gufo::hip::LaunchFusedSwiGLUGEMV(
            legacy_slab(layer.ffn_gate_exps, e).data,
            layer.ffn_gate_exps.type, legacy_slab(layer.ffn_up_exps, e).data,
            layer.ffn_up_exps.type, x, act, expert_ff, hidden, stream);
      }
      HIP_CHECK(hipStreamSynchronize(stream));
    });
    const double l_edown_us = timed([&] {
      for (std::uint32_t i = 0; i < config.expert_used_count; ++i) {
        const std::size_t e = h_sel[i];
        gufo::hip::LaunchGEMV(
            legacy_slab(layer.ffn_down_exps, e).data,
            layer.ffn_down_exps.type, act, expert_out, hidden, expert_ff,
            stream);
        gufo::hip::LaunchFlintExpertAccumulate(mlp_legacy, expert_out, h_w[i],
                                               hidden, stream);
      }
      HIP_CHECK(hipStreamSynchronize(stream));
    });

    std::cout << "  layer " << li << " type=" << static_cast<int>(type)
              << " down=" << static_cast<int>(layer.ffn_down_exps.type)
              << " shexp=" << static_cast<int>(layer.ffn_gate_shexp.type)
              << " slab=" << slab_bytes
              << " down_slab="
              << gufo::quant::EncodedSizeBytes(layer.ffn_down_exps.type,
                                               expert_slab)
              << " mlp max_abs=" << max_abs << " scale=" << scale
              << " legacy=" << legacy_us << "us fused=" << fused_us << "us"
              << "\n      f: router=" << f_router_us
              << " swiglu=" << f_swiglu_us << " down=" << f_down_us
              << " | l: eswiglu=" << l_eswiglu_us
              << " edown+acc=" << l_edown_us
              << (ok ? "  OK" : "  MISMATCH") << "\n";
    if (!ok) {
      // Per-expert act check on the first selected expert to isolate the
      // phase (SwiGLU vs down).
      std::vector<std::uint32_t> h_sel(8);
      HIP_CHECK(hipMemcpy(h_sel.data(), d_selected, 8 * 4,
                          hipMemcpyDeviceToHost));
      const std::size_t e0 = h_sel[0];
      std::cout << "    selected:";
      for (int i = 0; i < 6; ++i) {
        std::cout << " " << h_sel[i];
      }
      std::cout << "\n";

      // (a) Device-vs-host byte check on expert e0's gate and down slabs:
      // catches upload/remap offset errors independent of kernel math.
      auto slab = [&](const QwenTensorRef& t) {
        auto s = t;
        s.data = static_cast<const std::uint8_t*>(t.data) +
                 gufo::quant::EncodedSizeBytes(t.type, e0 * expert_slab);
        s.num_elements = expert_slab;
        return s;
      };
      const QwenTensorRef* gpu_tensors[] = {&layer.ffn_gate_exps,
                                            &layer.ffn_up_exps,
                                            &layer.ffn_down_exps};
      const QwenTensorRef* cpu_tensors[] = {&cpu_layer.ffn_gate_exps,
                                            &cpu_layer.ffn_up_exps,
                                            &cpu_layer.ffn_down_exps};
      const char* tensor_names[] = {"gate", "up", "down"};
      for (int ti = 0; ti < 3; ++ti) {
        const std::size_t probe_bytes = 64;
        std::vector<std::uint8_t> dev(probe_bytes), host(probe_bytes);
        HIP_CHECK(hipMemcpy(dev.data(), slab(*gpu_tensors[ti]).data,
                            probe_bytes, hipMemcpyDeviceToHost));
        std::memcpy(host.data(),
                    static_cast<const std::uint8_t*>(cpu_tensors[ti]->data) +
                        gufo::quant::EncodedSizeBytes(cpu_tensors[ti]->type,
                                                      e0 * expert_slab),
                    probe_bytes);
        std::cout << "    bytes " << tensor_names[ti]
                  << " e0: dev==host " << (dev == host ? "YES" : "NO")
                  << "\n";
      }

      // (b) CPU ground truth for the act of expert e0, rows 0..3.
      gufo::hip::LaunchFusedSwiGLUGEMV(slab(layer.ffn_gate_exps).data,
                                       layer.ffn_gate_exps.type,
                                       slab(layer.ffn_up_exps).data,
                                       layer.ffn_up_exps.type, x, act,
                                       expert_ff, hidden, stream);
      std::vector<float> act_leg(expert_ff), act_fus(expert_ff);
      HIP_CHECK(hipMemcpy(act_leg.data(), act, expert_ff * 4,
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(act_fus.data(), act_moe + e0 * expert_ff,
                          expert_ff * 4, hipMemcpyDeviceToHost));
      double a_max = 0.0, a_scale = 0.0;
      for (std::size_t i = 0; i < expert_ff; ++i) {
        a_max = std::max(a_max, std::abs(static_cast<double>(act_leg[i]) -
                                         act_fus[i]));
        a_scale = std::max(a_scale, std::abs(static_cast<double>(act_leg[i])));
      }
      std::cout << "    act expert " << e0 << ": max_abs=" << a_max
                << " scale=" << a_scale << "\n";

      const std::size_t gate_row_bytes = gufo::quant::QuantizedRowBytes(
          layer.ffn_gate_exps.type, hidden);
      const std::size_t up_row_bytes = gufo::quant::QuantizedRowBytes(
          layer.ffn_up_exps.type, hidden);
      const auto* cpu_gate =
          static_cast<const std::uint8_t*>(cpu_layer.ffn_gate_exps.data) +
          gufo::quant::EncodedSizeBytes(layer.ffn_gate_exps.type,
                                        e0 * expert_slab);
      const auto* cpu_up =
          static_cast<const std::uint8_t*>(cpu_layer.ffn_up_exps.data) +
          gufo::quant::EncodedSizeBytes(layer.ffn_up_exps.type,
                                        e0 * expert_slab);
      std::vector<float> g_row(hidden), u_row(hidden);
      for (int r = 0; r < 4; ++r) {
        if (layer.ffn_gate_exps.type == GgmlType::kQ8_0) {
          gufo::quant::DequantizeQ8_0(cpu_gate + r * gate_row_bytes,
                                      g_row.data(), hidden);
        } else if (layer.ffn_gate_exps.type == GgmlType::kQ6_K) {
          gufo::quant::DequantizeQ6_K(cpu_gate + r * gate_row_bytes,
                                      g_row.data(), hidden);
        }
        if (layer.ffn_up_exps.type == GgmlType::kQ8_0) {
          gufo::quant::DequantizeQ8_0(cpu_up + r * up_row_bytes,
                                      u_row.data(), hidden);
        } else if (layer.ffn_up_exps.type == GgmlType::kQ6_K) {
          gufo::quant::DequantizeQ6_K(cpu_up + r * up_row_bytes,
                                      u_row.data(), hidden);
        }
        double g = 0.0, u = 0.0;
        for (std::size_t i = 0; i < hidden; ++i) {
          g += static_cast<double>(g_row[i]) * h_x[i];
          u += static_cast<double>(u_row[i]) * h_x[i];
        }
        const float act_cpu =
            static_cast<float>((g / (1.0 + std::exp(-g))) * u);
        std::cout << "    row " << r << ": cpu=" << act_cpu
                  << " legacy=" << act_leg[r] << " fused=" << act_fus[r]
                  << "\n";
      }

      for (std::size_t i = 0; i < 8; ++i) {
        std::cout << "    [" << i << "] legacy=" << l[i] << " fused=" << f[i]
                  << "\n";
      }
    }
  }

  HIP_CHECK(hipFree(d_selected));
  HIP_CHECK(hipFree(job_list));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(block_start)));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string_view(argv[1]) == "synthq8") {
    SynthQ8SwiGLUCheck();
    return 0;
  }
  const char* env = std::getenv("GUFO_FLINT_GGUF");
  if (env == nullptr || *env == '\0') {
    std::cout << "SKIPPED: GUFO_FLINT_GGUF not set\n";
    return 77;
  }

  std::uint32_t max_context = 4096;
  std::uint32_t steps = 8;
  bool probe = argc > 1 && std::string_view(argv[1]) == "probe";
  if (argc > 1 && !probe) max_context = static_cast<std::uint32_t>(std::atoi(argv[1]));
  if (argc > 2) steps = static_cast<std::uint32_t>(std::atoi(argv[2]));
  Expect(steps > 0, "steps must be positive");

  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(env, &error);
  Expect(reader != nullptr, "open GGUF: " + error);

  // CPU reference weights: host pointers straight from the reader mapping.
  // The reader must outlive both weight sets.
  auto cpu_weights = gufo::models::FlintModelWeights::LoadFromGguf(*reader, &error);
  Expect(cpu_weights.has_value(), "load CPU weights: " + error);
  const auto& config = cpu_weights->config;
  Expect(config.num_layers > 0, "model must have layers");

  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  Expect(tokenizer != nullptr, "create tokenizer: " + error);
  const std::string prompt = "The quick brown fox jumps over the lazy dog.";
  std::vector<gufo::tokenization::TokenId> prompt_tokens =
      tokenizer->Encode(prompt);
  Expect(!prompt_tokens.empty(), "prompt must tokenize to at least one token");

  // GPU model: registered mapped copies, tensors re-pointed at device space.
  auto gpu_model = gufo::hip::FlintGpuModel::CreateFromGguf(
      std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)), &error);
  Expect(gpu_model != nullptr, "create GPU model: " + error);

  if (probe) {
    gufo::hip::FlintGpuArena arena(gpu_model->GetConfig(), 64);
    ProbeMoEBranch(gpu_model->GetWeights(), *cpu_weights, arena);
    return 0;
  }

  std::cout << "flint real smoke: " << env << "\n"
            << "  layers=" << config.num_layers
            << " hidden=" << config.hidden_size
            << " heads=" << config.num_attention_heads << "/"
            << config.num_key_value_heads << " head_dim=" << config.head_dim
            << " vocab=" << config.vocab_size
            << " experts=" << config.expert_count << " top"
            << config.expert_used_count
            << " window=" << config.sliding_window << "\n"
            << "  weight regions=" << gpu_model->GetWeightRegionCount()
            << " resident=" << gpu_model->GetResidentBytes() / (1ULL << 20)
            << " MiB\n"
            << "  prompt(" << prompt_tokens.size() << " tokens): " << prompt
            << "\n  max_context=" << max_context << " steps=" << steps << "\n";

  gufo::models::QwenKvCache kv(config.num_layers,
                               config.num_key_value_heads, max_context,
                               config.head_dim);
  gufo::models::FlintScratchArena arena(config);
  std::vector<float> cpu_logits(config.vocab_size);

  gufo::hip::FlintGpuArena gpu_arena(config, max_context);
  const auto buffers = gpu_arena.GetBuffers();
  std::vector<float> gpu_logits(config.vocab_size);
  std::cout << "  arena scratch=" << gpu_arena.GetScratchBytes() / (1ULL << 20)
            << " MiB\n";

  for (int pass = 0; pass < 2; ++pass) {
    // The KV cache is only read at [0, pos] and every position is rewritten
    // before it is read, so the second pass re-runs cleanly without a reset.
    std::uint32_t token = prompt_tokens[0];
    double total_ms = 0.0;
    std::cout << (pass == 0 ? "  pass 1 (warm):" : "  pass 2 (timed):");
    for (std::uint32_t step = 0; step < steps; ++step) {
      gufo::models::FlintForwardModel(token, step, *cpu_weights, kv, arena,
                                      cpu_logits);

      const auto t0 = std::chrono::steady_clock::now();
      gufo::hip::ExecuteFlintDecodeStep(gpu_model->GetWeights(), token, step,
                                        /*compute_logits=*/true, buffers);
      HIP_CHECK(hipStreamSynchronize(buffers.stream));
      const auto t1 = std::chrono::steady_clock::now();
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

      HIP_CHECK(hipMemcpy(gpu_logits.data(), buffers.logits,
                          gpu_logits.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      const auto parity = CompareLogits(cpu_logits, gpu_logits);
      Expect(parity.cosine >= 0.999,
             "cosine at step " + std::to_string(step) + " = " +
                 std::to_string(parity.cosine));
      Expect(parity.max_logp_diff <= 5e-3,
             "max |delta log p| at step " + std::to_string(step) + " = " +
                 std::to_string(parity.max_logp_diff));
      Expect(parity.argmax_cpu == parity.argmax_gpu,
             "argmax mismatch at step " + std::to_string(step) + ": cpu " +
                 std::to_string(parity.argmax_cpu) + " gpu " +
                 std::to_string(parity.argmax_gpu));
      std::cout << " s" << step << " tok=" << parity.argmax_gpu
                << " cos=" << parity.cosine
                << " dlogp=" << parity.max_logp_diff
                << " (" << total_ms / (step + 1) << " ms/step avg)";
      token = parity.argmax_cpu;
    }
    std::cout << "\n";
    if (pass == 1) {
      const double tps = steps * 1000.0 / total_ms;
      std::cout << "  decode: " << total_ms / steps << " ms/step = " << tps
                << " t/s\n";
    }
  }

  std::cout << "flint_real_smoke_test ok (" << steps
            << " steps x 2 passes, G3 format, real artifact)\n";
  return 0;
}
