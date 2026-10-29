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
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/flint_forward.hpp"
#include "src/models/qwen/hip/detail/flint_decode_step.hpp"
#include "src/models/qwen/hip/flint_gpu_model.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
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

// Online-tile decode attention check (2.6 prerequisite): the kernel that
// takes over past the LDS score-row budget (position 16129+ device-lost
// before) vs a CPU double-precision reference, at the flint geometry
// (32/4/128 + sinks, no window — the global-layer case). Also runs the
// baseline kernel at a position where its LDS row still fits, validating
// the reference itself. Needs no artifact.
void RunOnlineDecodeCheck() {
  using gufo::hip::detail::FlintKernelPolicy;
  constexpr std::uint32_t kHeads = 32;
  constexpr std::uint32_t kKvHeads = 4;
  constexpr std::uint32_t kHeadDim = 128;
  constexpr std::uint32_t kMaxContext = 32768;
  constexpr std::uint32_t kLayer = 0;
  constexpr std::size_t kKvWidth = kKvHeads * kHeadDim;
  constexpr std::size_t kCacheElems =
      static_cast<std::size_t>(kKvHeads) * kMaxContext * kHeadDim;
  constexpr float kScale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  std::vector<float> h_q(kHeads * kHeadDim);
  std::vector<float> h_sink(kHeads);
  std::vector<float> h_k(kCacheElems);
  std::vector<float> h_v(kCacheElems);
  std::uint32_t lcg = 0x9e3779b9;
  const auto next = [&lcg] {
    lcg = lcg * 1664525u + 1013904223u;
    return static_cast<float>(static_cast<double>(lcg >> 8) / 16777216.0 -
                              0.5);
  };
  for (auto& v : h_q) {
    v = next();
  }
  for (auto& s : h_sink) {
    s = next() * 2.0F;
  }
  for (auto& v : h_k) {
    v = next();
  }
  for (auto& v : h_v) {
    v = next();
  }

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_sink = nullptr,
        *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_q, h_q.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_sink, h_sink.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, h_k.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, h_v.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, kHeads * kHeadDim * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_sink, h_sink.data(), h_sink.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), h_k.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), h_v.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  const FlintKernelPolicy policy{d_sink, 0U};

  const auto cpu_reference = [&](std::uint32_t pos) {
    std::vector<double> out(kHeads * kHeadDim);
    const std::size_t seq_len = pos + 1;
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const std::uint32_t kv_h = h / (kHeads / kKvHeads);
      const float* q_h = h_q.data() + h * kHeadDim;
      double m = h_sink[h];
      std::vector<double> scores(seq_len);
      for (std::size_t p = 0; p < seq_len; ++p) {
        const float* k_p = h_k.data() +
                           ((kLayer * kKvHeads + kv_h) *
                                static_cast<std::size_t>(kMaxContext) +
                            p) *
                               kHeadDim;
        double dot = 0.0;
        for (std::size_t d = 0; d < kHeadDim; ++d) {
          dot += static_cast<double>(q_h[d]) * static_cast<double>(k_p[d]);
        }
        scores[p] = dot * kScale;
        m = std::max(m, scores[p]);
      }
      double l = std::exp(static_cast<double>(h_sink[h]) - m);
      for (std::size_t p = 0; p < seq_len; ++p) {
        scores[p] = std::exp(scores[p] - m);
        l += scores[p];
      }
      for (std::size_t d = 0; d < kHeadDim; ++d) {
        double acc = 0.0;
        for (std::size_t p = 0; p < seq_len; ++p) {
          const float* v_p = h_v.data() +
                             ((kLayer * kKvHeads + kv_h) *
                                  static_cast<std::size_t>(kMaxContext) +
                              p) *
                                 kHeadDim;
          acc += scores[p] * static_cast<double>(v_p[d]);
        }
        out[h * kHeadDim + d] = acc / l;
      }
    }
    return out;
  };

  const auto run_kernel = [&](std::uint32_t pos) {
    HIP_CHECK(hipMemsetAsync(d_out, 0, kHeads * kHeadDim * sizeof(float),
                             stream));
    gufo::hip::LaunchCausalDecodeAttention(
        d_q, /*gate=*/nullptr, d_k, d_v, /*k_cache_f16=*/nullptr,
        /*v_cache_f16=*/nullptr, d_out, kLayer, pos, 1, kMaxContext, kHeads,
        kKvHeads, kHeadDim, stream, std::span<float>{}, &policy);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<float> out(kHeads * kHeadDim);
    HIP_CHECK(hipMemcpy(out.data(), d_out, out.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    return out;
  };

  int failures = 0;
  for (const std::uint32_t pos : {8000u, 20000u}) {
    const std::string route = pos >= 16129 ? "online" : "baseline";
    const auto ref = cpu_reference(pos);
    const auto got = run_kernel(pos);
    double max_abs = 0.0;
    double dot_q = 0.0, nref = 0.0, ngot = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      max_abs = std::max(max_abs, std::abs(ref[i] - got[i]));
      dot_q += ref[i] * got[i];
      nref += ref[i] * ref[i];
      ngot += got[i] * got[i];
    }
    const double cos =
        dot_q / (std::sqrt(nref) * std::sqrt(ngot) + 1e-30);
    std::cout << "  onlinedecode pos=" << pos << " (" << route
              << "): cos=" << cos << " max_abs=" << max_abs << "\n";
    if (!(cos > 0.9999) || max_abs > 5e-4) {
      std::cerr << "FAIL: onlinedecode parity at pos " << pos << "\n";
      ++failures;
    }
  }

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_sink));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipStreamDestroy(stream));
  if (failures != 0) {
    std::exit(1);
  }
  std::cout << "  onlinedecode: PASS\n";
}

// Flash prefill attention check (2.6): FlintWmmaCausalAttention vs a CPU
// double-precision reference at the flint geometry, batch 16 and 512,
// with and without window, sinks always on. Needs no artifact.
/// Times the decode-attention dispatch shapes at a long span (flashbench
/// tuning probe): the graph ladder vs the legacy baseline vs the online
/// tile kernel, 49 layers' worth of launches per sample, synthetic caches.
/// CAVEAT learned the hard way: repeated relaunches of the same region are
/// L2-resident, which flatters uncoalesced shapes (the online tile read
/// 2x "faster" than coalesced rivals here while losing 25% in the serve's
/// cold-KV F-J4). Rank kernels with this probe, but ship on serve A/Bs.
void RunFlashBench() {
  using gufo::hip::detail::FlintKernelPolicy;
  constexpr std::uint32_t kHeads = 32;
  constexpr std::uint32_t kKvHeads = 4;
  constexpr std::uint32_t kHeadDim = 128;
  constexpr std::uint32_t kMaxContext = 8192;
  constexpr std::uint32_t kPos = 6600;
  constexpr std::uint32_t kLayers = 49;

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  float* d_k = nullptr;
  float* d_v = nullptr;
  float* d_q = nullptr;
  float* d_out = nullptr;
  float* d_sink = nullptr;
  std::uint32_t* d_control = nullptr;
  // Sized for the online-tier probe's 24576 stride; the 8192-stride
  // benches stay in the first planes.
  constexpr std::uint32_t kCacheStride = 24576;
  const std::size_t cache_elems =
      static_cast<std::size_t>(kKvHeads) * kCacheStride * kHeadDim;
  HIP_CHECK(hipMalloc(&d_k, cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_k, 0x3C, cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_v, 0x3C, cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q, kHeads * kHeadDim * sizeof(float)));
  HIP_CHECK(hipMemset(d_q, 0x3C, kHeads * kHeadDim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, kHeads * kHeadDim * sizeof(float)));
  std::vector<float> h_sink(kHeads, 0.5F);
  HIP_CHECK(hipMalloc(&d_sink, kHeads * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_sink, h_sink.data(), kHeads * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_control, 2 * sizeof(std::uint32_t)));
  const std::uint32_t control[2] = {42U, kPos};
  HIP_CHECK(hipMemcpy(d_control, control, sizeof(control),
                      hipMemcpyHostToDevice));

  const FlintKernelPolicy policy{d_sink, 0U};
  const FlintKernelPolicy policy_w{d_sink, 513U};
  const auto bench = [&](const char* name, auto&& launch) {
    // warm
    for (int i = 0; i < 3; ++i) {
      launch();
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto t0 = std::chrono::steady_clock::now();
    constexpr int kReps = 20;
    for (int i = 0; i < kReps; ++i) {
      for (std::uint32_t l = 0; l < kLayers; ++l) {
        launch();
      }
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() /
        (kReps * kLayers);
    std::cout << "  " << name << ": " << ms << " ms/layer @ pos " << kPos
              << "\n";
  };

  bench("graph ladder (d_pos, fixed-LDS clone @8192)",
        [&] {
          gufo::hip::LaunchAttention(
              d_q, nullptr, nullptr, nullptr, d_k, d_v, nullptr, nullptr,
              d_out, 0U, d_control + 1, kMaxContext, kHeads, kKvHeads,
              kHeadDim, stream, /*skip_kv_write=*/true, &policy);
        });
  bench("baseline (host pos, dynamic LDS)",
        [&] {
          gufo::hip::LaunchAttention(
              d_q, nullptr, nullptr, nullptr, d_k, d_v, nullptr, nullptr,
              d_out, 0U, kPos, kMaxContext, kHeads, kKvHeads, kHeadDim,
              stream, /*split_k_scratch=*/nullptr, /*skip_kv_write=*/true,
              &policy);
        });
  bench("online Ptr (d_pos, >8192 tier)",
        [&] {
          gufo::hip::LaunchAttention(
              d_q, nullptr, nullptr, nullptr, d_k, d_v, nullptr, nullptr,
              d_out, 0U, d_control + 1, 24576U, kHeads, kKvHeads, kHeadDim,
              stream, /*skip_kv_write=*/true, &policy);
        });
  bench("windowed baseline (host pos, span 513)",
        [&] {
          gufo::hip::LaunchAttention(
              d_q, nullptr, nullptr, nullptr, d_k, d_v, nullptr, nullptr,
              d_out, 0U, kPos, kMaxContext, kHeads, kKvHeads, kHeadDim,
              stream, /*split_k_scratch=*/nullptr, /*skip_kv_write=*/true,
              &policy_w);
        });

  // Cold-cache span sweep of the global-layer decode kernel. Before every
  // timed launch, a 256 MB memset on the same stream evicts L2/MALL so the
  // KV read comes from HBM. Timing uses hipEvent pairs recorded immediately
  // around the attention launch only; the flush is stream-ordered before the
  // start event, so it is outside the timed interval. Each span gets its own
  // cache with max_context == span and pos == span - 1, so exactly `span`
  // keys are read per head. Kernel selection mirrors LaunchAttention's
  // policy-window-0 branches (<=513 graph, <=4096 graph, else online).
  constexpr std::size_t kFlushBytes = 256ULL * 1024 * 1024;
  constexpr int kColdWarm = 3;
  constexpr int kColdReps = 20;
  float* d_flush = nullptr;
  HIP_CHECK(hipMalloc(&d_flush, kFlushBytes));
  constexpr std::uint32_t kColdMaxSpan = 65536;
  float* d_scratch = nullptr;
  HIP_CHECK(hipMalloc(
      &d_scratch,
      gufo::hip::FlintDecodeSplitKvScratchElements(kColdMaxSpan, kHeads,
                                                   kHeadDim) *
          sizeof(float)));
  std::vector<hipEvent_t> events(2 * kColdReps);
  for (auto& event : events) {
    HIP_CHECK(hipEventCreate(&event));
  }
  std::cout << "  cold-cache span sweep (256 MB flush before each launch, "
            << kColdReps << " reps, kernel timed alone)\n";
  for (const std::uint32_t span : {512U, 4096U, 8192U, 32768U, 65536U}) {
    const std::size_t cold_elems =
        static_cast<std::size_t>(kKvHeads) * span * kHeadDim;
    float* d_cold_k = nullptr;
    float* d_cold_v = nullptr;
    std::uint32_t* d_cold_pos = nullptr;
    HIP_CHECK(hipMalloc(&d_cold_k, cold_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_cold_v, cold_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_cold_pos, sizeof(std::uint32_t)));
    HIP_CHECK(hipMemset(d_cold_k, 0x3C, cold_elems * sizeof(float)));
    HIP_CHECK(hipMemset(d_cold_v, 0x3C, cold_elems * sizeof(float)));
    const std::uint32_t cold_pos = span - 1;
    HIP_CHECK(hipMemcpy(d_cold_pos, &cold_pos, sizeof(cold_pos),
                        hipMemcpyHostToDevice));
    const char* kernel = span <= 513    ? "graph<513>"
                         : span <= 4096 ? "graph<4096>"
                                        : "online";
    const auto flush = [&] {
      HIP_CHECK(hipMemsetAsync(d_flush, 0xA5, kFlushBytes, stream));
    };
    const auto launch_online = [&] {
      gufo::hip::LaunchAttention(
          d_q, nullptr, nullptr, nullptr, d_cold_k, d_cold_v, nullptr, nullptr,
          d_out, 0U, d_cold_pos, span, kHeads, kKvHeads, kHeadDim, stream,
          /*skip_kv_write=*/true, &policy);
    };
    const auto launch_splitkv = [&] {
      const bool ok = gufo::hip::LaunchFlintDecodeSplitKvAttention(
          d_q, d_cold_k, d_cold_v, nullptr, nullptr, d_out, 0U, 0U, d_cold_pos,
          span, kHeads, kKvHeads, kHeadDim, d_scratch, stream, &policy);
      if (!ok) {
        std::cerr << "FAIL: splitkv launch at span " << span << "\n";
        std::exit(1);
      }
    };
    // Flush-then-time loop shared by every kernel so the method is identical.
    const auto time_cold = [&](const auto& launch) {
      for (int i = 0; i < kColdWarm; ++i) {
        flush();
        launch();
      }
      for (int i = 0; i < kColdReps; ++i) {
        flush();
        HIP_CHECK(hipEventRecord(events[2 * i], stream));
        launch();
        HIP_CHECK(hipEventRecord(events[2 * i + 1], stream));
      }
      HIP_CHECK(hipStreamSynchronize(stream));
      double total_ms = 0.0;
      for (int i = 0; i < kColdReps; ++i) {
        float ms = 0.0F;
        HIP_CHECK(hipEventElapsedTime(&ms, events[2 * i], events[2 * i + 1]));
        total_ms += ms;
      }
      return total_ms / kColdReps;
    };
    const auto report = [&](const char* name, double ms_per_layer) {
      // unique: K and V read once per KV head. actual: each of the 32 query
      // heads re-reads its KV head (GQA 8:1), as the per-head kernel does.
      const double unique_bytes =
          2.0 * kKvHeads * static_cast<double>(span) * kHeadDim * sizeof(float);
      const double actual_bytes =
          2.0 * kHeads * static_cast<double>(span) * kHeadDim * sizeof(float);
      const double unique_gbps = unique_bytes / (ms_per_layer * 1e-3) / 1e9;
      const double actual_gbps = actual_bytes / (ms_per_layer * 1e-3) / 1e9;
      std::cout << std::fixed << std::setprecision(4) << "  span=" << span
                << " kernel=" << name << " ms/layer=" << ms_per_layer
                << std::setprecision(1) << " unique_GBps=" << unique_gbps
                << " actual_GBps=" << actual_gbps << std::defaultfloat
                << std::setprecision(6) << "\n";
    };
    report(kernel, time_cold(launch_online));
    report("splitkv", time_cold(launch_splitkv));
    HIP_CHECK(hipFree(d_cold_k));
    HIP_CHECK(hipFree(d_cold_v));
    HIP_CHECK(hipFree(d_cold_pos));
  }
  for (auto& event : events) {
    HIP_CHECK(hipEventDestroy(event));
  }
  HIP_CHECK(hipFree(d_flush));
  HIP_CHECK(hipFree(d_scratch));

  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_sink));
  HIP_CHECK(hipFree(d_control));
  HIP_CHECK(hipStreamDestroy(stream));
}

void RunFlashCheck() {
  using gufo::hip::detail::FlintKernelPolicy;
  constexpr std::uint32_t kHeads = 32;
  constexpr std::uint32_t kKvHeads = 4;
  constexpr std::uint32_t kHeadDim = 128;
  constexpr std::uint32_t kMaxContext = 4096;
  constexpr std::size_t kKvWidth = kKvHeads * kHeadDim;
  constexpr float kScale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  std::uint32_t lcg = 0x9e3779b9;
  const auto next = [&lcg] {
    lcg = lcg * 1664525u + 1013904223u;
    return static_cast<float>(static_cast<double>(lcg >> 8) / 16777216.0 -
                              0.5);
  };

  std::vector<float> h_sink(kHeads);
  for (auto& s : h_sink) {
    s = next() * 2.0F;
  }
  // Cache filled for the whole context so every visible key is populated.
  const std::size_t cache_elems =
      static_cast<std::size_t>(kKvHeads) * kMaxContext * kHeadDim;
  std::vector<float> h_k(cache_elems);
  std::vector<float> h_v(cache_elems);
  for (auto& v : h_k) {
    v = next();
  }
  for (auto& v : h_v) {
    v = next();
  }

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_sink = nullptr,
        *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_q, static_cast<std::size_t>(1024) * kHeads * kHeadDim *
                                sizeof(float)));
  HIP_CHECK(hipMalloc(&d_sink, h_sink.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, h_k.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, h_v.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, static_cast<std::size_t>(1024) * kHeads *
                                  kHeadDim * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_sink, h_sink.data(), h_sink.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), h_k.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), h_v.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  int failures = 0;
  for (const auto [batch, start_pos, window] :
       std::vector<std::tuple<std::size_t, std::uint32_t, std::uint32_t>>{
           {16, 0, 0}, {16, 0, 513}, {512, 0, 513}, {512, 1024, 513}}) {
    std::vector<float> h_q(static_cast<std::size_t>(batch) * kHeads * kHeadDim);
    for (auto& v : h_q) {
      v = next();
    }
    HIP_CHECK(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    const FlintKernelPolicy policy{d_sink, window};
    HIP_CHECK(hipMemsetAsync(
        d_out, 0,
        static_cast<std::size_t>(batch) * kHeads * kHeadDim * sizeof(float),
        stream));
    if (!gufo::hip::LaunchFlintWmmaAttention(
            d_q, d_k, d_v, d_out, /*layer_idx=*/0, start_pos, batch,
            kMaxContext, stream, &policy)) {
      std::cerr << "FAIL: launcher refused batch=" << batch << "\n";
      ++failures;
      continue;
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<float> got(h_q.size());
    HIP_CHECK(hipMemcpy(got.data(), d_out, got.size() * sizeof(float),
                        hipMemcpyDeviceToHost));

    // CPU double reference.
    double max_abs = 0.0;
    double dot = 0.0, nr = 0.0, ng = 0.0;
    for (std::size_t t = 0; t < batch; ++t) {
      const std::uint32_t absolute = start_pos + static_cast<std::uint32_t>(t);
      const std::size_t p_begin =
          window != 0 && absolute + 1 > window ? absolute + 1 - window : 0;
      for (std::uint32_t h = 0; h < kHeads; ++h) {
        const std::uint32_t kv_h = h / (kHeads / kKvHeads);
        const float* q_h =
            h_q.data() + (t * kHeads + h) * kHeadDim;
        double m = h_sink[h];
        std::vector<double> scores(absolute + 1 - p_begin);
        for (std::size_t p = p_begin; p <= absolute; ++p) {
          const float* k_p =
              h_k.data() +
              ((std::size_t{0} * kKvHeads + kv_h) * kMaxContext + p) *
                  kHeadDim;
          double dotk = 0.0;
          for (std::size_t d = 0; d < kHeadDim; ++d) {
            dotk += static_cast<double>(q_h[d]) * static_cast<double>(k_p[d]);
          }
          scores[p - p_begin] = dotk * kScale;
          m = std::max(m, scores[p - p_begin]);
        }
        double l = std::exp(static_cast<double>(h_sink[h]) - m);
        for (auto& s : scores) {
          s = std::exp(s - m);
          l += s;
        }
        for (std::size_t d = 0; d < kHeadDim; ++d) {
          double acc = 0.0;
          for (std::size_t p = p_begin; p <= absolute; ++p) {
            const float* v_p =
                h_v.data() +
                ((std::size_t{0} * kKvHeads + kv_h) * kMaxContext + p) *
                    kHeadDim;
            acc += scores[p - p_begin] * static_cast<double>(v_p[d]);
          }
          const double ref = acc / l;
          const double g =
              got[(t * kHeads + h) * kHeadDim + d];
          max_abs = std::max(max_abs, std::abs(ref - g));
          dot += ref * g;
          nr += ref * ref;
          ng += g * g;
        }
      }
    }
    const double cos = dot / (std::sqrt(nr) * std::sqrt(ng) + 1e-30);
    std::cout << "  flashcheck batch=" << batch << " start=" << start_pos
              << " window=" << window << ": cos=" << cos
              << " max_abs=" << max_abs << "\n";
    if (!(cos > 0.99999) || max_abs > 2e-4) {
      std::cerr << "FAIL: flash parity\n";
      ++failures;
    }
  }

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_sink));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipStreamDestroy(stream));
  if (failures != 0) {
    std::exit(1);
  }
  std::cout << "  flashcheck: PASS\n";
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
      GgmlType::kQ8_0, GgmlType::kQ8_0, d_x, d_act_moe, d_weights_full,
      nullptr, 0, 256, kM, kM, kK, kSlabBytes, kSlabBytes, stream);
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
      config.expert_count * expert_ff + shared_ff +  // act_moe
      gufo::hip::kFlintMoeMaxJobs * hidden;  // down partials (3.6)
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
  float* moe_partials =
      carve(gufo::hip::kFlintMoeMaxJobs * hidden);  // 3.6 down partials
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
    const std::size_t up_slab_bytes = gufo::quant::EncodedSizeBytes(
        layer.ffn_up_exps.type, expert_slab);
    gufo::hip::LaunchFlintFusedMoESwiGLU(
        layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
        layer.ffn_gate_shexp.data, layer.ffn_up_shexp.data,
        layer.ffn_gate_exps.type, layer.ffn_up_exps.type,
        layer.ffn_gate_shexp.type, layer.ffn_up_shexp.type, x, act_moe,
        weights_full, job_list, config.expert_used_count + 1,
        config.expert_count, expert_ff, shared_ff, hidden, slab_bytes,
        up_slab_bytes, stream);
    gufo::hip::LaunchFlintFusedMoEDown(layer.ffn_down_exps.data,
                                       layer.ffn_down_shexp.data,
                                       layer.ffn_down_exps.type,
                                       layer.ffn_down_shexp.type, act_moe,
                                       moe_partials, mlp, nullptr,
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
      const std::size_t up_slab_bytes = gufo::quant::EncodedSizeBytes(
          layer.ffn_up_exps.type, expert_slab);
      gufo::hip::LaunchFlintFusedMoESwiGLU(
          layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
          layer.ffn_gate_shexp.data, layer.ffn_up_shexp.data,
          layer.ffn_gate_exps.type, layer.ffn_up_exps.type,
          layer.ffn_gate_shexp.type, layer.ffn_up_shexp.type, x, act_moe,
          weights_full, job_list, config.expert_used_count + 1,
          config.expert_count, expert_ff, shared_ff, hidden, slab_bytes,
          up_slab_bytes, stream);
      HIP_CHECK(hipStreamSynchronize(stream));
    });
    const double f_down_us = timed([&] {
      gufo::hip::LaunchFlintFusedMoEDown(
          layer.ffn_down_exps.data, layer.ffn_down_shexp.data,
          layer.ffn_down_exps.type, layer.ffn_down_shexp.type, act_moe,
          moe_partials, mlp_fused, nullptr, weights_full, job_list,
          config.expert_used_count + 1,
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

namespace {

/// F-L4 parity and rate check: the chunked prefetch path against the
/// token-by-token path on the same tokens, then the same chunk split in two
/// (a chunk boundary must not move the logits), then a timed run for t/s.
void RunChunkParity(const gufo::hip::FlintGpuModel& model,
                    const std::vector<gufo::tokenization::TokenId>& tokens,
                    const std::vector<gufo::tokenization::TokenId>& full_tokens,
                    std::uint32_t max_context) {
  const auto& config = model.GetConfig();
  if (!gufo::hip::FlintPrefillSupported(model.GetWeights())) {
    std::cout << "chunk: model does not take the batched routes\n";
    return;
  }
  const std::size_t n = tokens.size();
  Expect(n > 1, "chunk parity needs at least two tokens");
  Expect(n <= gufo::hip::kFlintPrefillBatchCapacity,
         "chunk parity prompt exceeds one chunk");

  gufo::hip::FlintGpuArena arena(config, max_context);
  const auto buffers = arena.GetBuffers();
  const auto prefill = arena.GetPrefillBuffers();
  std::vector<float> chunk_logits(config.vocab_size);
  std::vector<float> token_logits(config.vocab_size);
  std::vector<float> split_logits(config.vocab_size);

  // Warm every kernel (weights and code pages), then time one chunk.
  gufo::hip::ExecuteFlintPrefillChunk(model.GetWeights(), tokens, 0,
                                      /*compute_logits=*/true, buffers,
                                      prefill);
  HIP_CHECK(hipStreamSynchronize(buffers.stream));
  HIP_CHECK(hipMemcpy(chunk_logits.data(), buffers.logits,
                      chunk_logits.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  const auto t0 = std::chrono::steady_clock::now();
  gufo::hip::FlintPrefillTimings timings;
  gufo::hip::ExecuteFlintPrefillChunk(model.GetWeights(), tokens, 0,
                                      /*compute_logits=*/true, buffers,
                                      prefill, &timings);
  HIP_CHECK(hipStreamSynchronize(buffers.stream));
  const auto t1 = std::chrono::steady_clock::now();
  const double chunk_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  // W3 sub-stage profile (2.6 audit): phase isolation by A/B. One warm +
  // one timed chunk per phase; the attention stage is already event-timed
  // in `timings`. QK cost = full - skipPV; PV cost = full - skipQK.
  if (std::getenv("GUFO_FLINT_ATTN_PROFILE") != nullptr) {
    const auto timed_chunk = [&](unsigned skip) {
      gufo::hip::AttnProfileSetSkip(skip, buffers.stream);
      const auto a0 = std::chrono::steady_clock::now();
      gufo::hip::FlintPrefillTimings t;
      gufo::hip::ExecuteFlintPrefillChunk(model.GetWeights(), tokens, 0,
                                          /*compute_logits=*/false, buffers,
                                          prefill, &t);
      HIP_CHECK(hipStreamSynchronize(buffers.stream));
      const auto a1 = std::chrono::steady_clock::now();
      gufo::hip::AttnProfileSetSkip(0u, buffers.stream);
      return std::make_pair(t.attention_ms,
                            std::chrono::duration<double, std::milli>(a1 - a0)
                                .count());
    };
    const auto [full_attn, full_ms] = timed_chunk(0u);
    const auto [nopv_attn, nopv_ms] = timed_chunk(2u);
    const auto [noqk_attn, noqk_ms] = timed_chunk(1u);
    const auto [noattn_attn, noattn_ms] = timed_chunk(6u);
    std::cout << "  attn A/B: full=" << full_attn << " ms (chunk "
              << full_ms << "), -PV=" << nopv_attn << " (chunk " << nopv_ms
              << "), -QK=" << noqk_attn << " (chunk " << noqk_ms << ")"
              << ", -body=" << noattn_attn << " (chunk " << noattn_ms << ")\n"
              << "  => QK ~" << (full_attn - noqk_attn)
              << " ms, PV ~" << (full_attn - nopv_attn)
              << " ms, norm/rope/kvwrite+launch ~" << noattn_attn << " ms\n";
  }

  // Token-by-token reference over the same positions.
  // GUFO_FLINT_DECODE_SYNC=N: sync every N decode steps. Queue-accumulation
  // discriminator for the G6 graph-replay hang (0/unset = no sync, the
  // production behaviour).
  const char* sync_env = std::getenv("GUFO_FLINT_DECODE_SYNC");
  const std::size_t sync_every =
      sync_env != nullptr ? static_cast<std::size_t>(std::atoi(sync_env)) : 0;
  for (std::size_t i = 0; i < n; ++i) {
    gufo::hip::ExecuteFlintDecodeStep(model.GetWeights(), tokens[i],
                                      static_cast<std::uint32_t>(i),
                                      /*compute_logits=*/i + 1 == n, buffers,
                                      /*timings=*/nullptr,
                                      arena.GetGraphState());
    if (sync_every != 0 && (i + 1) % sync_every == 0) {
      HIP_CHECK(hipStreamSynchronize(buffers.stream));
    }
  }
  HIP_CHECK(hipStreamSynchronize(buffers.stream));
  HIP_CHECK(hipMemcpy(token_logits.data(), buffers.logits,
                      token_logits.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  // Chunk boundary: half the tokens, then the other half.
  const std::size_t half = n / 2;
  gufo::hip::ExecuteFlintPrefillChunk(
      model.GetWeights(), std::span<const std::uint32_t>(tokens.data(), half),
      0, /*compute_logits=*/false, buffers, prefill);
  gufo::hip::ExecuteFlintPrefillChunk(
      model.GetWeights(),
      std::span<const std::uint32_t>(tokens.data() + half, n - half),
      static_cast<std::uint32_t>(half), /*compute_logits=*/true, buffers,
      prefill);
  HIP_CHECK(hipStreamSynchronize(buffers.stream));
  HIP_CHECK(hipMemcpy(split_logits.data(), buffers.logits,
                      split_logits.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  // G6, multiple split points (1.5): every split of the same chunk must
  // reproduce the unsplit logits bit-exactly — the flash kernel's fixed
  // 16-key tile order makes the accumulation position-determined only.
  // Split points cover the tile boundaries, a 1-token tail, and uneven
  // three-way splits.
  if (std::getenv("GUFO_FLINT_SPLITS") != nullptr) {
    int split_failures = 0;
    for (const std::size_t k : {std::size_t{2}, std::size_t{128},
                                std::size_t{255}, std::size_t{256},
                                std::size_t{257}, std::size_t{383},
                                std::size_t{384}, std::size_t{511},
                                std::size_t{512}}) {
      if (k >= n || n - k < 1) {
        continue;
      }
      gufo::hip::ExecuteFlintPrefillChunk(
          model.GetWeights(), std::span<const std::uint32_t>(tokens.data(), k),
          0, /*compute_logits=*/false, buffers, prefill);
      gufo::hip::ExecuteFlintPrefillChunk(
          model.GetWeights(),
          std::span<const std::uint32_t>(tokens.data() + k, n - k),
          static_cast<std::uint32_t>(k), /*compute_logits=*/true, buffers,
          prefill);
      HIP_CHECK(hipStreamSynchronize(buffers.stream));
      HIP_CHECK(hipMemcpy(split_logits.data(), buffers.logits,
                          split_logits.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      const bool exact =
          std::equal(split_logits.begin(), split_logits.end(),
                     chunk_logits.begin());
      if (!exact) {
        double max_abs = 0.0;
        for (std::size_t i = 0; i < split_logits.size(); ++i) {
          max_abs = std::max(
              max_abs,
              std::abs(static_cast<double>(split_logits[i]) - chunk_logits[i]));
        }
        std::cout << "  split@" << k << ": NOT bit-exact (max_abs " << max_abs
                  << ")\n";
        ++split_failures;
      } else {
        std::cout << "  split@" << k << ": bit-exact\n";
      }
    }
    // Three-way split: 341 + 341 + remainder.
    if (n >= 3) {
      const std::size_t k1 = n / 3;
      const std::size_t k2 = 2 * (n / 3);
      gufo::hip::ExecuteFlintPrefillChunk(
          model.GetWeights(),
          std::span<const std::uint32_t>(tokens.data(), k1), 0,
          /*compute_logits=*/false, buffers, prefill);
      gufo::hip::ExecuteFlintPrefillChunk(
          model.GetWeights(),
          std::span<const std::uint32_t>(tokens.data() + k1, k2 - k1),
          static_cast<std::uint32_t>(k1), /*compute_logits=*/false, buffers,
          prefill);
      gufo::hip::ExecuteFlintPrefillChunk(
          model.GetWeights(),
          std::span<const std::uint32_t>(tokens.data() + k2, n - k2),
          static_cast<std::uint32_t>(k2), /*compute_logits=*/true, buffers,
          prefill);
      HIP_CHECK(hipStreamSynchronize(buffers.stream));
      HIP_CHECK(hipMemcpy(split_logits.data(), buffers.logits,
                          split_logits.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      const bool exact = std::equal(split_logits.begin(), split_logits.end(),
                                    chunk_logits.begin());
      std::cout << "  split@3way(" << k1 << "," << k2 << "): "
                << (exact ? "bit-exact" : "NOT bit-exact") << "\n";
      if (!exact) {
        ++split_failures;
      }
    }
    if (split_failures != 0) {
      std::cout << "  G6 multi-split: FAIL (" << split_failures << ")\n";
    } else {
      std::cout << "  G6 multi-split: PASS\n";
    }
  }

  const auto vs_token = CompareLogits(token_logits, chunk_logits);
  const auto vs_split = CompareLogits(chunk_logits, split_logits);
  // Where does the chunk-vs-token disagreement live? Quantised-activation
  // GEMMs (chunk) against f32 GEMV (token) move tail logits far more than
  // head logits; report both regimes separately.
  float tail_max = 0.0F;
  float head_max = 0.0F;
  {
    std::size_t tail_idx = 0, head_idx = 0;
    for (std::size_t i = 0; i < config.vocab_size; ++i) {
      const float d = std::abs(token_logits[i] - chunk_logits[i]);
      if (token_logits[i] < -10.0F) {
        if (d > tail_max) {
          tail_max = d;
          tail_idx = i;
        }
      } else if (d > head_max) {
        head_max = d;
        head_idx = i;
      }
    }
    std::cout << "  logit diff: head(|l|<=10) max=" << head_max
              << " @token " << token_logits[head_idx] << " vs "
              << chunk_logits[head_idx] << "; tail(l<-10) max=" << tail_max
              << " @token " << token_logits[tail_idx] << " vs "
              << chunk_logits[tail_idx] << "\n";
  }
  std::cout << "chunk(" << n << " tokens): cos=" << vs_token.cosine
            << " dlogp=" << vs_token.max_logp_diff
            << " argmax(chunk/token)=" << vs_token.argmax_gpu << "/"
            << vs_token.argmax_cpu << "\n"
            << "  split(" << half << "+" << (n - half)
            << "): cos=" << vs_split.cosine
            << " dlogp=" << vs_split.max_logp_diff
            << " argmax=" << vs_split.argmax_gpu << "/" << vs_split.argmax_cpu
            << "\n"
            << "  rate: " << chunk_ms << " ms/chunk = "
            << (n * 1000.0 / chunk_ms) << " t/s\n";
  // Per-stage breakdown (tuning item 0.3): where the chunk's GPU time goes.
  std::cout << "  stages: norm=" << timings.norm_ms << " qkv=" << timings.qkv_ms
            << " attn=" << timings.attention_ms << " oproj=" << timings.oproj_ms
            << " router=" << timings.router_ms
            << " shared=" << timings.shared_ms
            << " routed=" << timings.routed_ms
            << " dense=" << timings.dense_ms << " tail=" << timings.tail_ms
            << " ms (sum=" << timings.total_ms() << ")\n";
  // Position-lottery check: the parity gates above judge the chunk path at
  // ONE position — the prompt's last token — which is an arbitrary position
  // when n < capacity. Re-run the same prompt at full capacity and evaluate
  // the head at the SAME position (n-1). Per-token chunk math must be
  // chunk-size independent (per-row GEMM independence, causal attention
  // spans, per-token top-k), so the two must agree bit-for-bit; a mismatch
  // here means an n-dependent kernel, not an unlucky position.
  if (n < full_tokens.size() &&
      full_tokens.size() == gufo::hip::kFlintPrefillBatchCapacity) {
    gufo::hip::ExecuteFlintPrefillChunk(model.GetWeights(), full_tokens, 0,
                                        /*compute_logits=*/false, buffers,
                                        prefill);
    gufo::hip::LaunchBatchedRMSNorm(
        prefill.hidden + static_cast<std::size_t>(n - 1) * config.hidden_size,
        static_cast<const float*>(model.GetWeights().output_norm.data),
        buffers.normed, nullptr, 1, config.hidden_size,
        1e-6F, buffers.stream);
    gufo::hip::LaunchGEMV(model.GetWeights().output.data, model.GetWeights().output.type,
               buffers.normed, buffers.logits, config.vocab_size,
               config.hidden_size, buffers.stream);
    HIP_CHECK(hipStreamSynchronize(buffers.stream));
    std::vector<float> pos_logits(config.vocab_size);
    HIP_CHECK(hipMemcpy(pos_logits.data(), buffers.logits,
                        pos_logits.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    float max_abs = 0.0F;
    bool bits_equal = true;
    for (std::size_t i = 0; i < pos_logits.size(); ++i) {
      if (std::memcmp(&pos_logits[i], &chunk_logits[i], sizeof(float)) != 0) {
        bits_equal = false;
      }
      max_abs = std::fmax(max_abs, std::abs(pos_logits[i] - chunk_logits[i]));
    }
    std::cout << "  position-lottery: chunk(" << full_tokens.size()
              << ")@pos" << (n - 1) << " vs chunk(" << n
              << ")@last: max_abs=" << max_abs
              << (bits_equal ? " (bit-identical)" : " (DIFFERS)") << "\n";
    Expect(max_abs <= 1e-4,
           "chunk per-token math must not depend on the chunk size");

  Expect(vs_token.cosine >= 0.999, "chunk vs token cosine");
  Expect(vs_token.argmax_cpu == vs_token.argmax_gpu,
         "chunk vs token argmax identity");
  // Cross-precision gate: the chunk path runs q8_1-activation WMMA GEMMs,
  // the token path f32 GEMV, so the strict G3 delta-log-p gate (written for
  // decode-vs-oracle, same arithmetic) does not transfer — deep-tail logits
  // (p ~ e^-35, unsampleable) absorb the whole quantisation wobble while the
  // head stays tight (measured: head <= 0.41, tail <= 2.63, argmax equal,
  // split-boundary bit-exact). Gate on the head and ignore the tail.
  //
  // Scope: the 0.5 threshold was calibrated at the full-capacity chunk (W1,
  // position 511). The position-lottery check above proves per-token chunk
  // math is chunk-size independent, so for n < capacity the last position is
  // an arbitrary prompt position whose routing landscape may sit near a
  // top-6 boundary; there the head delta expresses expert-selection flips
  // (measured 2.0-6.3 across router/MoE variants at n=181, all with
  // bit-identical chunk-internal math and cos >= 0.997) and is reported,
  // not gated. The n==capacity run carries the assertion.
  if (n == gufo::hip::kFlintPrefillBatchCapacity) {
    Expect(head_max <= 0.5F, "chunk vs token head logit delta");
  } else {
    std::cout << "  head delta (informational, n<capacity): " << head_max
              << "\n";
  }
  Expect(vs_split.argmax_cpu == vs_split.argmax_gpu,
         "chunk boundary argmax identity");
  Expect(vs_split.max_logp_diff <= 1e-4,
         "chunk boundary must not move the logits");

  }
}

}  // namespace

namespace {

/// F-J3 key-count proof: a decode step at 4k and 32k must touch exactly
/// min(pos+1, window) keys on a windowed layer and pos+1 on a global layer.
/// Runs the graph-captured path (default) and the legacy path
/// (GUFO_FLINT_DECODE_GRAPH=0).
void RunKeyCountProof(const gufo::hip::FlintGpuModel& model) {
  const auto& config = model.GetConfig();
  const auto& weights = model.GetWeights();
  constexpr std::uint32_t kCtx = 32768;
  const std::uint32_t positions[] = {4095u, 32767u};
  for (int graph_pass = 0; graph_pass < 2; ++graph_pass) {
    if (graph_pass == 1) {
      setenv("GUFO_FLINT_DECODE_GRAPH", "0", 1);
    }
    gufo::hip::FlintGpuArena arena(config, kCtx);
    const auto buffers = arena.GetBuffers();
    for (const std::uint32_t pos : positions) {
      gufo::hip::FlintKeyCountSet(true, buffers.stream);
      gufo::hip::ExecuteFlintDecodeStep(weights, /*token=*/1u, pos,
                                        /*compute_logits=*/false, buffers,
                                        nullptr, arena.GetGraphState());
      HIP_CHECK(hipStreamSynchronize(buffers.stream));
      std::vector<unsigned int> counts(config.num_layers, 0u);
      gufo::hip::FlintKeyCountRead(counts.data(), config.num_layers,
                                   buffers.stream);
      for (std::uint32_t l = 0; l < config.num_layers; ++l) {
        const bool windowed = weights.layers[l].attrs.windowed;
        const std::uint32_t expected =
            windowed ? std::min<std::uint32_t>(pos + 1, config.sliding_window)
                     : pos + 1;
        Expect(counts[l] == expected,
               "key count layer " + std::to_string(l) + " pos " +
                   std::to_string(pos) + " graph " +
                   std::to_string(graph_pass) + ": got " +
                   std::to_string(counts[l]) + " expected " +
                   std::to_string(expected));
      }
      gufo::hip::FlintKeyCountSet(false, buffers.stream);
      std::cout << "  keycount graph=" << graph_pass << " pos=" << pos
                << " ok (" << config.num_layers << " layers)\n";
    }
  }
  // Prefill: a full chunk ending at the 32k cap. The flash kernel's counter
  // is per 32-query block (floored to the 16-key tile grid), so a windowed
  // layer is bounded by window + 31 + 15 keys; a global layer sees the whole
  // prefix up to the chunk's last position.
  if (gufo::hip::FlintPrefillSupported(weights)) {
    const std::size_t n = gufo::hip::kFlintPrefillBatchCapacity;
    const std::uint32_t start_pos = kCtx - static_cast<std::uint32_t>(n);
    std::vector<gufo::tokenization::TokenId> tokens(n, 1u);
    gufo::hip::FlintGpuArena arena(config, kCtx);
    const auto buffers = arena.GetBuffers();
    const auto prefill = arena.GetPrefillBuffers();
    gufo::hip::FlintKeyCountSet(true, buffers.stream);
    gufo::hip::ExecuteFlintPrefillChunk(weights, tokens, start_pos,
                                        /*compute_logits=*/false, buffers,
                                        prefill);
    HIP_CHECK(hipStreamSynchronize(buffers.stream));
    std::vector<unsigned int> counts(config.num_layers, 0u);
    gufo::hip::FlintKeyCountRead(counts.data(), config.num_layers,
                                 buffers.stream);
    gufo::hip::FlintKeyCountSet(false, buffers.stream);
    const std::uint32_t slack = 31u + 15u;
    for (std::uint32_t l = 0; l < config.num_layers; ++l) {
      if (weights.layers[l].attrs.windowed) {
        Expect(counts[l] != 0 && counts[l] <= config.sliding_window + slack,
               "prefill key count layer " + std::to_string(l) + ": got " +
                   std::to_string(counts[l]) + " bound " +
                   std::to_string(config.sliding_window + slack));
      } else {
        Expect(counts[l] > config.sliding_window + slack,
               "prefill global layer " + std::to_string(l) +
                   " must see the full prefix, got " +
                   std::to_string(counts[l]));
      }
    }
    std::cout << "  keycount prefill start_pos=" << start_pos << " n=" << n
              << " ok (windowed <= " << config.sliding_window + slack << ")\n";
  }
  std::cout << "flint keycount proof ok\n";
}

}  // namespace

namespace {

/// Dominant weight quantisation of the artifact: the GGML type that owns the
/// most bytes among quantised tensors. Float tensors (norms, biases, router
/// scalars) are skipped so a few F32 tensors cannot mask the body of the model.
gufo::core::GgmlType DominantWeightType(const gufo::core::GgufReader& reader) {
  std::map<gufo::core::GgmlType, std::size_t> bytes_by_type;
  for (const auto& t : reader.GetTensors()) {
    switch (t.type) {
      case gufo::core::GgmlType::kF32:
      case gufo::core::GgmlType::kF16:
      case gufo::core::GgmlType::kBF16:
      case gufo::core::GgmlType::kI32:
        continue;
      default:
        bytes_by_type[t.type] += t.size_bytes;
    }
  }
  gufo::core::GgmlType best = gufo::core::GgmlType::kF16;
  std::size_t best_bytes = 0;
  for (const auto& [type, bytes] : bytes_by_type) {
    if (bytes > best_bytes) {
      best = type;
      best_bytes = bytes;
    }
  }
  return best;
}

/// Quant-aware fp32-path |dlogp| gate. Low-bit quants (IQ2/IQ3/IQ4/Q2-Q4
/// families) have near-tie routing and argmax forks that legitimately move
/// dlogp by O(0.1-1) while cos stays >= 0.999, so they get a loose bound.
/// Q5 and up keep the tight bound. Float weights keep the tight bound too.
double Fp32LogpGate(gufo::core::GgmlType weight_type) {
  using T = gufo::core::GgmlType;
  switch (weight_type) {
    case T::kIQ2_XXS:
    case T::kIQ3_S:
    case T::kIQ4_NL:
    case T::kIQ4_XS:
    case T::kQ2_K:
    case T::kQ3_K:
    case T::kQ4_0:
    case T::kQ4_1:
    case T::kQ4_K:
      return 0.5;
    default:
      return 5e-3;
  }
}

/// 7.6 exactness probe: does the prefill (verify) path produce the same greedy
/// argmax as step-by-step decode, row for row? Speculation is only
/// greedy-exact if it does. Builds a sequence by greedy decode, then re-runs
/// it through the prefill path in chunks of `k` and compares the per-row
/// argmax. Reports the mismatch rate and the top1-top2 margin at each
/// mismatch (a flip only happens at a near-tie).
void RunVerifyExactness(
    const gufo::hip::FlintGpuModel& model,
    const std::vector<gufo::tokenization::TokenId>& prompt_tokens,
    std::uint32_t max_context) {
  const auto& config = model.GetConfig();
  const std::size_t hidden = config.hidden_size;
  const std::size_t vocab = config.vocab_size;
  const std::size_t n = 256;  // sequence length
  const std::size_t k =  // verify chunk size (speculation draft cap)
      std::getenv("GUFO_FLINT_VERIFY_K") != nullptr
          ? static_cast<std::size_t>(std::atoi(std::getenv("GUFO_FLINT_VERIFY_K")))
          : 8;

  gufo::hip::FlintGpuArena arena(config, max_context);
  const auto buffers = arena.GetBuffers();
  const auto prefill = arena.GetPrefillBuffers();

  // Pass 1: greedy decode from the prompt, recording the sequence and the
  // argmax at every position (argmax[i] predicts token i+1).
  std::vector<std::uint32_t> seq(prompt_tokens.begin(), prompt_tokens.end());
  std::vector<std::uint32_t> decode_argmax;
  std::vector<float> logits(vocab);
  std::vector<float> decode_logits(n * vocab);
  const auto host_argmax = [&](const float* row, float* margin) {
    std::size_t best = 0;
    float second = -1e30F;
    for (std::size_t v = 0; v < vocab; ++v) {
      if (row[v] > row[best]) {
        second = row[best];
        best = v;
      } else if (row[v] > second) {
        second = row[v];
      }
    }
    if (margin != nullptr) {
      *margin = row[best] - second;
    }
    return static_cast<std::uint32_t>(best);
  };
  const auto decode_one = [&](std::uint32_t token, std::uint32_t pos) {
    gufo::hip::ExecuteFlintDecodeStep(model.GetWeights(), token, pos,
                                      /*compute_logits=*/true, buffers,
                                      /*timings=*/nullptr,
                                      arena.GetGraphState());
    HIP_CHECK(hipStreamSynchronize(buffers.stream));
    HIP_CHECK(hipMemcpy(logits.data(), buffers.logits,
                        vocab * sizeof(float), hipMemcpyDeviceToHost));
    std::copy(logits.begin(), logits.end(),
              decode_logits.begin() + static_cast<std::size_t>(pos) * vocab);
    return host_argmax(logits.data(), nullptr);
  };
  for (std::size_t i = 0; i < seq.size(); ++i) {
    decode_argmax.push_back(
        decode_one(seq[i], static_cast<std::uint32_t>(i)));
  }
  while (seq.size() < n) {
    const std::uint32_t next = decode_argmax.back();
    seq.push_back(next);
    decode_argmax.push_back(
        decode_one(next, static_cast<std::uint32_t>(seq.size() - 1)));
  }

  // Pass 2: prefill in chunks of k through the production verify tail
  // (ExecuteFlintVerifyChunk), reading back the per-row argmax and logits.
  std::vector<std::uint32_t> logits_tokens(k);
  std::vector<float> logits_all(k * vocab);
  std::vector<std::uint32_t> prefill_argmax(n);
  std::vector<float> prefill_margin(n);
  std::vector<float> prefill_logits(n * vocab);
  for (std::size_t start = 0; start < n; start += k) {
    const std::size_t kk = std::min(k, n - start);
    gufo::hip::ExecuteFlintVerifyChunk(
        model.GetWeights(),
        std::span<const std::uint32_t>(seq.data() + start, kk),
        static_cast<std::uint32_t>(start), buffers, prefill);
    HIP_CHECK(hipStreamSynchronize(buffers.stream));
    HIP_CHECK(hipMemcpy(logits_tokens.data(), prefill.verify_tokens,
                        kk * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(logits_all.data(), prefill.verify_logits,
                        kk * vocab * sizeof(float), hipMemcpyDeviceToHost));
    for (std::size_t j = 0; j < kk; ++j) {
      prefill_argmax[start + j] = logits_tokens[j];
      host_argmax(logits_all.data() + j * vocab, &prefill_margin[start + j]);
      std::copy(logits_all.begin() + j * vocab,
                logits_all.begin() + (j + 1) * vocab,
                prefill_logits.begin() + (start + j) * vocab);
    }
  }

  // Logit-space divergence between the two paths (cosine + max abs error),
  // to separate "small error, near-tie flips" from "large divergence".
  double cos_sum = 0.0;
  double max_abs = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const float* a = decode_logits.data() + i * vocab;
    const float* b = prefill_logits.data() + i * vocab;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (std::size_t v = 0; v < vocab; ++v) {
      dot += static_cast<double>(a[v]) * b[v];
      na += static_cast<double>(a[v]) * a[v];
      nb += static_cast<double>(b[v]) * b[v];
      max_abs = std::max(max_abs,
                         std::abs(static_cast<double>(a[v]) - b[v]));
    }
    cos_sum += dot / (std::sqrt(na) * std::sqrt(nb));
  }
  std::cout << "  logit divergence: mean cos=" << (cos_sum / n)
            << " max|delta|=" << max_abs << "\n";

  // Compare.
  std::size_t mismatches = 0;
  double min_mismatch_margin = 1e30;
  for (std::size_t i = 0; i < n; ++i) {
    if (decode_argmax[i] != prefill_argmax[i]) {
      ++mismatches;
      min_mismatch_margin =
          std::min(min_mismatch_margin,
                   static_cast<double>(prefill_margin[i]));
      if (mismatches <= 12) {
        std::cout << "  mismatch @" << i << ": decode=" << decode_argmax[i]
                  << " prefill=" << prefill_argmax[i]
                  << " margin=" << prefill_margin[i] << "\n";
      }
    }
  }
  std::cout << "verify exactness: " << mismatches << "/" << n
            << " mismatches (" << (100.0 * mismatches / n) << "%)";
  if (mismatches != 0) {
    std::cout << ", min mismatch margin=" << min_mismatch_margin;
  }
  std::cout << "\n";
}

/// 7.6 rollback proof: a verify chunk writes KV rows for every draft token,
/// but a rejected suffix leaves those rows stale. The next decode must not
/// read them. Arena A verifies all k draft rows, arena B verifies only the
/// j accepted rows; a decode at P+j must then be bit-identical in both,
/// because it reads only [0, P+j).
void RunRollbackCheck(const gufo::hip::FlintGpuModel& model,
                      const std::vector<gufo::tokenization::TokenId>& prompt_tokens,
                      std::uint32_t max_context) {
  const auto& config = model.GetConfig();
  const std::size_t vocab = config.vocab_size;
  constexpr std::size_t kPrefix = 16;
  constexpr std::size_t kDraft = 8;  // k: verify chunk size
  constexpr std::size_t kAccept = 3;  // j: accepted rows before rejection
  Expect(prompt_tokens.size() >= kPrefix + kDraft,
         "rollback prompt must tokenize to at least 24 tokens");

  const std::vector<std::uint32_t> prefix(
      prompt_tokens.begin(), prompt_tokens.begin() + kPrefix);
  const std::vector<std::uint32_t> draft(
      prompt_tokens.begin() + kPrefix,
      prompt_tokens.begin() + kPrefix + kDraft);
  const auto P = static_cast<std::uint32_t>(prefix.size());

  gufo::hip::FlintGpuArena arena_a(config, max_context);
  gufo::hip::FlintGpuArena arena_b(config, max_context);
  const auto buf_a = arena_a.GetBuffers();
  const auto buf_b = arena_b.GetBuffers();
  const auto pre_a = arena_a.GetPrefillBuffers();
  const auto pre_b = arena_b.GetPrefillBuffers();

  gufo::hip::ExecuteFlintPrefillChunk(model.GetWeights(), prefix, 0,
                                      /*compute_logits=*/false, buf_a, pre_a);
  gufo::hip::ExecuteFlintPrefillChunk(model.GetWeights(), prefix, 0,
                                      /*compute_logits=*/false, buf_b, pre_b);
  HIP_CHECK(hipStreamSynchronize(buf_a.stream));
  HIP_CHECK(hipStreamSynchronize(buf_b.stream));

  // Arena A: verify all k rows (stale rows [P+j, P+k) remain afterwards).
  gufo::hip::ExecuteFlintVerifyChunk(model.GetWeights(), draft, P, buf_a,
                                     pre_a);
  gufo::hip::ExecuteFlintDecodeStep(
      model.GetWeights(), draft[kAccept], P + kAccept,
      /*compute_logits=*/true, buf_a, nullptr, arena_a.GetGraphState());

  // Arena B: verify only the j accepted rows, then decode at the same spot.
  gufo::hip::ExecuteFlintVerifyChunk(
      model.GetWeights(),
      std::span<const std::uint32_t>(draft.data(), kAccept), P, buf_b, pre_b);
  gufo::hip::ExecuteFlintDecodeStep(
      model.GetWeights(), draft[kAccept], P + kAccept,
      /*compute_logits=*/true, buf_b, nullptr, arena_b.GetGraphState());

  HIP_CHECK(hipStreamSynchronize(buf_a.stream));
  HIP_CHECK(hipStreamSynchronize(buf_b.stream));
  std::vector<float> logits_a(vocab);
  std::vector<float> logits_b(vocab);
  HIP_CHECK(hipMemcpy(logits_a.data(), buf_a.logits, vocab * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(logits_b.data(), buf_b.logits, vocab * sizeof(float),
                      hipMemcpyDeviceToHost));

  if (std::memcmp(logits_a.data(), logits_b.data(), vocab * sizeof(float)) !=
      0) {
    for (std::size_t v = 0; v < vocab; ++v) {
      if (std::memcmp(&logits_a[v], &logits_b[v], sizeof(float)) != 0) {
        std::cout << "rollback: first diff at index " << v << " a="
                  << logits_a[v] << " b=" << logits_b[v] << "\n";
        break;
      }
    }
    Expect(false, "rollback: decode after rejected verify is not bit-exact");
  }
  std::cout << "rollback: k=" << kDraft << " j=" << kAccept
            << " bit-exact PASS\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string_view(argv[1]) == "synthq8") {
    SynthQ8SwiGLUCheck();
    return 0;
  }
  if (argc > 1 && std::string_view(argv[1]) == "onlinedecode") {
    RunOnlineDecodeCheck();
    return 0;
  }
  if (argc > 1 && std::string_view(argv[1]) == "flashcheck") {
    RunFlashCheck();
    return 0;
  }
  if (argc > 1 && std::string_view(argv[1]) == "flashbench") {
    RunFlashBench();
    return 0;
  }
  const char* env = std::getenv("GUFO_FLINT_GGUF");
  if (env == nullptr || *env == '\0') {
    std::cout << "SKIPPED: GUFO_FLINT_GGUF not set\n";
    return 77;
  }

  std::uint32_t max_context = 4096;
  std::uint32_t steps = 8;
  const std::string_view mode =
      argc > 1 ? std::string_view(argv[1]) : std::string_view{};
  bool probe = mode == "probe";
  const bool chunk = mode == "chunk";
  const bool verify = mode == "verify";
  const bool rollback = mode == "rollback";
  if (!mode.empty() && !probe && !chunk && !verify && !rollback &&
      mode != "keycount")
    max_context = static_cast<std::uint32_t>(std::atoi(argv[1]));
  if (argc > 2) steps = static_cast<std::uint32_t>(std::atoi(argv[2]));
  Expect(steps > 0, "steps must be positive");

  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(env, &error);
  Expect(reader != nullptr, "open GGUF: " + error);
  const gufo::core::GgmlType weight_type = DominantWeightType(*reader);

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

  if (chunk) {
    // Optional third arg: token count for the chunk-size sweep (tuning item
    // 2.5 — 181/512; capacity is 512 so larger is clamped). Default is the
    // full 512-token capacity.
    const std::size_t want =
        argc > 3
            ? static_cast<std::size_t>(std::atoi(argv[3]))
            : gufo::hip::kFlintPrefillBatchCapacity;
    const std::size_t count = std::min<std::size_t>(
        want, gufo::hip::kFlintPrefillBatchCapacity);
    Expect(count > 1, "chunk token count must be at least two");
    std::vector<gufo::tokenization::TokenId> tokens;
    while (tokens.size() < count) {
      tokens.insert(tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
    }
    tokens.resize(count);
    // Full-capacity sequence for the position-lottery check (same prompt
    // pattern repeated; the first `count` tokens match `tokens`).
    std::vector<gufo::tokenization::TokenId> tokens_full;
    while (tokens_full.size() < gufo::hip::kFlintPrefillBatchCapacity) {
      tokens_full.insert(tokens_full.end(), prompt_tokens.begin(),
                         prompt_tokens.end());
    }
    tokens_full.resize(gufo::hip::kFlintPrefillBatchCapacity);
    // Optional second arg: arena max_context (default 4096). The serve path
    // runs 32768 — the 6.6 GB KV plane changes the allocation landscape, so
    // the chunk must be checked there too.
    const std::uint32_t chunk_ctx =
        argc > 2
            ? static_cast<std::uint32_t>(std::atoi(argv[2]))
            : std::max<std::uint32_t>(max_context, 1024);
    RunChunkParity(*gpu_model, tokens, tokens_full, chunk_ctx);
    return 0;
  }

  if (verify) {
    RunVerifyExactness(*gpu_model, prompt_tokens, max_context);
    return 0;
  }

  if (rollback) {
    // The default prompt is too short for the 16+8 split; use a long one.
    std::vector<gufo::tokenization::TokenId> rollback_tokens = tokenizer->Encode(
        "The quick brown fox jumps over the lazy dog. The quick brown fox "
        "jumps over the lazy dog. The quick brown fox jumps over the lazy dog.");
    RunRollbackCheck(*gpu_model, rollback_tokens, max_context);
    return 0;
  }

  if (mode == "keycount") {
    RunKeyCountProof(*gpu_model);
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
  // Mirror the GPU's f16 KV twin planes (GUFO_FLINT_KV_F16=1 opts in —
  // default OFF while the serve determinism issue is open): round-once-
  // at-write is part of the computation under test, so the CPU oracle
  // stores the same rounded K/V and the G3 gate keeps its teeth.
  kv.SetF16Rounding([] {
    const char* v = std::getenv("GUFO_FLINT_KV_F16");
    return v != nullptr && v[0] == '1';
  }());
  gufo::models::FlintScratchArena arena(config);
  std::vector<float> cpu_logits(config.vocab_size);

  gufo::hip::FlintGpuArena gpu_arena(config, max_context);
  const auto buffers = gpu_arena.GetBuffers();
  std::vector<float> gpu_logits(config.vocab_size);
  std::cout << "  arena scratch=" << gpu_arena.GetScratchBytes() / (1ULL << 20)
            << " MiB\n";
  const bool profile_decode =
      std::getenv("GUFO_FLINT_DECODE_PROFILE") != nullptr;

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
      gufo::hip::FlintDecodeTimings dt;
      gufo::hip::ExecuteFlintDecodeStep(gpu_model->GetWeights(), token, step,
                                        /*compute_logits=*/true, buffers,
                                        profile_decode ? &dt : nullptr,
                                        gpu_arena.GetGraphState());
      HIP_CHECK(hipStreamSynchronize(buffers.stream));
      const auto t1 = std::chrono::steady_clock::now();
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
      if (profile_decode && step + 1 == steps) {
        std::cout << "\n  decode stages (final step, " << dt.total_ms
                  << " ms): embed=" << dt.embed_ms
                  << " norm+qkv=" << dt.attn_norm_qkv_ms
                  << " head+rope=" << dt.head_rope_ms
                  << " attn=" << dt.attention_ms
                  << " oproj=" << dt.oproj_ms
                  << " ffn-norm+router=" << dt.ffn_norm_router_ms
                  << " ffn=" << dt.moe_ms << " (" << dt.moe_layers << " moe/"
                  << dt.dense_layers << " dense)"
                  << " residual=" << dt.residual_ms
                  << " tail=" << dt.tail_ms;
      }

      HIP_CHECK(hipMemcpy(gpu_logits.data(), buffers.logits,
                          gpu_logits.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      const auto parity = CompareLogits(cpu_logits, gpu_logits);
      // G3 thresholds: under the f16 KV mirror both sides store the same
      // rounded K/V, but the f16 grid noise compounds through the
      // trajectory and forks near-ties (this artifact is near-tie-prone —
      // findings 18/19; the fp32 run already shows step 4 elevated at
      // 3.6e-4, and under f16 step 4's cycle-restart tie forks with
      // dlogp ~5.8 while cos stays >= 0.995 and the token sequence holds).
      // Cosine is the bug-catcher here (a cache addressing bug craters it
      // at step 0, not drift); dlogp only bounds the fork depth. The fp32
      // path keeps the original gates.
      const bool f16_kv = kv.StoresF16();
      const double cos_gate = f16_kv ? 0.995 : 0.999;
      const double logp_gate =
          f16_kv ? 10.0 : Fp32LogpGate(weight_type);
      Expect(parity.cosine >= cos_gate,
             "cosine at step " + std::to_string(step) + " = " +
                 std::to_string(parity.cosine));
      Expect(parity.max_logp_diff <= logp_gate,
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
