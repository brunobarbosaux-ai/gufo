// F-D1/F-D2 device tests: flint per-head sinks and sliding window in the
// baseline decode attention kernels, validated against an independent CPU
// reference at flint shapes (GQA, head_dim 128). Sink semantics follow the
// F-D6 order (scale → sink → softmax → weighted sum): the sink joins the
// running max and the softmax denominator and contributes no value.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

#define HIP_CHECK(call)                                                     \
  do {                                                                      \
    const hipError_t status__ = (call);                                     \
    if (status__ != hipSuccess) {                                           \
      std::cerr << "HIP failure at " << __FILE__ << ":" << __LINE__ << ": " \
                << hipGetErrorString(status__) << "\n";                     \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

namespace {

using gufo::hip::detail::FlintKernelPolicy;

constexpr std::uint32_t kHeads = 4;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kHeadDim = 128;
constexpr std::uint32_t kMaxContext = 1024;
constexpr std::uint32_t kLayer = 3;
constexpr std::uint32_t kWindow = 513;
constexpr std::uint32_t kLayers = 8;

std::size_t Fp32CacheElements() {
  return static_cast<std::size_t>(kLayers) * kKvHeads * kMaxContext * kHeadDim;
}
std::size_t F16CacheElements() { return Fp32CacheElements(); }

// Device fp32 layout: ((layer*kKvHeads + kv)*max_context + p)*kHeadDim + d
std::size_t Fp32Offset(std::uint32_t kv, std::uint32_t p, std::uint32_t d) {
  return ((static_cast<std::size_t>(kLayer) * kKvHeads + kv) * kMaxContext +
          p) *
             kHeadDim +
         d;
}

// Device f16 layout: ((layer*max_context + p)*kKvHeads + kv)*kHeadDim + d
std::size_t F16Offset(std::uint32_t kv, std::uint32_t p, std::uint32_t d) {
  return ((static_cast<std::size_t>(kLayer) * kMaxContext + p) * kKvHeads +
          kv) *
             kHeadDim +
         d;
}

// Logical KV image indexed by (which, kv, p, d); which: 0 = K, 1 = V.
std::size_t LogicalOffset(std::uint32_t which, std::uint32_t kv,
                          std::uint32_t p, std::uint32_t d) {
  return (static_cast<std::size_t>(which) * kKvHeads + kv) *
             static_cast<std::size_t>(kMaxContext) * kHeadDim +
         p * kHeadDim + d;
}

std::vector<float> MakeQ() {
  std::vector<float> q(kHeads * kHeadDim);
  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = 0.6F * std::sin(static_cast<float>(i % 197) * 0.11F) -
           0.2F * std::cos(static_cast<float>(i % 89) * 0.07F);
  }
  return q;
}

// K/V image; positions before the window start carry 1e3-magnitude values
// so any window leak shows up immediately against the reference.
std::vector<float> MakeKv(std::uint32_t window_begin) {
  std::vector<float> kv(2ULL * kKvHeads * kMaxContext * kHeadDim);
  for (std::uint32_t which = 0; which < 2; ++which) {
    for (std::uint32_t kvh = 0; kvh < kKvHeads; ++kvh) {
      for (std::uint32_t p = 0; p < kMaxContext; ++p) {
        const bool leak = p < window_begin;
        for (std::uint32_t d = 0; d < kHeadDim; ++d) {
          const float base =
              0.4F * std::sin(static_cast<float>((kvh * 31 + p) % 251) *
                              0.13F) +
              0.1F * std::cos(static_cast<float>(d % 61) * 0.21F) +
              0.05F * static_cast<float>(which);
          kv[LogicalOffset(which, kvh, p, d)] = leak ? 1.0e3F + base : base;
        }
      }
    }
  }
  return kv;
}

// Per-head sinks spanning the observed flint range [-0.15, +10].
std::vector<float> MakeSinks() { return {0.25F, 6.5F, -0.15F, 10.0F}; }

float HalfRound(float x) { return __half2float(__float2half(x)); }

// Independent reference; `round_half` mirrors the f16 cache rounding.
void Reference(const std::vector<float>& q, const std::vector<float>& kv,
               bool round_half, const float* sinks, std::uint32_t window,
               std::uint32_t pos, std::vector<float>& out) {
  out.assign(q.size(), 0.0F);
  const std::size_t seq = static_cast<std::size_t>(pos) + 1;
  const std::size_t begin = (window != 0 && seq > window) ? seq - window : 0;
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  const std::size_t head_stride = q.size() / kHeads;
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    const std::uint32_t kvh = h / (kHeads / kKvHeads);
    std::vector<float> scores(seq - begin);
    float max_score = -std::numeric_limits<float>::infinity();
    if (sinks != nullptr) {
      max_score = sinks[h];
    }
    for (std::size_t p = begin; p < seq; ++p) {
      float dot = 0.0F;
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        const float k = kv[LogicalOffset(0, kvh, p, d)];
        dot += q[h * head_stride + d] * (round_half ? HalfRound(k) : k);
      }
      scores[p - begin] = dot * scale;
      max_score = std::max(max_score, scores[p - begin]);
    }
    float sum = 0.0F;
    for (float& s : scores) {
      s = std::exp(s - max_score);
      sum += s;
    }
    if (sinks != nullptr) {
      sum += std::exp(sinks[h] - max_score);
    }
    for (std::size_t p = begin; p < seq; ++p) {
      const float w = scores[p - begin] / sum;
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        const float v = kv[LogicalOffset(1, kvh, p, d)];
        out[h * head_stride + d] += w * (round_half ? HalfRound(v) : v);
      }
    }
  }
}

struct Cache {
  float* d_k32 = nullptr;
  float* d_v32 = nullptr;
  __half* d_k16 = nullptr;
  __half* d_v16 = nullptr;
};

Cache UploadCaches(const std::vector<float>& kv) {
  Cache cache;
  std::vector<float> image32(Fp32CacheElements(), 0.0F);
  std::vector<__half> image16(F16CacheElements(), __float2half(0.0F));
  for (std::uint32_t kvh = 0; kvh < kKvHeads; ++kvh) {
    for (std::uint32_t p = 0; p < kMaxContext; ++p) {
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        const float k = kv[LogicalOffset(0, kvh, p, d)];
        const float v = kv[LogicalOffset(1, kvh, p, d)];
        image32[Fp32Offset(kvh, p, d)] = k;
        image16[F16Offset(kvh, p, d)] = __float2half(k);
        // V uses a second cache allocation with the same layout.
        (void)v;
      }
    }
  }
  std::vector<float> v_image32(Fp32CacheElements(), 0.0F);
  std::vector<__half> v_image16(F16CacheElements(), __float2half(0.0F));
  for (std::uint32_t kvh = 0; kvh < kKvHeads; ++kvh) {
    for (std::uint32_t p = 0; p < kMaxContext; ++p) {
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        const float v = kv[LogicalOffset(1, kvh, p, d)];
        v_image32[Fp32Offset(kvh, p, d)] = v;
        v_image16[F16Offset(kvh, p, d)] = __float2half(v);
      }
    }
  }
  const std::size_t bytes32 = Fp32CacheElements() * sizeof(float);
  const std::size_t bytes16 = F16CacheElements() * sizeof(__half);
  HIP_CHECK(hipMalloc(&cache.d_k32, bytes32));
  HIP_CHECK(hipMalloc(&cache.d_v32, bytes32));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&cache.d_k16), bytes16));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&cache.d_v16), bytes16));
  HIP_CHECK(
      hipMemcpy(cache.d_k32, image32.data(), bytes32, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(cache.d_v32, v_image32.data(), bytes32, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(cache.d_k16, image16.data(), bytes16, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(cache.d_v16, v_image16.data(), bytes16, hipMemcpyHostToDevice));
  return cache;
}

void FreeCache(Cache& cache) {
  hipFree(cache.d_k32);
  hipFree(cache.d_v32);
  hipFree(cache.d_k16);
  hipFree(cache.d_v16);
  cache = Cache{};
}

void CheckClose(const std::vector<float>& expected,
                const std::vector<float>& actual, float tolerance,
                const std::string& what) {
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const float delta = std::abs(expected[i] - actual[i]);
    if (delta > tolerance) {
      std::cerr << what << ": mismatch at " << i << ", expected "
                << expected[i] << ", got " << actual[i] << "\n";
      std::exit(1);
    }
  }
}

// One decode step through LaunchAttention with skip_kv_write.
std::vector<float> RunDecode(const std::vector<float>& q,
                             const float* host_sinks, std::uint32_t window,
                             std::uint32_t pos, const Cache& cache,
                             bool use_f16) {
  float* d_q = nullptr;
  float* d_out = nullptr;
  float* d_sinks = nullptr;
  HIP_CHECK(hipMalloc(&d_q, q.size() * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_out, q.size() * sizeof(float)));
  FlintKernelPolicy policy{nullptr, window};
  if (host_sinks != nullptr) {
    HIP_CHECK(hipMalloc(&d_sinks, kHeads * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_sinks, host_sinks, kHeads * sizeof(float),
                        hipMemcpyHostToDevice));
    policy.sinks = d_sinks;
  }
  gufo::hip::LaunchAttention(
      d_q, nullptr, nullptr, nullptr,
      use_f16 ? nullptr : cache.d_k32, use_f16 ? nullptr : cache.d_v32,
      use_f16 ? static_cast<void*>(cache.d_k16) : nullptr,
      use_f16 ? static_cast<void*>(cache.d_v16) : nullptr, d_out, kLayer, pos,
      kMaxContext, kHeads, kKvHeads, kHeadDim, nullptr, nullptr, true,
      &policy);
  std::vector<float> out(q.size());
  HIP_CHECK(hipMemcpy(out.data(), d_out, q.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  hipFree(d_q);
  hipFree(d_out);
  hipFree(d_sinks);
  return out;
}

void TestSinksAndWindow() {
  const auto q = MakeQ();
  const auto sinks = MakeSinks();

  struct Case {
    std::uint32_t pos;
    std::uint32_t window;
    bool use_sinks;
    const char* what;
  };
  const Case cases[] = {
      {0, kWindow, true, "pos 0 sink-dominated"},
      {600, kWindow, true, "pos 600 window 513 (begin 88)"},
      {512, kWindow, true, "pos 512 seq == window (begin 0)"},
      {513, kWindow, true, "pos 513 window edge (begin 1)"},
      {700, 0, true, "pos 700 no window with sinks"},
      {600, kWindow, false, "pos 600 window without sinks"},
      {17, 0, false, "pos 17 plain softmax"},
  };

  for (const auto& test_case : cases) {
    const std::size_t seq = test_case.pos + 1;
    const std::size_t begin =
        (test_case.window != 0 && seq > test_case.window)
            ? seq - test_case.window
            : 0;
    const auto kv = MakeKv(static_cast<std::uint32_t>(begin));
    const auto cache = UploadCaches(kv);
    const float* sink_data = test_case.use_sinks ? sinks.data() : nullptr;

    std::vector<float> expected;
    Reference(q, kv, false, sink_data, test_case.window, test_case.pos,
              expected);
    const auto actual = RunDecode(q, sink_data, test_case.window, test_case.pos,
                                  cache, false);
    CheckClose(expected, actual, 2.0e-4F, test_case.what);

    std::vector<float> expected_half;
    Reference(q, kv, true, sink_data, test_case.window, test_case.pos,
              expected_half);
    const auto actual_half = RunDecode(q, sink_data, test_case.window,
                                       test_case.pos, cache, true);
    CheckClose(expected_half, actual_half, 4.0e-2F,
               std::string(test_case.what) + " [f16 cache]");

    FreeCache(const_cast<Cache&>(cache));
  }
  std::cout << "  sinks/window decode cases ok\n";
}

// LaunchCausalDecodeAttention with a policy must equal the per-row reference
// (the policy gates split-K and online out).
void TestCausalDecodeBatchWithPolicy() {
  const auto q = MakeQ();
  const auto sinks = MakeSinks();
  constexpr std::uint32_t kBatch = 3;
  const std::uint32_t start_pos = 600;
  const std::size_t seq = start_pos + kBatch;
  const std::size_t begin = seq > kWindow ? seq - kWindow : 0;
  const auto kv = MakeKv(static_cast<std::uint32_t>(begin));
  auto cache = UploadCaches(kv);

  std::vector<float> q_rows(kBatch * kHeads * kHeadDim);
  for (std::uint32_t row = 0; row < kBatch; ++row) {
    for (std::size_t i = 0; i < q.size(); ++i) {
      q_rows[row * q.size() + i] = q[i] * (1.0F + 0.1F * row);
    }
  }

  float* d_q = nullptr;
  float* d_out = nullptr;
  float* d_sinks = nullptr;
  HIP_CHECK(hipMalloc(&d_q, q_rows.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_q, q_rows.data(), q_rows.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_out, q_rows.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_sinks, kHeads * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_sinks, sinks.data(), kHeads * sizeof(float),
                      hipMemcpyHostToDevice));

  FlintKernelPolicy policy{d_sinks, kWindow};
  gufo::hip::LaunchCausalDecodeAttention(
      d_q, nullptr, cache.d_k32, cache.d_v32, nullptr, nullptr, d_out, kLayer,
      start_pos, kBatch, kMaxContext, kHeads, kKvHeads, kHeadDim, nullptr, {},
      &policy);

  std::vector<float> out(q_rows.size());
  HIP_CHECK(hipMemcpy(out.data(), d_out, q_rows.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::uint32_t row = 0; row < kBatch; ++row) {
    const std::vector<float> row_q(q_rows.begin() + row * q.size(),
                                   q_rows.begin() + (row + 1) * q.size());
    std::vector<float> expected;
    Reference(row_q, kv, false, sinks.data(), kWindow, start_pos + row,
              expected);
    const std::vector<float> actual(out.begin() + row * q.size(),
                                    out.begin() + (row + 1) * q.size());
    CheckClose(expected, actual, 2.0e-4F, "causal batch row");
  }

  hipFree(d_q);
  hipFree(d_out);
  hipFree(d_sinks);
  FreeCache(cache);
  std::cout << "  causal decode batch with policy ok\n";
}

// Batched prefill with a policy: every token attends causally to its own
// window-bounded prefix with the per-head sink in the denominator.
void TestBatchedPrefillSinksWindow() {
  const auto q_full = MakeQ();
  const auto sinks = MakeSinks();
  constexpr std::uint32_t kBatch = 5;
  const std::uint32_t start_pos = 500;
  const std::size_t seq_end = start_pos + kBatch;
  const std::size_t begin = seq_end > kWindow ? seq_end - kWindow : 0;
  const auto kv = MakeKv(static_cast<std::uint32_t>(begin));
  auto cache = UploadCaches(kv);

  // One prompt chunk: every token row uses the same q pattern scaled by
  // its index so rows differ.
  std::vector<float> q_rows(kBatch * kHeads * kHeadDim);
  for (std::uint32_t t = 0; t < kBatch; ++t) {
    for (std::size_t i = 0; i < q_full.size(); ++i) {
      q_rows[t * q_full.size() + i] = q_full[i] * (1.0F + 0.05F * t);
    }
  }

  float* d_q = nullptr;
  float* d_out = nullptr;
  float* d_sinks = nullptr;
  HIP_CHECK(hipMalloc(&d_q, q_rows.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_q, q_rows.data(), q_rows.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_out, q_rows.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_sinks, kHeads * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_sinks, sinks.data(), kHeads * sizeof(float),
                      hipMemcpyHostToDevice));

  FlintKernelPolicy policy{d_sinks, kWindow};
  // fp32 cache
  gufo::hip::LaunchBatchedAttention(
      d_q, nullptr, nullptr, nullptr, cache.d_k32, cache.d_v32, nullptr,
      nullptr, d_out, kLayer, start_pos, kBatch, kMaxContext, kHeads, kKvHeads,
      kHeadDim, nullptr, true, nullptr, &policy);
  std::vector<float> out(q_rows.size());
  HIP_CHECK(hipMemcpy(out.data(), d_out, q_rows.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  for (std::uint32_t t = 0; t < kBatch; ++t) {
    const std::vector<float> row_q(q_rows.begin() + t * q_full.size(),
                                   q_rows.begin() + (t + 1) * q_full.size());
    std::vector<float> expected;
    Reference(row_q, kv, false, sinks.data(), kWindow, start_pos + t,
              expected);
    const std::vector<float> actual(out.begin() + t * q_full.size(),
                                    out.begin() + (t + 1) * q_full.size());
    CheckClose(expected, actual, 2.0e-4F, "batched prefill row");
  }

  // f16 cache
  gufo::hip::LaunchBatchedAttention(
      d_q, nullptr, nullptr, nullptr, nullptr, nullptr, cache.d_k16,
      cache.d_v16, d_out, kLayer, start_pos, kBatch, kMaxContext, kHeads,
      kKvHeads, kHeadDim, nullptr, true, nullptr, &policy);
  HIP_CHECK(hipMemcpy(out.data(), d_out, q_rows.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  for (std::uint32_t t = 0; t < kBatch; ++t) {
    const std::vector<float> row_q(q_rows.begin() + t * q_full.size(),
                                   q_rows.begin() + (t + 1) * q_full.size());
    std::vector<float> expected;
    Reference(row_q, kv, true, sinks.data(), kWindow, start_pos + t, expected);
    const std::vector<float> actual(out.begin() + t * q_full.size(),
                                    out.begin() + (t + 1) * q_full.size());
    CheckClose(expected, actual, 4.0e-2F, "batched prefill row [f16 cache]");
  }

  hipFree(d_q);
  hipFree(d_out);
  hipFree(d_sinks);
  FreeCache(cache);
  std::cout << "  batched prefill sinks/window ok\n";
}

// Split-KV GQA-sharing decode (LaunchFlintDecodeSplitKvAttention) at the flint
// geometry (32 query / 4 KV heads, head_dim 128) over long spans. One layer:
// the fp32 cache is head-major and the f16 twin token-major, as FlintKvOffset
// lays them out with layer_base 0 and rows 0.
constexpr std::uint32_t kSplitHeads = 32;
constexpr std::uint32_t kSplitKvHeads = 4;
constexpr std::uint32_t kSplitHeadDim = 128;
constexpr std::uint32_t kSplitMaxContext = 32768;

std::vector<float> MakeSplitQ() {
  std::vector<float> q(kSplitHeads * kSplitHeadDim);
  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = 0.6F * std::sin(static_cast<float>(i % 197) * 0.11F) -
           0.2F * std::cos(static_cast<float>(i % 89) * 0.07F);
  }
  return q;
}

// Head-major fp32 image [kv][p][d]. Pseudo-random phases keep scores varied
// across all positions so the softmax weights are not near-uniform.
std::vector<float> MakeSplitKv(std::uint32_t which) {
  std::vector<float> image(static_cast<std::size_t>(kSplitKvHeads) *
                           kSplitMaxContext * kSplitHeadDim);
  for (std::uint32_t kvh = 0; kvh < kSplitKvHeads; ++kvh) {
    for (std::uint32_t p = 0; p < kSplitMaxContext; ++p) {
      for (std::uint32_t d = 0; d < kSplitHeadDim; ++d) {
        const std::uint32_t mixed =
            p * 2654435761U + d * 40503U + kvh * 97U + which * 1013U;
        const float phase = static_cast<float>(mixed % 1024U) / 1024.0F;
        image[((static_cast<std::size_t>(kvh) * kSplitMaxContext) + p) *
                  kSplitHeadDim +
              d] = 3.0F * std::sin(6.2831853F * phase);
      }
    }
  }
  return image;
}

std::vector<float> RoundedCopy(const std::vector<float>& image) {
  std::vector<float> out(image.size());
  for (std::size_t i = 0; i < image.size(); ++i) {
    out[i] = HalfRound(image[i]);
  }
  return out;
}

// CPU reference over the head-major image; `k_src`/`v_src` are either the raw
// image or its f16-rounded copy.
void SplitReference(const std::vector<float>& q, const std::vector<float>& k_src,
                    const std::vector<float>& v_src, const float* sinks,
                    std::uint32_t window, std::uint32_t pos,
                    std::vector<float>& out) {
  out.assign(q.size(), 0.0F);
  const std::size_t seq = static_cast<std::size_t>(pos) + 1;
  const std::size_t begin = (window != 0 && seq > window) ? seq - window : 0;
  const float scale = 1.0F / std::sqrt(static_cast<float>(kSplitHeadDim));
  const std::size_t group = kSplitHeads / kSplitKvHeads;
  for (std::uint32_t h = 0; h < kSplitHeads; ++h) {
    const std::uint32_t kvh = h / group;
    const float* k_base =
        k_src.data() +
        static_cast<std::size_t>(kvh) * kSplitMaxContext * kSplitHeadDim;
    const float* v_base =
        v_src.data() +
        static_cast<std::size_t>(kvh) * kSplitMaxContext * kSplitHeadDim;
    const float* q_h = q.data() + static_cast<std::size_t>(h) * kSplitHeadDim;
    std::vector<float> scores(seq - begin);
    float max_score = sinks != nullptr ? sinks[h]
                                       : -std::numeric_limits<float>::infinity();
    for (std::size_t p = begin; p < seq; ++p) {
      const float* k_p = k_base + p * kSplitHeadDim;
      float dot = 0.0F;
      for (std::uint32_t d = 0; d < kSplitHeadDim; ++d) {
        dot += q_h[d] * k_p[d];
      }
      scores[p - begin] = dot * scale;
      max_score = std::max(max_score, scores[p - begin]);
    }
    float sum = 0.0F;
    for (float& s : scores) {
      s = std::exp(s - max_score);
      sum += s;
    }
    if (sinks != nullptr) {
      sum += std::exp(sinks[h] - max_score);
    }
    float* out_h = out.data() + static_cast<std::size_t>(h) * kSplitHeadDim;
    for (std::size_t p = begin; p < seq; ++p) {
      const float w = scores[p - begin] / sum;
      const float* v_p = v_base + p * kSplitHeadDim;
      for (std::uint32_t d = 0; d < kSplitHeadDim; ++d) {
        out_h[d] += w * v_p[d];
      }
    }
  }
}

void TestSplitKvDecode() {
  const auto q = MakeSplitQ();
  const auto k_img = MakeSplitKv(0);
  const auto v_img = MakeSplitKv(1);
  const auto k_img16 = RoundedCopy(k_img);
  const auto v_img16 = RoundedCopy(v_img);
  std::vector<float> sinks(kSplitHeads);
  for (std::uint32_t h = 0; h < kSplitHeads; ++h) {
    sinks[h] = -0.15F + 0.3F * static_cast<float>(h);
  }

  // f16 twin: token-major ((p * kv_heads + kvh) * head_dim + d).
  const std::size_t elems = k_img.size();
  std::vector<__half> k16(elems), v16(elems);
  for (std::uint32_t kvh = 0; kvh < kSplitKvHeads; ++kvh) {
    for (std::uint32_t p = 0; p < kSplitMaxContext; ++p) {
      for (std::uint32_t d = 0; d < kSplitHeadDim; ++d) {
        const std::size_t src =
            ((static_cast<std::size_t>(kvh) * kSplitMaxContext) + p) *
                kSplitHeadDim +
            d;
        const std::size_t dst =
            ((static_cast<std::size_t>(p) * kSplitKvHeads) + kvh) *
                kSplitHeadDim +
            d;
        k16[dst] = __float2half(k_img[src]);
        v16[dst] = __float2half(v_img[src]);
      }
    }
  }

  float* d_q = nullptr;
  float* d_out = nullptr;
  float* d_sinks = nullptr;
  float* d_scratch = nullptr;
  float* d_k32 = nullptr;
  float* d_v32 = nullptr;
  __half* d_k16 = nullptr;
  __half* d_v16 = nullptr;
  std::uint32_t* d_pos = nullptr;
  const std::size_t scratch_elems = gufo::hip::FlintDecodeSplitKvScratchElements(
      kSplitMaxContext, kSplitHeads, kSplitHeadDim);
  HIP_CHECK(hipMalloc(&d_q, q.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, q.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_sinks, kSplitHeads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_scratch, scratch_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k32, elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v32, elems * sizeof(float)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_k16), elems * sizeof(__half)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_v16), elems * sizeof(__half)));
  HIP_CHECK(hipMalloc(&d_pos, sizeof(std::uint32_t)));
  HIP_CHECK(hipMemcpy(d_q, q.data(), q.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_sinks, sinks.data(), kSplitHeads * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k32, k_img.data(), elems * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v32, v_img.data(), elems * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k16, k16.data(), elems * sizeof(__half),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v16, v16.data(), elems * sizeof(__half),
                      hipMemcpyHostToDevice));

  constexpr std::uint32_t kSpans[] = {1, 7, 256, 257, 4096, 32768};
  constexpr std::uint32_t kWindows[] = {0, kWindow};
  std::vector<float> out(q.size());
  std::vector<float> expected;
  for (const std::uint32_t span : kSpans) {
    const std::uint32_t pos = span - 1;
    HIP_CHECK(hipMemcpy(d_pos, &pos, sizeof(pos), hipMemcpyHostToDevice));
    for (const std::uint32_t window : kWindows) {
      for (const bool use_sinks : {false, true}) {
        const float* sink_data = use_sinks ? sinks.data() : nullptr;
        FlintKernelPolicy policy{use_sinks ? d_sinks : nullptr, window, 0};
        const std::string label = "split-kv span " + std::to_string(span) +
                                  " window " + std::to_string(window) +
                                  (use_sinks ? " sinks" : " no-sinks");

        SplitReference(q, k_img, v_img, sink_data, window, pos, expected);
        HIP_CHECK(hipMemset(d_out, 0, q.size() * sizeof(float)));
        if (!gufo::hip::LaunchFlintDecodeSplitKvAttention(
                d_q, d_k32, d_v32, nullptr, nullptr, d_out, 0, 0, d_pos,
                kSplitMaxContext, kSplitHeads, kSplitKvHeads, kSplitHeadDim,
                d_scratch, nullptr, &policy)) {
          std::cerr << label << ": launch rejected\n";
          std::exit(1);
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(out.data(), d_out, q.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        CheckClose(expected, out, 1.0e-4F, label);

        SplitReference(q, k_img16, v_img16, sink_data, window, pos, expected);
        HIP_CHECK(hipMemset(d_out, 0, q.size() * sizeof(float)));
        if (!gufo::hip::LaunchFlintDecodeSplitKvAttention(
                d_q, nullptr, nullptr, d_k16, d_v16, d_out, 0, 0, d_pos,
                kSplitMaxContext, kSplitHeads, kSplitKvHeads, kSplitHeadDim,
                d_scratch, nullptr, &policy)) {
          std::cerr << label << ": f16 launch rejected\n";
          std::exit(1);
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(out.data(), d_out, q.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        CheckClose(expected, out, 4.0e-2F, label + " [f16 cache]");
      }
    }
  }

  hipFree(d_q);
  hipFree(d_out);
  hipFree(d_sinks);
  hipFree(d_scratch);
  hipFree(d_k32);
  hipFree(d_v32);
  hipFree(d_k16);
  hipFree(d_v16);
  hipFree(d_pos);
  std::cout << "  split-kv gqa decode cases ok\n";
}

}  // namespace
#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Flint decode ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }
  TestSinksAndWindow();
  TestCausalDecodeBatchWithPolicy();
  TestBatchedPrefillSinksWindow();
  TestSplitKvDecode();
  std::cout << "Flint decode ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping flint decode ops test.\n";
  return 77;
#endif
}
