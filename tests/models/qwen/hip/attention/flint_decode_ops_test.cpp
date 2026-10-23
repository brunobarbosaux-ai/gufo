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
  std::cout << "Flint decode ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping flint decode ops test.\n";
  return 77;
#endif
}
