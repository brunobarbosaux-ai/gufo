// Flint GPU decode step vs the validated CPU reference (F-D dispatch entry).
// Builds a synthetic 3-layer flint model (dense lead layer + MoE layers,
// windowed + global, rope + NoPE, f32 weights with a BF16 router), runs the
// same three tokens through FlintForwardModel (CPU) and
// ExecuteFlintDecodeStep (GPU), and compares logits step by step. The CPU
// reference is the D4 oracle: it matched the llama.cpp ladder on the real
// artifact, so agreement here validates the GPU orchestration (sinks,
// window, NoPE gating, MoE selection, shared expert) end to end.

#include "src/models/qwen/hip/detail/flint_decode_step.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/flint_forward.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/state.hpp"

namespace {

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

// Deterministic pseudo-random values in [-0.3, 0.3).
struct Lcg {
  std::uint64_t state = 0x9E3779B97F4A7C15ULL;
  float Next() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    const float unit =
        static_cast<float>((state >> 11) & 0xFFFFFF) /
        static_cast<float>(0xFFFFFF);
    return (unit - 0.5F) * 0.6F;
  }
};

constexpr std::uint32_t kLayers = 3;
constexpr std::uint32_t kHidden = 64;
constexpr std::uint32_t kHeads = 4;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kHeadDim = 16;
constexpr std::uint32_t kVocab = 16;
constexpr std::uint32_t kIntermediate = 128;
constexpr std::uint32_t kExperts = 8;
constexpr std::uint32_t kUsed = 2;
constexpr std::uint32_t kExpertFf = 32;
constexpr std::uint32_t kSharedFf = 64;
constexpr std::uint32_t kMaxContext = 64;

// Owns every host weight buffer so refs stay valid for both runtimes.
class HostStore {
 public:
  gufo::models::QwenTensorRef Make(const void* data, std::size_t elements,
                                   gufo::core::GgmlType type) {
    auto& blob = slots_.emplace_back();
    const std::size_t bytes =
        gufo::quant::EncodedSizeBytes(type, elements);
    blob.assign(bytes, 0);
    std::memcpy(blob.data(), data, bytes);
    gufo::models::QwenTensorRef ref;
    ref.data = blob.data();
    ref.type = type;
    ref.num_elements = elements;
    return ref;
  }

  gufo::models::QwenTensorRef MakeRandom(
      std::size_t elements, gufo::core::GgmlType type, Lcg& rng,
      float scale = 1.0F) {
    std::vector<float> host(elements);
    for (auto& v : host) v = rng.Next() * scale;
    if (type == gufo::core::GgmlType::kBF16) {
      std::vector<std::uint16_t> bf16(elements);
      for (std::size_t i = 0; i < elements; ++i) {
        const std::uint32_t bits =
            *reinterpret_cast<std::uint32_t*>(&host[i]);
        const std::uint32_t rounded =
            (bits + 0x8000U + ((bits >> 16U) & 1U)) >> 16U;
        bf16[i] = static_cast<std::uint16_t>(rounded);
      }
      return Make(bf16.data(), elements, type);
    }
    return Make(host.data(), elements, type);
  }

  gufo::models::QwenTensorRef MakeOnes(std::size_t elements) {
    std::vector<float> host(elements, 1.0F);
    return Make(host.data(), elements, gufo::core::GgmlType::kF32);
  }

 private:
  std::vector<std::vector<std::uint8_t>> slots_;
};

gufo::models::FlintModelWeights BuildSyntheticWeights(HostStore& store) {
  const auto F32 = gufo::core::GgmlType::kF32;
  const auto BF16 = gufo::core::GgmlType::kBF16;
  Lcg rng;

  gufo::models::FlintModelWeights w;
  auto& config = w.config;
  config.num_layers = kLayers;
  config.hidden_size = kHidden;
  config.intermediate_size = kIntermediate;
  config.num_attention_heads = kHeads;
  config.num_key_value_heads = kKvHeads;
  config.head_dim = kHeadDim;
  config.vocab_size = kVocab;
  config.context_length = kMaxContext;
  config.rotary_dim = kHeadDim;
  config.rope_theta = 10000.0F;
  config.expert_count = kExperts;
  config.expert_used_count = kUsed;
  config.expert_feed_forward_length = kExpertFf;
  config.expert_shared_count = 1;
  config.expert_shared_feed_forward_length = kSharedFf;
  config.expert_gating_func = 2;
  config.expert_weights_scale = 1.0F;
  config.expert_weights_norm = true;
  config.leading_dense_block_count = 1;
  config.sliding_window = 8;
  config.sliding_window_pattern = {false, true, true};
  config.rope_pattern = {false, true, true};

  w.token_embd = store.MakeRandom(kVocab * kHidden, F32, rng);
  w.output_norm = store.MakeOnes(kHidden);
  w.output = store.MakeRandom(kVocab * kHidden, F32, rng);

  w.layers.resize(kLayers);
  for (std::uint32_t l = 0; l < kLayers; ++l) {
    auto& layer = w.layers[l];
    layer.attrs.windowed = config.sliding_window_pattern[l];
    layer.attrs.rope = config.rope_pattern[l];
    layer.attrs.dense_ffn = l < config.leading_dense_block_count;

    layer.attn_norm = store.MakeOnes(kHidden);
    layer.attn_q = store.MakeRandom(kHeads * kHeadDim * kHidden, F32, rng);
    layer.attn_k = store.MakeRandom(kKvHeads * kHeadDim * kHidden, F32, rng);
    layer.attn_v = store.MakeRandom(kKvHeads * kHeadDim * kHidden, F32, rng);
    layer.attn_output =
        store.MakeRandom(kHidden * kHeads * kHeadDim, F32, rng);
    layer.attn_q_norm = store.MakeOnes(kHeadDim);
    layer.attn_k_norm = store.MakeOnes(kHeadDim);
    // flint sinks are large positive logits; vary them per head.
    std::vector<float> sinks(kHeads);
    for (auto& v : sinks) v = 1.0F + std::abs(rng.Next()) * 6.0F;
    layer.attn_sinks = store.Make(sinks.data(), kHeads, F32);
    layer.ffn_norm = store.MakeOnes(kHidden);

    if (layer.attrs.dense_ffn) {
      layer.ffn_gate = store.MakeRandom(kIntermediate * kHidden, F32, rng);
      layer.ffn_up = store.MakeRandom(kIntermediate * kHidden, F32, rng);
      layer.ffn_down = store.MakeRandom(kHidden * kIntermediate, F32, rng);
    } else {
      layer.ffn_gate_inp = store.MakeRandom(kExperts * kHidden, BF16, rng);
      layer.ffn_gate_exps =
          store.MakeRandom(kExperts * kExpertFf * kHidden, F32, rng);
      layer.ffn_up_exps =
          store.MakeRandom(kExperts * kExpertFf * kHidden, F32, rng);
      layer.ffn_down_exps =
          store.MakeRandom(kExperts * kHidden * kExpertFf, F32, rng);
      layer.ffn_gate_shexp = store.MakeRandom(kSharedFf * kHidden, F32, rng);
      layer.ffn_up_shexp = store.MakeRandom(kSharedFf * kHidden, F32, rng);
      layer.ffn_down_shexp = store.MakeRandom(kHidden * kSharedFf, F32, rng);
    }
  }
  return w;
}

// Deep-copies one tensor ref to device memory and returns a patched ref.
gufo::models::QwenTensorRef ToDevice(const gufo::models::QwenTensorRef& ref,
                                     std::vector<void*>& allocs) {
  if (ref.empty()) return {};
  const std::size_t bytes =
      gufo::quant::EncodedSizeBytes(ref.type, ref.num_elements);
  void* d = nullptr;
  HIP_CHECK(hipMalloc(&d, bytes));
  HIP_CHECK(hipMemcpy(d, ref.data, bytes, hipMemcpyHostToDevice));
  allocs.push_back(d);
  auto out = ref;
  out.data = d;
  out.available_bytes = bytes;
  return out;
}

gufo::models::FlintModelWeights ToDeviceModel(
    const gufo::models::FlintModelWeights& src, std::vector<void*>& allocs) {
  auto dst = src;
  dst.token_embd = ToDevice(src.token_embd, allocs);
  dst.output_norm = ToDevice(src.output_norm, allocs);
  dst.output = ToDevice(src.output, allocs);
  for (std::size_t l = 0; l < src.layers.size(); ++l) {
    auto& d = dst.layers[l];
    const auto& s = src.layers[l];
    d.attn_norm = ToDevice(s.attn_norm, allocs);
    d.attn_q = ToDevice(s.attn_q, allocs);
    d.attn_k = ToDevice(s.attn_k, allocs);
    d.attn_v = ToDevice(s.attn_v, allocs);
    d.attn_output = ToDevice(s.attn_output, allocs);
    d.attn_q_norm = ToDevice(s.attn_q_norm, allocs);
    d.attn_k_norm = ToDevice(s.attn_k_norm, allocs);
    d.attn_sinks = ToDevice(s.attn_sinks, allocs);
    d.ffn_norm = ToDevice(s.ffn_norm, allocs);
    d.ffn_gate = ToDevice(s.ffn_gate, allocs);
    d.ffn_up = ToDevice(s.ffn_up, allocs);
    d.ffn_down = ToDevice(s.ffn_down, allocs);
    d.ffn_gate_inp = ToDevice(s.ffn_gate_inp, allocs);
    d.ffn_gate_exps = ToDevice(s.ffn_gate_exps, allocs);
    d.ffn_up_exps = ToDevice(s.ffn_up_exps, allocs);
    d.ffn_down_exps = ToDevice(s.ffn_down_exps, allocs);
    d.ffn_gate_shexp = ToDevice(s.ffn_gate_shexp, allocs);
    d.ffn_up_shexp = ToDevice(s.ffn_up_shexp, allocs);
    d.ffn_down_shexp = ToDevice(s.ffn_down_shexp, allocs);
  }
  return dst;
}

float* AllocDevice(std::size_t floats, std::vector<void*>& allocs) {
  void* d = nullptr;
  HIP_CHECK(hipMalloc(&d, floats * sizeof(float)));
  HIP_CHECK(hipMemset(d, 0, floats * sizeof(float)));
  allocs.push_back(d);
  return static_cast<float*>(d);
}

void CompareLogits(const std::vector<float>& cpu,
                   const std::vector<float>& gpu, std::uint32_t pos) {
  double dot = 0.0, norm_a = 0.0, norm_b = 0.0, max_abs = 0.0, scale = 0.0;
  std::size_t argmax_cpu = 0, argmax_gpu = 0;
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    dot += static_cast<double>(cpu[i]) * gpu[i];
    norm_a += static_cast<double>(cpu[i]) * cpu[i];
    norm_b += static_cast<double>(gpu[i]) * gpu[i];
    max_abs = std::max(max_abs, std::abs(static_cast<double>(cpu[i]) -
                                         static_cast<double>(gpu[i])));
    scale = std::max(scale, std::abs(static_cast<double>(cpu[i])));
    if (cpu[i] > cpu[argmax_cpu]) argmax_cpu = i;
    if (gpu[i] > gpu[argmax_gpu]) argmax_gpu = i;
  }
  const double cosine = dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-30);
  Expect(cosine > 0.9999,
         "cosine at pos " + std::to_string(pos) + " = " +
             std::to_string(cosine));
  Expect(max_abs <= 0.02 * std::max(1.0, scale),
         "max abs diff at pos " + std::to_string(pos) + " = " +
             std::to_string(max_abs) + " (scale " + std::to_string(scale) + ")");
  Expect(argmax_cpu == argmax_gpu,
         "argmax mismatch at pos " + std::to_string(pos) + ": cpu " +
             std::to_string(argmax_cpu) + " gpu " + std::to_string(argmax_gpu));
}

}  // namespace

// Op-by-op layer-0 probe (debug): each GPU launcher vs a hand CPU compute.
void ProbeLayer0Ops(const gufo::models::FlintModelWeights& weights,
                    std::vector<void*>& allocs);

namespace {

void ReportDiff(const char* what, const std::vector<float>& gpu,
                const std::vector<float>& cpu) {
  double max_abs = 0.0, scale = 0.0;
  for (std::size_t i = 0; i < gpu.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(static_cast<double>(gpu[i]) -
                                         static_cast<double>(cpu[i])));
    scale = std::max(scale, std::abs(static_cast<double>(cpu[i])));
  }
  std::cout << what << ": max_abs_diff=" << max_abs << " scale=" << scale
            << (max_abs < 1e-3 * std::max(1.0, scale) ? "  OK" : "  MISMATCH")
            << "\n";
}

}  // namespace

void ProbeLayer0Ops(const gufo::models::FlintModelWeights& weights,
                    std::vector<void*>& allocs) {
  using namespace gufo;
  auto dw = ToDeviceModel(weights, allocs);
  const auto& d = dw.layers[0];

  float* hidden = AllocDevice(kHidden, allocs);
  float* normed = AllocDevice(kHidden, allocs);
  float* q = AllocDevice(kHeads * kHeadDim, allocs);
  float* k = AllocDevice(kKvHeads * kHeadDim, allocs);
  float* v = AllocDevice(kKvHeads * kHeadDim, allocs);
  float* attn_out = AllocDevice(kHeads * kHeadDim, allocs);
  float* proj = AllocDevice(kHidden, allocs);
  float* act = AllocDevice(kIntermediate, allocs);
  float* k_cache = AllocDevice(
      static_cast<std::size_t>(kLayers) * kKvHeads * kMaxContext * kHeadDim,
      allocs);
  float* v_cache = AllocDevice(
      static_cast<std::size_t>(kLayers) * kKvHeads * kMaxContext * kHeadDim,
      allocs);
  std::uint32_t* out_token = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&out_token), 4));
  allocs.push_back(out_token);

  const std::uint32_t token = 3;
  const std::uint32_t pos = 0;

  std::vector<float> h(kHidden);
  auto fetch = [&](float* dptr, std::vector<float>& out) {
    HIP_CHECK(hipMemcpy(out.data(), dptr, out.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
  };

  // 1. Embedding.
  hip::LaunchEmbeddingLookup(dw.token_embd.data, dw.token_embd.type, token,
                             hidden, kHidden);
  fetch(hidden, h);
  {
    const float* row =
        static_cast<const float*>(weights.token_embd.data) + token * kHidden;
    std::vector<float> cpu(row, row + kHidden);
    ReportDiff("embedding", h, cpu);
  }

  // 2. RMSNorm (weight = 1).
  hip::LaunchRMSNorm(hidden, static_cast<const float*>(d.attn_norm.data),
                     normed, kHidden, 1e-6F);
  fetch(normed, h);
  {
    double ms = 0.0;
    for (std::uint32_t i = 0; i < kHidden; ++i) ms += double(h[i]) * h[i];
    // Recompute from the *device* hidden we already validated.
    std::vector<float> hid = h;  // h currently holds normed; recompute below
    (void)hid;
    // CPU expected from the embedding row.
    const float* row =
        static_cast<const float*>(weights.token_embd.data) + token * kHidden;
    ms = 0.0;
    for (std::uint32_t i = 0; i < kHidden; ++i)
      ms += double(row[i]) * row[i];
    const float inv = 1.0F / std::sqrt(static_cast<float>(ms / kHidden) + 1e-6F);
    std::vector<float> cpu(kHidden);
    for (std::uint32_t i = 0; i < kHidden; ++i) cpu[i] = row[i] * inv;
    ReportDiff("rmsnorm", h, cpu);
  }

  // 3. Q projection GEMV from the CPU-normed vector (write it to device).
  {
    const float* row =
        static_cast<const float*>(weights.token_embd.data) + token * kHidden;
    double ms = 0.0;
    for (std::uint32_t i = 0; i < kHidden; ++i) ms += double(row[i]) * row[i];
    const float inv = 1.0F / std::sqrt(static_cast<float>(ms / kHidden) + 1e-6F);
    std::vector<float> cpu_normed(kHidden);
    for (std::uint32_t i = 0; i < kHidden; ++i) cpu_normed[i] = row[i] * inv;
    HIP_CHECK(hipMemcpy(normed, cpu_normed.data(), kHidden * sizeof(float),
                        hipMemcpyHostToDevice));

    const float* qw = static_cast<const float*>(weights.layers[0].attn_q.data);
    std::vector<float> cpu_q(kHeads * kHeadDim);
    for (std::size_t r = 0; r < cpu_q.size(); ++r) {
      double acc = 0.0;
      for (std::uint32_t ccol = 0; ccol < kHidden; ++ccol)
        acc += double(qw[r * kHidden + ccol]) * cpu_normed[ccol];
      cpu_q[r] = static_cast<float>(acc);
    }
    hip::LaunchGEMV(d.attn_q.data, d.attn_q.type, normed, q,
                    kHeads * kHeadDim, kHidden);
    fetch(q, h);
    h.resize(kHeads * kHeadDim);
    fetch(q, h);
    ReportDiff("gemv_q", h, cpu_q);

    // 4. Per-head Q norm.
    hip::LaunchPerHeadRMSNorm(q, static_cast<const float*>(d.attn_q_norm.data),
                              q, kHeads, kHeadDim, 1e-6F);
    fetch(q, h);
    {
      std::vector<float> cpu_qn(kHeads * kHeadDim);
      for (std::uint32_t head = 0; head < kHeads; ++head) {
        double ms2 = 0.0;
        for (std::uint32_t e = 0; e < kHeadDim; ++e) {
          const double x = cpu_q[head * kHeadDim + e];
          ms2 += x * x;
        }
        const float inv2 =
            1.0F / std::sqrt(static_cast<float>(ms2 / kHeadDim) + 1e-6F);
        for (std::uint32_t e = 0; e < kHeadDim; ++e)
          cpu_qn[head * kHeadDim + e] =
              cpu_q[head * kHeadDim + e] * inv2;  // q_norm weight = 1
      }
      ReportDiff("perhead_qnorm", h, cpu_qn);
    }
  }

  // 5. Attention at pos 0 with sinks (single key).
  {
    hip::LaunchGEMV(d.attn_k.data, d.attn_k.type, normed, k,
                    kKvHeads * kHeadDim, kHidden);
    hip::LaunchGEMV(d.attn_v.data, d.attn_v.type, normed, v,
                    kKvHeads * kHeadDim, kHidden);
    const hip::detail::FlintKernelPolicy policy{
        static_cast<const float*>(d.attn_sinks.data), 0U};
    hip::LaunchAttention(q, k, v, nullptr, k_cache, v_cache, nullptr, nullptr,
                         attn_out, 0, pos, kMaxContext, kHeads, kKvHeads,
                         kHeadDim, nullptr, nullptr, false, &policy);
    fetch(attn_out, h);
    h.resize(kHeads * kHeadDim);
    fetch(attn_out, h);

    // CPU: out_h = w * v0, w = exp(s)/(exp(s)+exp(sink_h)) with
    // s = q_h . k_kv / sqrt(head_dim) (layer 0: NoPE, global).
    std::vector<float> qh(kHeads * kHeadDim), kh(kKvHeads * kHeadDim),
        vh(kKvHeads * kHeadDim);
    fetch(q, qh);
    fetch(k, kh);
    fetch(v, vh);
    std::vector<float> cpu_attn(kHeads * kHeadDim);
    const float* sinks = static_cast<const float*>(weights.layers[0].attn_sinks.data);
    for (std::uint32_t head = 0; head < kHeads; ++head) {
      const std::uint32_t kv = head / (kHeads / kKvHeads);
      double s = 0.0;
      for (std::uint32_t e = 0; e < kHeadDim; ++e)
        s += double(qh[head * kHeadDim + e]) * kh[kv * kHeadDim + e];
      s /= std::sqrt(static_cast<double>(kHeadDim));
      const double m = std::max(s, static_cast<double>(sinks[head]));
      const double wgt =
        std::exp(s - m) / (std::exp(s - m) + std::exp(sinks[head] - m));
      for (std::uint32_t e = 0; e < kHeadDim; ++e)
        cpu_attn[head * kHeadDim + e] =
            static_cast<float>(wgt * vh[kv * kHeadDim + e]);
    }
    ReportDiff("attention_pos0_sinks", h, cpu_attn);

    // 6. O-projection + residual add. Expected O-proj input is the GPU
    // attention output; fetch it and use the same vector on both sides so
    // this probe isolates the GEMV.
    hip::LaunchGEMV(d.attn_output.data, d.attn_output.type, attn_out, proj,
                    kHidden, kHeads * kHeadDim);
    std::vector<float> cpu_proj(kHidden);
    std::vector<float> attn_gpu(kHeads * kHeadDim);
    fetch(attn_out, attn_gpu);
    {
      const float* ow =
          static_cast<const float*>(weights.layers[0].attn_output.data);
      for (std::uint32_t r = 0; r < kHidden; ++r) {
        double acc = 0.0;
        for (std::uint32_t ccol = 0; ccol < kHeads * kHeadDim; ++ccol)
          acc += double(ow[r * kHeads * kHeadDim + ccol]) *
                 double(attn_gpu[ccol]);
        cpu_proj[r] = static_cast<float>(acc);
      }
      fetch(proj, h);
      ReportDiff("gemv_o_proj", h, cpu_proj);
    }

    // 7. Dense FFN: fused SwiGLU GEMV + down GEMV, from the CPU-normed input.
    {
      std::vector<float> cpu_act(kIntermediate);
      const float* gw =
          static_cast<const float*>(weights.layers[0].ffn_gate.data);
      const float* uw =
          static_cast<const float*>(weights.layers[0].ffn_up.data);
      std::vector<float> normed_cpu(kHidden);
      HIP_CHECK(hipMemcpy(normed_cpu.data(), normed, kHidden * sizeof(float),
                          hipMemcpyDeviceToHost));
      for (std::uint32_t r = 0; r < kIntermediate; ++r) {
        double g = 0.0, u = 0.0;
        for (std::uint32_t ccol = 0; ccol < kHidden; ++ccol) {
          g += double(gw[r * kHidden + ccol]) * normed_cpu[ccol];
          u += double(uw[r * kHidden + ccol]) * normed_cpu[ccol];
        }
        cpu_act[r] = static_cast<float>(g / (1.0 + std::exp(-g)) * u);
      }
      // f32 pairs use the decode step's fallback: two GEMVs + activation.
      float* gate_scratch = AllocDevice(kIntermediate, allocs);
      hip::LaunchGEMV(d.ffn_gate.data, d.ffn_gate.type, normed, gate_scratch,
                      kIntermediate, kHidden);
      hip::LaunchGEMV(d.ffn_up.data, d.ffn_up.type, normed, act,
                      kIntermediate, kHidden);
      hip::LaunchBatchedSwiGLUActivation(gate_scratch, act, act, nullptr,
                                         kIntermediate);
      std::vector<float> act_gpu(kIntermediate);
      fetch(act, act_gpu);
      ReportDiff("swiglu_dense", act_gpu, cpu_act);

      std::vector<float> cpu_down(kHidden);
      const float* dwn =
          static_cast<const float*>(weights.layers[0].ffn_down.data);
      for (std::uint32_t r = 0; r < kHidden; ++r) {
        double acc = 0.0;
        for (std::uint32_t ccol = 0; ccol < kIntermediate; ++ccol)
          acc += double(dwn[r * kIntermediate + ccol]) * double(cpu_act[ccol]);
        cpu_down[r] = static_cast<float>(acc);
      }
      float* down_out = AllocDevice(kHidden, allocs);
      hip::LaunchGEMV(d.ffn_down.data, d.ffn_down.type, act, down_out, kHidden,
                      kIntermediate);
      fetch(down_out, h);
      h.resize(kHidden);
      fetch(down_out, h);
      ReportDiff("gemv_ffn_down", h, cpu_down);
    }
  }

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
}

// DFlash-2 target-layer capture (7.7): the decode step must copy the
// post-layer residual after each tapped layer into the capture buffer, in tap
// order. Compare against the CPU reference's per-layer hidden state.
void TestTargetLayerCapture() {
  HostStore store;
  auto weights = BuildSyntheticWeights(store);
  const std::uint32_t taps[2] = {0, 2};
  const std::uint32_t token = 3;
  const std::uint32_t pos = 0;

  // CPU reference: embedding + every layer, capturing the tapped residuals.
  gufo::models::QwenKvCache cpu_kv(weights.config.num_layers, kKvHeads,
                                   kMaxContext, kHeadDim);
  gufo::models::FlintScratchArena cpu_arena(weights.config);
  gufo::models::ForwardEmbedding(token, weights.token_embd, kHidden,
                                 cpu_arena.hidden);
  std::vector<std::vector<float>> cpu_taps(2, std::vector<float>(kHidden));
  for (std::uint32_t l = 0; l < weights.config.num_layers; ++l) {
    gufo::models::FlintForwardLayer(cpu_arena.hidden, weights.layers[l],
                                    weights.config, cpu_kv, l, pos, cpu_arena);
    for (std::size_t t = 0; t < 2; ++t) {
      if (taps[t] == l) {
        std::copy(cpu_arena.hidden.begin(), cpu_arena.hidden.begin() + kHidden,
                  cpu_taps[t].begin());
      }
    }
  }

  // GPU: same weights, fresh KV cache, capture enabled.
  std::vector<void*> allocs;
  const auto device_weights = ToDeviceModel(weights, allocs);
  gufo::hip::FlintGpuStepBuffers buffers;
  const std::size_t kv_elems = static_cast<std::size_t>(
                                   weights.config.num_layers) *
                               kKvHeads * kMaxContext * kHeadDim;
  buffers.hidden = AllocDevice(kHidden, allocs);
  buffers.normed = AllocDevice(kHidden, allocs);
  buffers.q = AllocDevice(kHeads * kHeadDim, allocs);
  buffers.k = AllocDevice(kKvHeads * kHeadDim, allocs);
  buffers.v = AllocDevice(kKvHeads * kHeadDim, allocs);
  buffers.attn_out = AllocDevice(kHeads * kHeadDim, allocs);
  buffers.attn_proj = AllocDevice(kHidden, allocs);
  buffers.act = AllocDevice(kIntermediate, allocs);
  buffers.act_gate = AllocDevice(kIntermediate, allocs);
  buffers.expert_out = AllocDevice(kHidden, allocs);
  buffers.mlp_out = AllocDevice(kHidden, allocs);
  buffers.router_logits = AllocDevice(kExperts, allocs);
  buffers.logits = AllocDevice(kVocab, allocs);
  buffers.k_cache = AllocDevice(kv_elems, allocs);
  buffers.v_cache = AllocDevice(kv_elems, allocs);
  buffers.max_context = kMaxContext;
  std::vector<std::size_t> kv_layer_base(weights.config.num_layers);
  std::vector<std::uint32_t> kv_layer_rows(weights.config.num_layers,
                                           kMaxContext);
  for (std::uint32_t l = 0; l < weights.config.num_layers; ++l) {
    kv_layer_base[l] = static_cast<std::size_t>(l) * kKvHeads * kMaxContext *
                       kHeadDim;
  }
  buffers.kv_layer_base = kv_layer_base.data();
  buffers.kv_layer_rows = kv_layer_rows.data();
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&buffers.selected),
                      8 * sizeof(std::uint32_t)));
  allocs.push_back(buffers.selected);
  buffers.expert_weights = AllocDevice(8, allocs);
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&buffers.d_out_token),
                      sizeof(std::uint32_t)));
  allocs.push_back(buffers.d_out_token);

  float* capture = AllocDevice(2 * kHidden, allocs);
  buffers.target_layer_features = capture;
  buffers.target_layer_ids = taps;
  buffers.target_layer_count = 2;

  gufo::hip::ExecuteFlintDecodeStep(device_weights, token, pos,
                                    /*compute_logits=*/false, buffers);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> gpu_capture(2 * kHidden, 0.0F);
  HIP_CHECK(hipMemcpy(gpu_capture.data(), capture,
                      gpu_capture.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  for (std::size_t t = 0; t < 2; ++t) {
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      const float diff =
          std::abs(gpu_capture[t * kHidden + i] - cpu_taps[t][i]);
      Expect(diff <= 1e-3F * std::max(1.0F, std::abs(cpu_taps[t][i])),
             "capture tap " + std::to_string(t) + " element " +
                 std::to_string(i) + " differs by " + std::to_string(diff));
    }
  }
  std::cout << "target-layer capture ok (taps 0,2 match the CPU residual)\n";
  for (void* p : allocs) HIP_CHECK(hipFree(p));
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string_view(argv[1]) == "probe") {
    HostStore probe_store;
    auto probe_weights = BuildSyntheticWeights(probe_store);
    std::vector<void*> probe_allocs;
    ProbeLayer0Ops(probe_weights, probe_allocs);
    for (void* p : probe_allocs) HIP_CHECK(hipFree(p));
    std::cout << "probe ok\n";
    return 0;
  }
  // Debug aid: restrict the layer count to bisect divergences
  // (1 = dense + global + NoPE only).
  std::uint32_t layer_limit = kLayers;
  if (argc > 1) layer_limit = static_cast<std::uint32_t>(std::atoi(argv[1]));

  HostStore store;
  auto weights = BuildSyntheticWeights(store);
  if (layer_limit < kLayers) {
    weights.config.num_layers = layer_limit;
    weights.layers.resize(layer_limit);
  }

  // CPU reference: three sequential decode steps.
  gufo::models::QwenKvCache kv(weights.config.num_layers, kKvHeads,
                               kMaxContext, kHeadDim);
  gufo::models::FlintScratchArena arena(weights.config);
  std::vector<std::vector<float>> cpu_logits;
  const std::uint32_t tokens[3] = {3, 11, 7};
  for (std::uint32_t step = 0; step < 3; ++step) {
    std::vector<float> logits(kVocab, 0.0F);
    gufo::models::FlintForwardModel(tokens[step], step, weights, kv, arena,
                                    logits);
    cpu_logits.push_back(logits);
  }

  // GPU: same weights, same tokens, fresh cache.
  std::vector<void*> allocs;
  const auto device_weights = ToDeviceModel(weights, allocs);

  gufo::hip::FlintGpuStepBuffers buffers;
  const std::size_t kv_bytes_elems = static_cast<std::size_t>(
                                          weights.config.num_layers) *
                                    kKvHeads * kMaxContext * kHeadDim;
  buffers.hidden = AllocDevice(kHidden, allocs);
  buffers.normed = AllocDevice(kHidden, allocs);
  buffers.q = AllocDevice(kHeads * kHeadDim, allocs);
  buffers.k = AllocDevice(kKvHeads * kHeadDim, allocs);
  buffers.v = AllocDevice(kKvHeads * kHeadDim, allocs);
  buffers.attn_out = AllocDevice(kHeads * kHeadDim, allocs);
  buffers.attn_proj = AllocDevice(kHidden, allocs);
  buffers.act = AllocDevice(kIntermediate, allocs);
  buffers.act_gate = AllocDevice(kIntermediate, allocs);
  buffers.expert_out = AllocDevice(kHidden, allocs);
  buffers.mlp_out = AllocDevice(kHidden, allocs);
  buffers.router_logits = AllocDevice(kExperts, allocs);
  buffers.logits = AllocDevice(kVocab, allocs);
  buffers.k_cache = AllocDevice(kv_bytes_elems, allocs);
  buffers.v_cache = AllocDevice(kv_bytes_elems, allocs);
  buffers.max_context = kMaxContext;
  // F-I2 ring metadata (the arena's FlintArenaPlaneElements). At this context
  // every layer's ring is max_context rows, so the planes are uniform.
  std::vector<std::size_t> kv_layer_base(weights.config.num_layers);
  std::vector<std::uint32_t> kv_layer_rows(weights.config.num_layers,
                                           kMaxContext);
  for (std::uint32_t l = 0; l < weights.config.num_layers; ++l) {
    kv_layer_base[l] = static_cast<std::size_t>(l) * kKvHeads * kMaxContext *
                       kHeadDim;
  }
  buffers.kv_layer_base = kv_layer_base.data();
  buffers.kv_layer_rows = kv_layer_rows.data();
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&buffers.selected),
                      8 * sizeof(std::uint32_t)));
  allocs.push_back(buffers.selected);
  buffers.expert_weights = AllocDevice(8, allocs);
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&buffers.d_out_token),
                      sizeof(std::uint32_t)));
  allocs.push_back(buffers.d_out_token);

  for (std::uint32_t step = 0; step < 3; ++step) {
    gufo::hip::ExecuteFlintDecodeStep(device_weights, tokens[step], step,
                                      /*compute_logits=*/true, buffers);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> gpu_logits(kVocab, 0.0F);
    HIP_CHECK(hipMemcpy(gpu_logits.data(), buffers.logits,
                        kVocab * sizeof(float), hipMemcpyDeviceToHost));
    CompareLogits(cpu_logits[step], gpu_logits, step);

    std::uint32_t out_token = 0;
    HIP_CHECK(hipMemcpy(&out_token, buffers.d_out_token, sizeof(out_token),
                        hipMemcpyDeviceToHost));
    std::size_t argmax = 0;
    for (std::size_t i = 1; i < kVocab; ++i) {
      if (gpu_logits[i] > gpu_logits[argmax]) argmax = i;
    }
    Expect(out_token == static_cast<std::uint32_t>(argmax),
           "device argmax disagrees with logits at step " +
               std::to_string(step));
  }

  // Snapshot/restore round trip (5.3c): capture the fp32 KV planes and the
  // frontier logits after 3 tokens, advance one more step, restore the
  // snapshot, and require the restored logits to equal the captured ones and
  // the rerun of the advanced step to reproduce its logits bit for bit.
  const std::size_t plane_elems = kv_bytes_elems;
  std::vector<float> snap_k(plane_elems), snap_v(plane_elems);
  std::vector<float> snap_logits(kVocab, 0.0F);
  HIP_CHECK(hipMemcpy(snap_k.data(), buffers.k_cache,
                      plane_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(snap_v.data(), buffers.v_cache,
                      plane_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(snap_logits.data(), buffers.logits,
                      kVocab * sizeof(float), hipMemcpyDeviceToHost));
  gufo::hip::ExecuteFlintDecodeStep(device_weights, 5, 3,
                                    /*compute_logits=*/true, buffers);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> advanced_logits(kVocab, 0.0F);
  HIP_CHECK(hipMemcpy(advanced_logits.data(), buffers.logits,
                      kVocab * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(buffers.k_cache, snap_k.data(),
                      plane_elems * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(buffers.v_cache, snap_v.data(),
                      plane_elems * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(buffers.logits, snap_logits.data(),
                      kVocab * sizeof(float), hipMemcpyHostToDevice));
  std::vector<float> restored_logits(kVocab, 0.0F);
  HIP_CHECK(hipMemcpy(restored_logits.data(), buffers.logits,
                      kVocab * sizeof(float), hipMemcpyDeviceToHost));
  Expect(restored_logits == snap_logits,
         "restored frontier logits differ from the snapshot");
  gufo::hip::ExecuteFlintDecodeStep(device_weights, 5, 3,
                                    /*compute_logits=*/true, buffers);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> rerun_logits(kVocab, 0.0F);
  HIP_CHECK(hipMemcpy(rerun_logits.data(), buffers.logits,
                      kVocab * sizeof(float), hipMemcpyDeviceToHost));
  Expect(rerun_logits == advanced_logits,
         "rerun after restore does not reproduce the advanced step");
  std::cout << "snapshot round trip ok (" << plane_elems
            << " KV elems per plane, bit-exact rerun)\n";

  // compute_logits=false must leave the hidden state correct without
  // touching logits: run one more step and confirm it does not crash and
  // logits buffer is untouched (still the previous step's values).
  std::vector<float> saved(kVocab, 0.0F);
  HIP_CHECK(hipMemcpy(saved.data(), buffers.logits, kVocab * sizeof(float),
                      hipMemcpyDeviceToHost));
  gufo::hip::ExecuteFlintDecodeStep(device_weights, 5, 3,
                                    /*compute_logits=*/false, buffers);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> unchanged(kVocab, 0.0F);
  HIP_CHECK(hipMemcpy(unchanged.data(), buffers.logits, kVocab * sizeof(float),
                      hipMemcpyDeviceToHost));
  Expect(unchanged == saved, "compute_logits=false must not write logits");

  TestTargetLayerCapture();

  std::cout << "flint_decode_step_test ok (3 tokens CPU/GPU parity + "
               "no-logits step + target-layer capture)\n";
  for (void* p : allocs) HIP_CHECK(hipFree(p));
  return 0;
}
