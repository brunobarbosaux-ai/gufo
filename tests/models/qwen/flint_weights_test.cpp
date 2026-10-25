// Flint load-path tests (F-B1, F-B2, F-C1, F-C2, F-C4): the exact tensor
// inventory, per-layer stack attributes, router BF16 acceptance, and
// fail-closed diagnostics that name the layer and tensor.
//
// The real-artifact walk (F-C3) runs when GUFO_FLINT_GGUF points at the
// pinned flint GGUF; it mmaps the header only, so it costs no copies.

#include "src/models/qwen/flint_forward.hpp"
#include "src/models/qwen/state.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/flint_options.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"

#include <algorithm>
#include <memory>

namespace {

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

// In-memory GGUF builder (same shape as the gguf_reader_test fixture).
class FlintGgufBuilder {
 public:
  FlintGgufBuilder() {
    const char magic[4] = {'G', 'G', 'U', 'F'};
    AppendBytes(magic, 4);
    AppendPod(static_cast<std::uint32_t>(3));
    tensor_count_pos_ = buffer_.size();
    AppendPod(static_cast<std::uint64_t>(0));
    metadata_count_pos_ = buffer_.size();
    AppendPod(static_cast<std::uint64_t>(0));
  }

  void AddMetadataString(std::string_view key, std::string_view val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
    AppendString(val);
    ++metadata_count_;
  }
  void AddMetadataUint32(std::string_view key, std::uint32_t val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kUint32));
    AppendPod(val);
    ++metadata_count_;
  }
  void AddMetadataFloat32(std::string_view key, float val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kFloat32));
    AppendPod(val);
    ++metadata_count_;
  }
  void AddMetadataBool(std::string_view key, bool val) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kBool));
    const std::uint8_t b = val ? 1 : 0;
    AppendPod(b);
    ++metadata_count_;
  }
  void AddMetadataUint8Array(std::string_view key,
                             const std::vector<std::uint8_t>& values) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kArray));
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kUint8));
    AppendPod(static_cast<std::uint64_t>(values.size()));
    for (const auto v : values) AppendPod(v);
    ++metadata_count_;
  }

  void AddTensor(std::string_view name, const std::vector<std::uint64_t>& dims,
                 gufo::core::GgmlType type) {
    std::uint64_t elements = 1;
    for (const auto d : dims) elements *= d;
    const std::uint64_t bytes = gufo::quant::EncodedSizeBytes(type, elements);
    tensors_.push_back({std::string(name), dims, type, next_offset_});
    next_offset_ += bytes;
    next_offset_ = (next_offset_ + 31) / 32 * 32;  // GGUF default alignment
  }

  std::vector<std::uint8_t> Build() {
    const std::uint64_t tensor_count = tensors_.size();
    std::memcpy(buffer_.data() + tensor_count_pos_, &tensor_count,
                sizeof(tensor_count));
    std::memcpy(buffer_.data() + metadata_count_pos_, &metadata_count_,
                sizeof(metadata_count_));
    for (const auto& t : tensors_) {
      AppendString(t.name);
      AppendPod(static_cast<std::uint32_t>(t.dims.size()));
      for (const auto d : t.dims) AppendPod(d);
      AppendPod(static_cast<std::uint32_t>(t.type));
      AppendPod(t.offset);
    }
    std::size_t rem = buffer_.size() % 32;
    if (rem != 0) buffer_.resize(buffer_.size() + (32 - rem), 0);
    buffer_.resize(buffer_.size() + next_offset_, 0xAB);
    return buffer_;
  }

 private:
  struct TensorRecord {
    std::string name;
    std::vector<std::uint64_t> dims;
    gufo::core::GgmlType type;
    std::uint64_t offset;
  };

  template <typename T>
  void AppendPod(T value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    buffer_.insert(buffer_.end(), bytes, bytes + sizeof(value));
  }
  void AppendBytes(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    buffer_.insert(buffer_.end(), bytes, bytes + size);
  }
  void AppendString(std::string_view s) {
    AppendPod(static_cast<std::uint64_t>(s.size()));
    AppendBytes(s.data(), s.size());
  }

  std::vector<std::uint8_t> buffer_;
  std::vector<TensorRecord> tensors_;
  std::size_t tensor_count_pos_ = 0;
  std::size_t metadata_count_pos_ = 0;
  std::uint64_t next_offset_ = 0;
  std::uint64_t metadata_count_ = 0;
};

// A minimal but fully valid flint artifact: 3 layers, layer 0 dense,
// layers 1-2 MoE, window/RoPE patterns that vary per layer.
struct FlintFixtureOptions {
  std::string omit_tensor;  // exact tensor name to drop (negative tests)
  gufo::core::GgmlType router_type = gufo::core::GgmlType::kBF16;
  std::vector<std::uint8_t> rope_pattern = {0, 1, 1};  // layer 0 NoPE (F-D3)
};

constexpr std::uint64_t kHidden = 64;
constexpr std::uint64_t kHeads = 4;
constexpr std::uint64_t kKvHeads = 2;
constexpr std::uint64_t kHeadDim = 16;
constexpr std::uint64_t kVocab = 16;
constexpr std::uint64_t kIntermediate = 128;
constexpr std::uint64_t kExperts = 8;
constexpr std::uint64_t kExpertFf = 32;
constexpr std::uint64_t kSharedFf = 64;
constexpr std::uint32_t kLayers = 3;
constexpr std::uint32_t kLeadingDense = 1;

bool Omitted(const FlintFixtureOptions& opt, const std::string& name) {
  return opt.omit_tensor == name;
}

std::vector<std::uint8_t> BuildFlintFixture(const FlintFixtureOptions& opt = {}) {
  FlintGgufBuilder b;
  b.AddMetadataString("general.architecture", "flint");
  b.AddMetadataString("general.name", "flint-test-fixture");
  b.AddMetadataUint32("flint.block_count", kLayers);
  b.AddMetadataUint32("flint.context_length", 1024);
  b.AddMetadataUint32("flint.embedding_length", kHidden);
  b.AddMetadataUint32("flint.feed_forward_length", kIntermediate);
  b.AddMetadataUint32("flint.attention.head_count", kHeads);
  b.AddMetadataUint32("flint.attention.head_count_kv", kKvHeads);
  b.AddMetadataUint32("flint.attention.key_length", kHeadDim);
  b.AddMetadataUint32("flint.attention.value_length", kHeadDim);
  b.AddMetadataFloat32("flint.rope.freq_base", 10000.0F);
  b.AddMetadataFloat32("flint.attention.layer_norm_rms_epsilon", 1e-6F);
  b.AddMetadataUint32("flint.expert_count", kExperts);
  b.AddMetadataUint32("flint.expert_used_count", 2);
  b.AddMetadataUint32("flint.expert_feed_forward_length", kExpertFf);
  b.AddMetadataUint32("flint.expert_shared_count", 1);
  b.AddMetadataUint32("flint.expert_shared_feed_forward_length", kSharedFf);
  b.AddMetadataUint32("flint.expert_gating_func", 2);
  b.AddMetadataFloat32("flint.expert_weights_scale", 1.0F);
  b.AddMetadataBool("flint.expert_weights_norm", true);
  b.AddMetadataUint32("flint.leading_dense_block_count", kLeadingDense);
  b.AddMetadataUint32("flint.attention.sliding_window", 8);
  b.AddMetadataUint8Array("flint.attention.sliding_window_pattern",
                          {0, 1, 1});
  b.AddMetadataUint8Array("flint.attention.rope_pattern", opt.rope_pattern);

  const auto F32 = gufo::core::GgmlType::kF32;
  const auto Q8 = gufo::core::GgmlType::kQ8_0;

  if (!Omitted(opt, "token_embd.weight"))
    b.AddTensor("token_embd.weight", {kHidden, kVocab}, F32);
  if (!Omitted(opt, "output_norm.weight"))
    b.AddTensor("output_norm.weight", {kHidden}, F32);
  if (!Omitted(opt, "output.weight"))
    b.AddTensor("output.weight", {kHidden, kVocab}, F32);

  for (std::uint32_t i = 0; i < kLayers; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    if (!Omitted(opt, p + "attn_norm.weight"))
      b.AddTensor(p + "attn_norm.weight", {kHidden}, F32);
    if (!Omitted(opt, p + "attn_q.weight"))
      b.AddTensor(p + "attn_q.weight", {kHidden, kHeads * kHeadDim}, Q8);
    if (!Omitted(opt, p + "attn_k.weight"))
      b.AddTensor(p + "attn_k.weight", {kHidden, kKvHeads * kHeadDim}, Q8);
    if (!Omitted(opt, p + "attn_v.weight"))
      b.AddTensor(p + "attn_v.weight", {kHidden, kKvHeads * kHeadDim}, Q8);
    if (!Omitted(opt, p + "attn_output.weight"))
      b.AddTensor(p + "attn_output.weight", {kHeads * kHeadDim, kHidden}, Q8);
    if (!Omitted(opt, p + "attn_q_norm.weight"))
      b.AddTensor(p + "attn_q_norm.weight", {kHeadDim}, F32);
    if (!Omitted(opt, p + "attn_k_norm.weight"))
      b.AddTensor(p + "attn_k_norm.weight", {kHeadDim}, F32);
    if (!Omitted(opt, p + "attn_sinks.weight"))
      b.AddTensor(p + "attn_sinks.weight", {kHeads}, F32);
    if (!Omitted(opt, p + "ffn_norm.weight"))
      b.AddTensor(p + "ffn_norm.weight", {kHidden}, F32);

    if (i < kLeadingDense) {
      if (!Omitted(opt, p + "ffn_gate.weight"))
        b.AddTensor(p + "ffn_gate.weight", {kHidden, kIntermediate}, Q8);
      if (!Omitted(opt, p + "ffn_up.weight"))
        b.AddTensor(p + "ffn_up.weight", {kHidden, kIntermediate}, Q8);
      if (!Omitted(opt, p + "ffn_down.weight"))
        b.AddTensor(p + "ffn_down.weight", {kIntermediate, kHidden}, Q8);
    } else {
      if (!Omitted(opt, p + "ffn_gate_inp.weight"))
        b.AddTensor(p + "ffn_gate_inp.weight", {kHidden, kExperts},
                    opt.router_type);
      if (!Omitted(opt, p + "ffn_gate_exps.weight"))
        b.AddTensor(p + "ffn_gate_exps.weight", {kHidden, kExpertFf, kExperts},
                    Q8);
      if (!Omitted(opt, p + "ffn_up_exps.weight"))
        b.AddTensor(p + "ffn_up_exps.weight", {kHidden, kExpertFf, kExperts},
                    Q8);
      if (!Omitted(opt, p + "ffn_down_exps.weight"))
        b.AddTensor(p + "ffn_down_exps.weight", {kExpertFf, kHidden, kExperts},
                    Q8);
      if (!Omitted(opt, p + "ffn_gate_shexp.weight"))
        b.AddTensor(p + "ffn_gate_shexp.weight", {kHidden, kSharedFf}, Q8);
      if (!Omitted(opt, p + "ffn_up_shexp.weight"))
        b.AddTensor(p + "ffn_up_shexp.weight", {kHidden, kSharedFf}, Q8);
      if (!Omitted(opt, p + "ffn_down_shexp.weight"))
        b.AddTensor(p + "ffn_down_shexp.weight", {kSharedFf, kHidden}, Q8);
    }
  }
  return b.Build();
}

gufo::core::GgufReader* OpenFixture(const std::vector<std::uint8_t>& bytes,
                                    std::unique_ptr<gufo::core::GgufReader>& out,
                                    std::string& err) {
  out = gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &err);
  return out.get();
}

// ---------------------------------------------------------------- tests

void TestLoadsFullInventory() {
  const auto bytes = BuildFlintFixture();
  std::unique_ptr<gufo::core::GgufReader> reader;
  std::string err;
  auto* r = OpenFixture(bytes, reader, err);
  Expect(r != nullptr, "fixture opens: " + err);

  std::string load_err;
  const auto weights =
      gufo::models::FlintModelWeights::LoadFromGguf(*r, &load_err);
  Expect(weights.has_value(), "loads full inventory: " + load_err);
  Expect(weights->layers.size() == kLayers, "all layers present");

  // F-B1/F-B2: stack attributes per layer.
  Expect(weights->layers[0].attrs.dense_ffn, "layer 0 is dense");
  Expect(weights->layers[1].attrs.dense_ffn == false, "layer 1 is MoE");
  Expect(weights->layers[2].attrs.dense_ffn == false, "layer 2 is MoE");
  Expect(!weights->layers[0].attrs.windowed, "layer 0 global");
  Expect(weights->layers[1].attrs.windowed, "layer 1 windowed");
  Expect(!weights->layers[0].attrs.rope, "layer 0 NoPE");
  Expect(weights->layers[1].attrs.rope, "layer 1 RoPE");

  // Every tensor present with the right element counts.
  const auto& l0 = weights->layers[0];
  Expect(l0.attn_sinks.num_elements == kHeads, "sinks [heads]");
  Expect(l0.attn_q.num_elements == kHeads * kHeadDim * kHidden, "attn_q");
  Expect(!l0.ffn_gate.empty() && l0.ffn_gate_inp.empty(),
         "dense layer has no router");
  const auto& l1 = weights->layers[1];
  Expect(l1.ffn_gate_inp.type == gufo::core::GgmlType::kBF16,
         "router is BF16 (F-C2)");
  Expect(l1.ffn_gate_exps.num_elements == kHidden * kExpertFf * kExperts,
         "gate_exps");
  Expect(l1.ffn_down_exps.num_elements == kExpertFf * kHidden * kExperts,
         "down_exps");
  Expect(l1.ffn_down_shexp.num_elements == kSharedFf * kHidden,
         "down_shexp");

  // F-B1 sizing trap: every layer is a full-attention layer, no SSM state.
  Expect(weights->config.full_attention_interval == 1, "interval forced to 1");
  Expect(weights->config.FullAttentionLayerCount() == kLayers,
         "KV covers all layers");
  Expect(weights->config.SsmLayerCount() == 0, "no SSM layers");
  std::cout << "TestLoadsFullInventory ok\n";
}

void TestMissingTensorNamesLayerAndTensor() {
  // F-C4: a missing tensor must name the layer and the tensor.
  const auto bytes = BuildFlintFixture({.omit_tensor = "blk.1.attn_sinks.weight"});
  std::unique_ptr<gufo::core::GgufReader> reader;
  std::string err;
  auto* r = OpenFixture(bytes, reader, err);
  Expect(r != nullptr, "fixture opens");
  std::string load_err;
  const auto weights =
      gufo::models::FlintModelWeights::LoadFromGguf(*r, &load_err);
  Expect(!weights.has_value(), "missing sinks rejected");
  Expect(load_err.find("blk.1") != std::string::npos &&
             load_err.find("attn_sinks") != std::string::npos,
         "error names layer and tensor: " + load_err);
  std::cout << "TestMissingTensorNamesLayerAndTensor ok\n";
}

void TestMoeLayerRequiresEveryExpertTensor() {
  const auto bytes =
      BuildFlintFixture({.omit_tensor = "blk.2.ffn_up_shexp.weight"});
  std::unique_ptr<gufo::core::GgufReader> reader;
  std::string err;
  auto* r = OpenFixture(bytes, reader, err);
  Expect(r != nullptr, "fixture opens");
  std::string load_err;
  const auto weights =
      gufo::models::FlintModelWeights::LoadFromGguf(*r, &load_err);
  Expect(!weights.has_value(), "missing shexp rejected");
  Expect(load_err.find("blk.2") != std::string::npos &&
             load_err.find("ffn_up_shexp") != std::string::npos,
         "error names layer and tensor: " + load_err);
  std::cout << "TestMoeLayerRequiresEveryExpertTensor ok\n";
}

void TestRouterRejectsQuantizedType() {
  // F-C2: BF16 is accepted (covered by the happy path); a quantized router
  // must be refused, never silently decoded.
  const auto bytes = BuildFlintFixture(
      {.router_type = gufo::core::GgmlType::kQ8_0});
  std::unique_ptr<gufo::core::GgufReader> reader;
  std::string err;
  auto* r = OpenFixture(bytes, reader, err);
  Expect(r != nullptr, "fixture opens");
  std::string load_err;
  const auto weights =
      gufo::models::FlintModelWeights::LoadFromGguf(*r, &load_err);
  Expect(!weights.has_value(), "quantized router rejected");
  Expect(load_err.find("router") != std::string::npos,
         "error names the router role: " + load_err);
  std::cout << "TestRouterRejectsQuantizedType ok\n";
}

void TestTiedHeadRejected() {
  const auto bytes = BuildFlintFixture({.omit_tensor = "output.weight"});
  std::unique_ptr<gufo::core::GgufReader> reader;
  std::string err;
  auto* r = OpenFixture(bytes, reader, err);
  Expect(r != nullptr, "fixture opens");
  std::string load_err;
  const auto weights =
      gufo::models::FlintModelWeights::LoadFromGguf(*r, &load_err);
  Expect(!weights.has_value(), "tied head refused");
  Expect(load_err.find("untied LM head") != std::string::npos,
         "error explains untied requirement: " + load_err);
  std::cout << "TestTiedHeadRejected ok\n";
}

void TestQwenRuntimeRefusesFlint() {
  // The D3 policy gate stays: flint never loads through the qwen runtime.
  const auto bytes = BuildFlintFixture();
  std::unique_ptr<gufo::core::GgufReader> reader;
  std::string err;
  auto* r = OpenFixture(bytes, reader, err);
  Expect(r != nullptr, "fixture opens");
  std::string load_err;
  const auto weights =
      gufo::models::QwenModelWeights::LoadFromGguf(*r, &load_err);
  Expect(!weights.has_value(), "qwen runtime refuses flint");
  Expect(load_err.find("FlintModelWeights::LoadFromGguf") !=
             std::string::npos,
         "error points at the flint loader: " + load_err);
  std::cout << "TestQwenRuntimeRefusesFlint ok\n";
}

void TestScratchArenaHasNoSsmState() {
  // F-B1: the flint arena carries routing/expert scratch and full-context
  // scores, and no SSM span exists at all.
  const auto bytes = BuildFlintFixture();
  std::unique_ptr<gufo::core::GgufReader> reader;
  std::string err;
  auto* r = OpenFixture(bytes, reader, err);
  Expect(r != nullptr, "fixture opens");
  std::string load_err;
  const auto weights =
      gufo::models::FlintModelWeights::LoadFromGguf(*r, &load_err);
  Expect(weights.has_value(), "loads: " + load_err);

  const gufo::models::FlintScratchArena arena(weights->config);
  Expect(arena.attn_scores.size() == 1024, "scores sized to the context");
  Expect(arena.router_probs.size() == kExperts, "router scratch");
  Expect(arena.logits.size() == kVocab, "logit scratch");
  Expect(arena.expert_gate.size() >= kExpertFf &&
             arena.shared_gate.size() == kSharedFf,
         "expert + shared scratch");
  std::cout << "TestScratchArenaHasNoSsmState ok\n";
}

// F-D5 CPU reference: the fixture model runs a token through every layer
// (dense + MoE, windowed + global, rope + NoPE) and produces finite,
// deterministic logits.
void TestForwardModelSmoke() {
  const auto bytes = BuildFlintFixture();
  std::unique_ptr<gufo::core::GgufReader> reader;
  std::string err;
  auto* r = OpenFixture(bytes, reader, err);
  Expect(r != nullptr, "fixture opens");
  std::string load_err;
  const auto weights =
      gufo::models::FlintModelWeights::LoadFromGguf(*r, &load_err);
  Expect(weights.has_value(), "loads: " + load_err);

  gufo::models::QwenKvCache kv(kLayers, kKvHeads, 1024, kHeadDim);
  gufo::models::FlintScratchArena arena(weights->config);
  std::vector<float> logits(kVocab, 0.0F);

  gufo::models::FlintForwardModel(0, 0, *weights, kv, arena, logits);
  for (const float v : logits) {
    Expect(std::isfinite(v), "logits finite at pos 0");
  }
  const std::vector<float> first_logits = logits;
  // Second token extends the KV; still finite.
  gufo::models::FlintForwardModel(5, 1, *weights, kv, arena, logits);
  for (const float v : logits) {
    Expect(std::isfinite(v), "logits finite at pos 1");
  }
  // A fresh replay of the same first token reproduces it bit-exactly.
  std::vector<float> logits_again(kVocab, 0.0F);
  gufo::models::QwenKvCache kv2(kLayers, kKvHeads, 1024, kHeadDim);
  gufo::models::FlintScratchArena arena2(weights->config);
  gufo::models::FlintForwardModel(0, 0, *weights, kv2, arena2, logits_again);
  Expect(logits_again == first_logits,
         "deterministic across identical replays");
  std::cout << "TestForwardModelSmoke ok\n";
}

// F-D3 (1.8): rope_pattern drives RoPE in the forward path. Two fixtures
// identical except rope_pattern ({0,1,1} vs {0,0,0}): layer 0 is NoPE in
// both, so its K is bit-identical across fixtures and position-invariant;
// layer 1 sees an identical input in both runs (layer 0 unchanged), so its
// K differs only when the pattern says RoPE — and its V (never rotated)
// must stay bit-identical.
void TestNopeHonoredInForward() {
  const auto bytes_rope = BuildFlintFixture();
  const auto bytes_nope =
      BuildFlintFixture({.omit_tensor = "", .rope_pattern = {0, 0, 0}});
  std::unique_ptr<gufo::core::GgufReader> reader_rope;
  std::unique_ptr<gufo::core::GgufReader> reader_nope;
  std::string err;
  auto* r_rope = OpenFixture(bytes_rope, reader_rope, err);
  Expect(r_rope != nullptr, "rope fixture opens");
  auto* r_nope = OpenFixture(bytes_nope, reader_nope, err);
  Expect(r_nope != nullptr, "nope fixture opens");

  std::string load_err;
  const auto rope_weights =
      gufo::models::FlintModelWeights::LoadFromGguf(*r_rope, &load_err);
  Expect(rope_weights.has_value(), "rope fixture loads: " + load_err);
  const auto nope_weights =
      gufo::models::FlintModelWeights::LoadFromGguf(*r_nope, &load_err);
  Expect(nope_weights.has_value(), "nope fixture loads: " + load_err);

  const auto& config = rope_weights->config;
  Expect(config.IsRopeLayer(0) == false, "layer 0 NoPE by pattern");
  Expect(config.IsRopeLayer(1) && config.IsRopeLayer(2),
         "layers 1-2 apply RoPE by pattern");
  Expect(config.IsWindowedLayer(0) == false, "layer 0 global by pattern");
  Expect(config.IsWindowedLayer(1) && config.IsWindowedLayer(2),
         "layers 1-2 windowed by pattern");

  constexpr std::uint32_t kPos = 7;
  const auto run_at = [&](const gufo::models::FlintModelWeights& weights,
                          gufo::models::QwenKvCache& kv) {
    gufo::models::FlintScratchArena arena(weights.config);
    std::vector<float> logits(kVocab, 0.0F);
    gufo::models::FlintForwardModel(0, kPos, weights, kv, arena, logits);
  };
  gufo::models::QwenKvCache kv_rope(kLayers, kKvHeads, 1024, kHeadDim);
  gufo::models::QwenKvCache kv_nope(kLayers, kKvHeads, 1024, kHeadDim);
  run_at(*rope_weights, kv_rope);
  run_at(*nope_weights, kv_nope);

  // NoPE layer: K is position-independent and rope_pattern-blind.
  const auto k0_rope = kv_rope.GetKeySlice(0, 0, kPos);
  const auto k0_nope = kv_nope.GetKeySlice(0, 0, kPos);
  Expect(std::equal(k0_rope.begin(), k0_rope.end(), k0_nope.begin()),
         "NoPE layer K unaffected by rope_pattern (F-D3)");
  gufo::models::QwenKvCache kv_pos0(kLayers, kKvHeads, 1024, kHeadDim);
  {
    gufo::models::FlintScratchArena arena(config);
    std::vector<float> logits(kVocab, 0.0F);
    gufo::models::FlintForwardModel(0, 0, *rope_weights, kv_pos0, arena,
                                    logits);
  }
  const auto k0_pos0 = kv_pos0.GetKeySlice(0, 0, 0);
  Expect(std::equal(k0_pos0.begin(), k0_pos0.end(), k0_rope.begin()),
         "NoPE layer K is position-invariant (F-D3)");

  // Rope layer: K rotates exactly where the pattern says; V never rotates.
  const auto k1_rope = kv_rope.GetKeySlice(1, 0, kPos);
  const auto k1_nope = kv_nope.GetKeySlice(1, 0, kPos);
  Expect(!std::equal(k1_rope.begin(), k1_rope.end(), k1_nope.begin()),
         "rope layer K rotates when the pattern says rope (F-D3)");
  const auto v1_rope = kv_rope.GetValueSlice(1, 0, kPos);
  const auto v1_nope = kv_nope.GetValueSlice(1, 0, kPos);
  Expect(std::equal(v1_rope.begin(), v1_rope.end(), v1_nope.begin()),
         "V is never rotated by RoPE (F-D3)");
  std::cout << "TestNopeHonoredInForward ok\n";
}

// F-C3: walk the pinned artifact's real header. Header-only mmap; runs
// wherever GUFO_FLINT_GGUF is set (skipped elsewhere).
void TestRealArtifactInventory() {
  const char* path = std::getenv("GUFO_FLINT_GGUF");
  if (path == nullptr || *path == '\0') {
    std::cout << "TestRealArtifactInventory skipped (GUFO_FLINT_GGUF unset)\n";
    return;
  }
  std::string err;
  auto reader = gufo::core::GgufReader::OpenFile(path, &err);
  Expect(reader != nullptr, "real artifact opens: " + err);

  std::string load_err;
  const auto weights = gufo::models::FlintModelWeights::LoadFromGguf(
      *reader, &load_err);
  Expect(weights.has_value(), "real artifact loads: " + load_err);
  Expect(weights->layers.size() == 49, "49 layers");
  Expect(weights->config.FullAttentionLayerCount() == 49, "KV 49 layers");

  // F-D3 (1.8): the 14 global layers are exactly the NoPE layers — the
  // window and rope patterns mirror each other on the pinned artifact.
  std::size_t global_layers = 0;
  for (std::uint32_t i = 0; i < weights->config.num_layers; ++i) {
    if (!weights->config.IsWindowedLayer(i)) {
      ++global_layers;
      Expect(!weights->config.IsRopeLayer(i),
             "global layer " + std::to_string(i) + " is NoPE (F-D3)");
    }
  }
  Expect(global_layers == 14, "14 global layers of 49");

  // The UD quantizer mixes Q6_K and Q8_0 gate/up experts across layers;
  // both must be accepted by the projection role (F-C3 coverage proof).
  int q6k_layers = 0, q8_layers = 0;
  for (const auto& l : weights->layers) {
    if (l.attrs.dense_ffn) continue;
    Expect(l.ffn_gate_inp.type == gufo::core::GgmlType::kBF16,
           "router BF16 on every MoE layer");
    Expect(l.attn_sinks.num_elements == 32, "sinks [32]");
    if (l.ffn_gate_exps.type == gufo::core::GgmlType::kQ6_K) {
      ++q6k_layers;
    } else if (l.ffn_gate_exps.type == gufo::core::GgmlType::kQ8_0) {
      ++q8_layers;
    } else {
      Expect(false, "unexpected gate_exps type");
    }
  }
  Expect(q6k_layers == 46 && q8_layers == 1,
         "mixed Q6_K/Q8_0 expert coverage (46+1)");
  std::cout << "TestRealArtifactInventory ok (46 Q6_K + 1 Q8_0 MoE layers)\n";
}

// F-B3/F-B4: the option gate rejects every drafting/vision flag, naming it.
void TestFlintOptionGate() {
  std::string err;
  Expect(gufo::core::ValidateFlintOptions(false, false, false, false, false, &err),
         "clean options pass");
  err.clear();
  Expect(!gufo::core::ValidateFlintOptions(true, false, false, false, false, &err),
         "--mmproj rejected");
  Expect(err.find("--mmproj") != std::string::npos, "names --mmproj: " + err);
  err.clear();
  Expect(!gufo::core::ValidateFlintOptions(false, false, true, false, false, &err),
         "--mtp-model rejected");
  Expect(err.find("--mtp-model") != std::string::npos,
         "names --mtp-model: " + err);
  err.clear();
  Expect(!gufo::core::ValidateFlintOptions(false, false, false, true, false, &err),
         "--dflash-model rejected");
  Expect(err.find("--dflash-model") != std::string::npos,
         "names --dflash-model: " + err);
  err.clear();
  Expect(!gufo::core::ValidateFlintOptions(false, false, false, false, true, &err),
         "--dspark-model rejected");
  Expect(err.find("--dspark-model") != std::string::npos,
         "names --dspark-model: " + err);
  err.clear();
  Expect(!gufo::core::ValidateFlintOptions(false, true, false, false, false, &err),
         "--speculative rejected");
  Expect(err.find("--speculative") != std::string::npos,
         "names --speculative: " + err);
  std::cout << "TestFlintOptionGate ok\n";
}

}  // namespace

int main() {
  TestLoadsFullInventory();
  TestMissingTensorNamesLayerAndTensor();
  TestMoeLayerRequiresEveryExpertTensor();
  TestRouterRejectsQuantizedType();
  TestTiedHeadRejected();
  TestQwenRuntimeRefusesFlint();
  TestScratchArenaHasNoSsmState();
  TestFlintOptionGate();
  TestForwardModelSmoke();
  TestNopeHonoredInForward();
  TestRealArtifactInventory();
  std::cout << "flint_weights_test: all tests passed\n";
  return 0;
}
