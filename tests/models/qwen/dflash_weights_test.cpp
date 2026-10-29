// DFlash-2 draft load-path tests. The Flint DFlash-2 export omits
// `dflash.rope.dimension_count` (it carries only `dflash.rope.freq_base`), so
// the loader must default the rotary dimension to the head dimension exactly as
// llama.cpp does. The in-memory fixture pins that fallback and the two explicit
// forms; the real-artifact walk (env-gated) proves it on the shipped Q8_0 draft.

#include "src/models/qwen/dflash_weights.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace {

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

// Minimal GGUF writer, same shape as the gguf_reader_test fixture.
class DFlashGgufBuilder {
 public:
  DFlashGgufBuilder() {
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
  void AddMetadataUint32Array(std::string_view key,
                              const std::vector<std::uint32_t>& values) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kArray));
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kUint32));
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

constexpr std::uint64_t kHidden = 64;
constexpr std::uint64_t kIntermediate = 128;
constexpr std::uint64_t kHeads = 4;
constexpr std::uint64_t kKvHeads = 2;
constexpr std::uint64_t kHeadDim = 16;
constexpr std::uint64_t kAttention = kHeads * kHeadDim;
constexpr std::uint64_t kKv = kKvHeads * kHeadDim;
constexpr std::uint64_t kConvKernel = 2;
constexpr std::uint64_t kConvGroup = 16;
constexpr std::uint64_t kGroups = kHidden / kConvGroup;
constexpr std::uint64_t kDynamic = 2 * kConvKernel * kGroups;
constexpr std::uint64_t kSelectorRank = 8;
constexpr std::uint64_t kVocab = 32;
constexpr std::uint32_t kLayers = 2;

// How the fixture expresses the rotary dimension.
enum class RopeForm {
  kAbsent,     // no dimension_count and no dimension_sections (the Flint case)
  kExplicit,   // dflash.rope.dimension_count = 8
  kSections,   // dflash.rope.dimension_sections = [4, 0, 0, 0]
};

std::vector<std::uint8_t> BuildDFlashFixture(RopeForm rope) {
  DFlashGgufBuilder b;
  b.AddMetadataString("general.architecture", "dflash");
  b.AddMetadataUint32("dflash.embedding_length", kHidden);
  b.AddMetadataUint32("dflash.feed_forward_length", kIntermediate);
  b.AddMetadataUint32("dflash.attention.head_count", kHeads);
  b.AddMetadataUint32("dflash.attention.head_count_kv", kKvHeads);
  b.AddMetadataUint32("dflash.attention.key_length", kHeadDim);
  b.AddMetadataFloat32("dflash.rope.freq_base", 10000.0F);
  b.AddMetadataUint32("dflash.context_length", 1024);
  b.AddMetadataUint32("dflash.block_count", kLayers);
  b.AddMetadataUint32("dflash.block_size", 8);
  b.AddMetadataUint32("dflash.conv_kernel_size", kConvKernel);
  b.AddMetadataUint32("dflash.conv_group_size", kConvGroup);
  b.AddMetadataUint32("dflash.selector_rank", kSelectorRank);
  b.AddMetadataUint32("dflash.selector_top_k", 4);
  b.AddMetadataUint32("dflash.attention.sliding_window", 16);
  b.AddMetadataUint32("tokenizer.ggml.mask_token_id", 100);
  b.AddMetadataBool("dflash.attention.causal", false);
  b.AddMetadataUint32Array("dflash.target_layers", {1, 2});
  switch (rope) {
    case RopeForm::kAbsent:
      break;
    case RopeForm::kExplicit:
      b.AddMetadataUint32("dflash.rope.dimension_count", 8);
      break;
    case RopeForm::kSections:
      b.AddMetadataUint32Array("dflash.rope.dimension_sections",
                               {4, 0, 0, 0});
      break;
  }

  const auto F32 = gufo::core::GgmlType::kF32;
  const auto Q8 = gufo::core::GgmlType::kQ8_0;

  b.AddTensor("selector_predecessor.weight", {32, kSelectorRank}, Q8);
  b.AddTensor("selector_successor.weight", {32, kSelectorRank}, Q8);
  b.AddTensor("selector_hidden.weight", {kHidden, kSelectorRank}, Q8);
  b.AddTensor("fc.weight", {kHidden, 2 * kHidden}, Q8);
  b.AddTensor("enc.output_norm.weight", {kHidden}, F32);
  b.AddTensor("output_norm.weight", {kHidden}, F32);

  for (std::uint32_t i = 0; i < kLayers; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    b.AddTensor(p + "attn_norm.weight", {kHidden}, F32);
    b.AddTensor(p + "attn_q.weight", {kHidden, kAttention}, Q8);
    b.AddTensor(p + "attn_k.weight", {kHidden, kKv}, Q8);
    b.AddTensor(p + "attn_v.weight", {kHidden, kKv}, Q8);
    b.AddTensor(p + "attn_output.weight", {kAttention, kHidden}, Q8);
    b.AddTensor(p + "attn_q_norm.weight", {kHeadDim}, F32);
    b.AddTensor(p + "attn_k_norm.weight", {kHeadDim}, F32);
    b.AddTensor(p + "ffn_norm.weight", {kHidden}, F32);
    b.AddTensor(p + "ffn_gate.weight", {kHidden, kIntermediate}, Q8);
    b.AddTensor(p + "ffn_up.weight", {kHidden, kIntermediate}, Q8);
    b.AddTensor(p + "ffn_down.weight", {kIntermediate, kHidden}, Q8);
    b.AddTensor(p + "attn_conv_base", {kHidden, kConvKernel, 2}, F32);
    b.AddTensor(p + "attn_conv_proj.weight", {kHidden, kDynamic}, Q8);
    b.AddTensor(p + "ffn_conv_base", {kHidden, kConvKernel, 2}, F32);
    b.AddTensor(p + "ffn_conv_proj.weight", {kHidden, kDynamic}, Q8);
  }
  return b.Build();
}

std::optional<gufo::speculative::QwenDFlashWeights> LoadFixture(
    const std::vector<std::uint8_t>& bytes, std::string& err) {
  auto reader = gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(),
                                                   &err);
  Expect(reader != nullptr, "fixture opens: " + err);
  return gufo::speculative::QwenDFlashWeights::LoadFromGguf(*reader, &err);
}

// The Flint DFlash-2 export omits the rotary dimension; the loader must fall
// back to the head dimension (llama.cpp parity).
void TestRotaryDimDefaultsToHeadDim() {
  const auto bytes = BuildDFlashFixture(RopeForm::kAbsent);
  std::string err;
  const auto weights = LoadFixture(bytes, err);
  Expect(weights.has_value(), "absent rotary dimension loads: " + err);
  Expect(weights->config.head_dim == kHeadDim, "head_dim parsed");
  Expect(weights->config.rotary_dim == kHeadDim,
         "absent rotary dimension defaults to head_dim");
  Expect(weights->dflash_config.target_layer_ids ==
             std::vector<std::uint32_t>({0, 1}),
         "target layers normalized to zero-based outputs");
  std::cout << "TestRotaryDimDefaultsToHeadDim ok\n";
}

void TestExplicitRotaryDimWins() {
  const auto bytes = BuildDFlashFixture(RopeForm::kExplicit);
  std::string err;
  const auto weights = LoadFixture(bytes, err);
  Expect(weights.has_value(), "explicit rotary dimension loads: " + err);
  Expect(weights->config.rotary_dim == 8,
         "explicit rotary dimension is honored");
  std::cout << "TestExplicitRotaryDimWins ok\n";
}

void TestDegenerateRopeSections() {
  const auto bytes = BuildDFlashFixture(RopeForm::kSections);
  std::string err;
  const auto weights = LoadFixture(bytes, err);
  Expect(weights.has_value(), "degenerate sections load: " + err);
  Expect(weights->config.rotary_dim == 8,
         "degenerate four-section M-RoPE yields 2 * sections[0]");
  std::cout << "TestDegenerateRopeSections ok\n";
}

// Real-artifact walk: runs wherever GUFO_FLINT_DFLASH_GGUF points at the
// shipped draft (header-only mmap, no weight copies).
void TestRealDraftArtifact() {
  const char* path = std::getenv("GUFO_FLINT_DFLASH_GGUF");
  if (path == nullptr || *path == '\0') {
    std::cout << "TestRealDraftArtifact skipped (GUFO_FLINT_DFLASH_GGUF unset)\n";
    return;
  }
  std::string err;
  auto reader = gufo::core::GgufReader::OpenFile(path, &err);
  Expect(reader != nullptr, "real draft opens: " + err);
  const auto weights =
      gufo::speculative::QwenDFlashWeights::LoadFromGguf(*reader, &err);
  Expect(weights.has_value(), "real draft loads: " + err);
  Expect(weights->config.rotary_dim == weights->config.head_dim,
         "real draft rotary dimension equals head_dim");
  Expect(weights->dflash_config.num_layers == 5, "real draft has 5 layers");
  std::cout << "TestRealDraftArtifact ok (head_dim="
            << weights->config.head_dim
            << " rotary_dim=" << weights->config.rotary_dim << ")\n";
}

}  // namespace

int main() {
  TestRotaryDimDefaultsToHeadDim();
  TestExplicitRotaryDimWins();
  TestDegenerateRopeSections();
  TestRealDraftArtifact();
  std::cout << "dflash_weights_test: all tests passed\n";
  return 0;
}
