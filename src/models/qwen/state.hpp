#ifndef GUFO_MODELS_QWEN_STATE_HPP_
#define GUFO_MODELS_QWEN_STATE_HPP_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/model_config.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models {

/// Non-owning reference to a mapped tensor. Consumers validate the exact
/// formats supported for each tensor role.
struct QwenTensorRef {
  const void* data = nullptr;
  core::GgmlType type = core::GgmlType::kF32;
  std::size_t num_elements = 0;
  /// Bytes available from `data` to the end of its mapped storage region.
  /// Synthetic/in-process tensors default to unbounded trusted storage.
  std::size_t available_bytes = std::numeric_limits<std::size_t>::max();

  [[nodiscard]] bool empty() const noexcept {
    return data == nullptr || num_elements == 0;
  }

  [[nodiscard]] std::size_t EncodedSizeBytes() const noexcept {
    return gufo::quant::EncodedSizeBytes(type, num_elements);
  }

  [[nodiscard]] bool FitsAvailableStorage() const noexcept {
    const std::size_t encoded_bytes = EncodedSizeBytes();
    return encoded_bytes != 0 && encoded_bytes <= available_bytes;
  }

  [[nodiscard]] float Get(std::size_t index) const noexcept {
    if (type == core::GgmlType::kF32) {
      return static_cast<const float*>(data)[index];
    }
    if (type == core::GgmlType::kBF16) {
      const auto u16 = static_cast<const std::uint16_t*>(data)[index];
      const std::uint32_t u32 = static_cast<std::uint32_t>(u16) << 16;
      float f = 0.0F;
      std::memcpy(&f, &u32, sizeof(float));
      return f;
    }
    if (type == core::GgmlType::kF16) {
      return gufo::quant::Fp16ToFloat(
          static_cast<const std::uint16_t*>(data)[index]);
    }
    if (type == core::GgmlType::kQ8_K) {
      constexpr std::size_t kBlockSize = 256;
      const std::size_t block_idx = index / kBlockSize;
      const std::size_t pos = index % kBlockSize;
      const auto* block =
          reinterpret_cast<const gufo::quant::block_q8_K*>(data) + block_idx;
      return block->d * static_cast<float>(block->qs[pos]);
    }
    if (type == core::GgmlType::kQ8_0) {
      // Q8_0 block: fp16 scale + 32 int8 values (QK=32, 34 bytes).
      constexpr std::size_t kBlockSize = 32;
      const std::size_t block_idx = index / kBlockSize;
      const std::size_t pos = index % kBlockSize;
      const auto* block =
          reinterpret_cast<const gufo::quant::block_q8_0*>(data) + block_idx;
      return gufo::quant::Fp16ToFloat(block->d) *
             static_cast<float>(block->qs[pos]);
    }
    if (type == core::GgmlType::kQ3_K || type == core::GgmlType::kQ4_K ||
        type == core::GgmlType::kQ5_K || type == core::GgmlType::kQ6_K) {
      // QK=256 block-packed matmul weights. Dequantize the containing block via
      // the canonical (parity-tested) full-row dequant helpers, then index into
      // it. Previously this fell through to a silent 0.0F.
      constexpr std::size_t kBlockSize = 256;
      const std::size_t block_idx = index / kBlockSize;
      const std::size_t pos = index % kBlockSize;
      const std::size_t block_bytes =
          type == core::GgmlType::kQ3_K   ? sizeof(gufo::quant::block_q3_K)
          : type == core::GgmlType::kQ4_K ? sizeof(gufo::quant::block_q4_K)
          : type == core::GgmlType::kQ5_K ? sizeof(gufo::quant::block_q5_K)
                                          : sizeof(gufo::quant::block_q6_K);
      const auto* block_ptr =
          static_cast<const std::uint8_t*>(data) + block_idx * block_bytes;
      float block_buf[kBlockSize];
      switch (type) {
        case core::GgmlType::kQ3_K:
          gufo::quant::DequantizeQ3_K(block_ptr, block_buf, kBlockSize);
          break;
        case core::GgmlType::kQ4_K:
          gufo::quant::DequantizeQ4_K(block_ptr, block_buf, kBlockSize);
          break;
        case core::GgmlType::kQ5_K:
          gufo::quant::DequantizeQ5_K(block_ptr, block_buf, kBlockSize);
          break;
        case core::GgmlType::kQ6_K:
          gufo::quant::DequantizeQ6_K(block_ptr, block_buf, kBlockSize);
          break;
        default:
          break;
      }
      return block_buf[pos];
    }
    // Unsupported (or non-block-aligned) type: fail loudly instead of silently
    // returning 0.0F.
    assert(false && "QwenTensorRef::Get: unsupported GgmlType");
    std::abort();
  }

  [[nodiscard]] std::span<const float> AsFloatSpan() const noexcept {
    if (type == core::GgmlType::kF32 && data != nullptr) {
      return {static_cast<const float*>(data), num_elements};
    }
    return {};
  }
};

/// Tensor weight references for a single transformer layer block (Linear SSM or
/// Full Attention).
struct QwenLayerWeights {
  bool is_full_attention = false;

  // Common Layer Norms
  QwenTensorRef attn_norm;
  QwenTensorRef ffn_norm;

  // Full Attention Weights (every 4th block)
  QwenTensorRef attn_q;
  QwenTensorRef attn_k;
  QwenTensorRef attn_v;
  QwenTensorRef attn_output;
  QwenTensorRef attn_q_norm;
  QwenTensorRef attn_k_norm;

  // Linear Attention / SSM Weights (3 of 4 blocks)
  QwenTensorRef attn_qkv;
  QwenTensorRef attn_gate;
  QwenTensorRef ssm_a;
  QwenTensorRef ssm_conv1d;
  QwenTensorRef ssm_dt;
  QwenTensorRef ssm_alpha;
  QwenTensorRef ssm_beta;
  QwenTensorRef ssm_norm;
  QwenTensorRef ssm_out;

  // Feed Forward Network
  QwenTensorRef ffn_gate;
  QwenTensorRef ffn_up;
  QwenTensorRef ffn_down;

  // MoE Feed Forward Network (qwen35moe). Present iff the model config has
  // expert_count > 0; the dense tensors above stay empty then. The expert
  // tensors keep their full [cols, rows, experts] payload flat; the expert
  // axis stride is the encoded size of one expert matrix.
  QwenTensorRef ffn_gate_inp;        // router, [hidden x n_experts] F32
  QwenTensorRef ffn_gate_inp_shexp;  // shared-expert gate, [hidden] F32
  QwenTensorRef ffn_gate_exps;       // [hidden x expert_ff x n_experts]
  QwenTensorRef ffn_up_exps;         // [hidden x expert_ff x n_experts]
  QwenTensorRef ffn_down_exps;       // [expert_ff x hidden x n_experts]
  QwenTensorRef ffn_gate_shexp;      // [hidden x shared_ff]
  QwenTensorRef ffn_up_shexp;        // [hidden x shared_ff]
  QwenTensorRef ffn_down_shexp;      // [shared_ff x hidden]
};

/// Per-layer stack attributes for the flint attention-only stack (F-B1).
/// Flint runs attention on every layer; these switches select the attention
/// variant and the FFN kind. No flint layer is an SSM layer, and nothing in
/// the flint load path allocates recurrent state.
struct FlintLayerAttributes {
  /// sliding_window_pattern[i]: the layer attends through the sliding window
  /// (F-D2 bounds the key range); false means global attention.
  bool windowed = false;
  /// rope_pattern[i]: the layer applies RoPE; false means NoPE (F-D3).
  bool rope = false;
  /// i < leading_dense_block_count: plain SwiGLU FFN instead of MoE (F-B2).
  bool dense_ffn = false;
};

/// Tensor weight references for one flint layer: attention tensors on every
/// layer, dense SwiGLU on the lead layers, sigmoid-router MoE plus one
/// shared expert on the rest (F-C1 inventory).
struct FlintLayerWeights {
  FlintLayerAttributes attrs;

  QwenTensorRef attn_norm;
  QwenTensorRef attn_q;
  QwenTensorRef attn_k;
  QwenTensorRef attn_v;
  QwenTensorRef attn_output;
  QwenTensorRef attn_q_norm;
  QwenTensorRef attn_k_norm;
  /// Learned f32 sink logit per query head ([num_attention_heads]); added to
  /// the attention scores before softmax (F-D1).
  QwenTensorRef attn_sinks;

  QwenTensorRef ffn_norm;

  // Dense SwiGLU FFN (attrs.dense_ffn only).
  QwenTensorRef ffn_gate;
  QwenTensorRef ffn_up;
  QwenTensorRef ffn_down;

  // MoE (attrs.dense_ffn == false only): sigmoid router over expert_count,
  // expert stack, and one ungated shared expert (F-E1/F-E3).
  QwenTensorRef ffn_gate_inp;
  QwenTensorRef ffn_gate_exps;
  QwenTensorRef ffn_up_exps;
  QwenTensorRef ffn_down_exps;
  QwenTensorRef ffn_gate_shexp;
  QwenTensorRef ffn_up_shexp;
  QwenTensorRef ffn_down_shexp;
};

/// Flint tensor references mapped directly from GGUF storage.
struct FlintModelWeights {
  core::ModelConfig config;
  QwenTensorRef token_embd;
  std::vector<FlintLayerWeights> layers;
  QwenTensorRef output_norm;
  QwenTensorRef output;

  /// Loads and binds the exact flint tensor inventory from a GgufReader
  /// (F-C1). Every tensor the stack needs is required (F-C4); the error
  /// names the layer and tensor on any miss.
  [[nodiscard]] static std::optional<FlintModelWeights> LoadFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

/// Full model tensor references mapped directly from GGUF storage.
struct QwenModelWeights {
  core::ModelConfig config;
  QwenTensorRef token_embd;
  std::vector<QwenLayerWeights> layers;
  QwenTensorRef output_norm;
  QwenTensorRef output;

  /// Loads and binds weights from a GgufReader.
  [[nodiscard]] static std::optional<QwenModelWeights> LoadFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

/// Contiguous Key-Value Cache for auto-regressive generation.
class QwenKvCache {
public:
  QwenKvCache(std::uint32_t num_layers, std::uint32_t num_kv_heads,
              std::uint32_t max_context, std::uint32_t head_dim);

  void Reset() noexcept { current_pos_ = 0; }
  [[nodiscard]] std::uint32_t GetCurrentPos() const noexcept {
    return current_pos_;
  }
  void AdvancePos() noexcept { ++current_pos_; }

  /// Stores K/V through the f16 grid (round-to-nearest-even) on write —
  /// mirrors the GPU's f16 KV twin planes (GUFO_FLINT_KV_F16) so the CPU
  /// oracle computes the same math and the G3 gate keeps its teeth. Only
  /// the flint write path (FlintWriteKv) honors this; the qwen forward
  /// writes through the slices directly and is unaffected.
  void SetF16Rounding(bool on) noexcept { f16_rounding_ = on; }
  [[nodiscard]] bool StoresF16() const noexcept { return f16_rounding_; }

  [[nodiscard]] std::span<float> GetKeySlice(std::uint32_t layer,
                                             std::uint32_t kv_head,
                                             std::uint32_t pos) noexcept;
  [[nodiscard]] std::span<const float> GetKeySlice(
      std::uint32_t layer, std::uint32_t kv_head,
      std::uint32_t pos) const noexcept;

  [[nodiscard]] std::span<float> GetValueSlice(std::uint32_t layer,
                                               std::uint32_t kv_head,
                                               std::uint32_t pos) noexcept;
  [[nodiscard]] std::span<const float> GetValueSlice(
      std::uint32_t layer, std::uint32_t kv_head,
      std::uint32_t pos) const noexcept;

private:
  std::uint32_t num_layers_;
  std::uint32_t num_kv_heads_;
  std::uint32_t max_context_;
  std::uint32_t head_dim_;
  std::uint32_t current_pos_{0};
  std::vector<float> k_data_;
  std::vector<float> v_data_;
  bool f16_rounding_{false};
};

/// Preallocated activation scratch arena for zero-heap-allocation decode
/// passes.
struct QwenScratchArena {
  explicit QwenScratchArena(const core::ModelConfig& config);

  std::vector<float> buffer;
  std::span<float> hidden;
  std::span<float> normed;
  std::span<float> q;
  std::span<float> k;
  std::span<float> v;
  std::span<float> attn_scores;
  std::span<float> attn_out;
  std::span<float> mlp_gate;
  std::span<float> mlp_up;
  std::span<float> mlp_act;
  std::span<float> mlp_out;
  std::span<float> ssm_qkv;
  std::span<float> ssm_gate;
  std::span<float> ssm_out_buf;
  std::span<float> logits;

  // MoE scratch (empty spans on dense models).
  std::span<float> moe_router_logits;
  std::span<std::uint32_t> moe_expert_ids;
  std::span<float> moe_expert_weights;
  std::span<float> moe_expert_gate;
  std::span<float> moe_expert_up;
  std::span<float> moe_expert_act;
  std::span<float> moe_expert_out;
  std::span<float> moe_shexp_gate;
  std::span<float> moe_shexp_up;
  std::span<float> moe_shexp_act;
  std::span<float> moe_shexp_out;
  std::vector<std::uint32_t> moe_ids_storage;
};

/// Preallocated activation scratch for the flint attention-only stack
/// (F-B1). Mirrors QwenScratchArena minus every SSM span; adds the MoE
/// routing and expert scratch. A flint decode step never touches recurrent
/// state, so no such span exists here.
struct FlintScratchArena {
  explicit FlintScratchArena(const core::ModelConfig& config);

  std::vector<float> buffer;
  std::span<float> hidden;
  std::span<float> normed;
  std::span<float> q;
  std::span<float> k;
  std::span<float> v;
  /// Full sequence scores: global layers attend over every position, so
  /// this is sized to the context, not the window.
  std::span<float> attn_scores;
  /// Attention context (num_heads * head_dim), pre-O-projection. Flint's
  /// attention size (4096) exceeds the hidden size, so this is not the
  /// hidden-width buffer the qwen arena uses.
  std::span<float> attn_out;
  /// Post-O-projection activation (hidden width), fed to the residual add.
  std::span<float> attn_proj;
  /// Sigmoid router logits and the renormalised top-k weights (F-E1).
  std::span<float> router_probs;
  std::span<float> expert_gate;
  std::span<float> expert_up;
  std::span<float> expert_act;
  std::span<float> shared_gate;
  std::span<float> shared_up;
  std::span<float> shared_act;
  /// Per-expert down-projection output (hidden width), accumulated into
  /// mlp_out with the renormalised routing weight.
  std::span<float> expert_out;
  std::span<float> mlp_out;
  std::span<float> logits;
};

}  // namespace gufo::models

#endif  // GUFO_MODELS_QWEN_STATE_HPP_
