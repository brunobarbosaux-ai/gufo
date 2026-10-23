#ifndef GUFO_CORE_MODEL_CONFIG_HPP_
#define GUFO_CORE_MODEL_CONFIG_HPP_

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::core {

/// Unified architectural configuration parsed dynamically from GGUF metadata or
/// JSON.
struct ModelConfig {
  std::string architecture{"qwen35"};
  std::string model_name{"qwen3.5-4b-text"};
  std::uint32_t num_layers{36};  ///< Main transformer layers, excluding MTP.
  std::uint32_t hidden_size{2560};
  std::uint32_t intermediate_size{9728};  ///< 17408 for Qwen3.8-27B.
  std::uint32_t num_attention_heads{16};
  std::uint32_t num_key_value_heads{4};
  std::uint32_t head_dim{256};
  std::uint32_t vocab_size{248320};  ///< 248320 for Qwen3.5/3.8
  std::uint32_t context_length{32768};
  std::uint32_t full_attention_interval{
      4};  ///< 3 linear attention + 1 full attention
  std::uint32_t mtp_num_layers{1};
  std::uint32_t ssm_conv_kernel{4};
  std::uint32_t ssm_state_size{128};
  std::uint32_t ssm_group_count{16};
  std::uint32_t ssm_time_step_rank{32};
  std::uint32_t ssm_inner_size{4096};
  std::uint32_t rotary_dim{64};
  float rope_theta{10000000.0F};
  float rope_scale{1.0F};
  bool is_text_only{true};

  // Mixture-of-experts geometry (flint.*). Zero-valued for dense artifacts;
  // only judged when expert_count > 0.
  std::uint32_t expert_count{0};
  std::uint32_t expert_used_count{0};
  std::uint32_t expert_feed_forward_length{0};
  std::uint32_t expert_shared_count{0};
  std::uint32_t expert_shared_feed_forward_length{0};
  std::uint32_t expert_gating_func{
      0};  ///< 2 = sigmoid over the full expert set
  float expert_weights_scale{1.0F};
  bool expert_weights_norm{false};
  std::uint32_t leading_dense_block_count{0};

  // Per-layer attention switches (flint.attention.*). Required for flint:
  // sliding_window_pattern[i] = true means layer i attends through the
  // sliding window, false means full attention; rope_pattern[i] = true means
  // the layer applies RoPE, false means NoPE. Read independently (F-A2) even
  // when an artifact carries identical arrays.
  std::uint32_t sliding_window{0};
  std::vector<bool> sliding_window_pattern;
  std::vector<bool> rope_pattern;

  [[nodiscard]] constexpr std::uint32_t AttentionSize() const noexcept {
    return num_attention_heads * head_dim;
  }

  [[nodiscard]] constexpr std::uint32_t FullAttentionLayerCount()
      const noexcept {
    return full_attention_interval == 0 ? 0
                                        : num_layers / full_attention_interval;
  }

  [[nodiscard]] constexpr std::uint32_t SsmLayerCount() const noexcept {
    return num_layers - FullAttentionLayerCount();
  }

  /// Dense state index for a recurrent transformer layer.
  [[nodiscard]] constexpr std::uint32_t SsmLayerIndex(
      std::uint32_t layer) const noexcept {
    return full_attention_interval == 0
               ? layer
               : layer - layer / full_attention_interval;
  }

  [[nodiscard]] constexpr std::uint32_t SsmValueSize() const noexcept {
    return ssm_time_step_rank == 0 ? 0 : ssm_inner_size / ssm_time_step_rank;
  }

  [[nodiscard]] constexpr std::uint32_t SsmQkvSize() const noexcept {
    return (2 * ssm_group_count * ssm_state_size) + ssm_inner_size;
  }

  [[nodiscard]] constexpr bool IsFlintArch() const noexcept {
    return architecture == "flint";
  }

  [[nodiscard]] constexpr bool IsMoe() const noexcept {
    return expert_count > 0;
  }

  /// Sliding-window membership; out-of-range layers are full attention.
  [[nodiscard]] bool IsWindowedLayer(std::uint32_t layer) const noexcept {
    return layer < sliding_window_pattern.size() &&
           sliding_window_pattern[layer];
  }

  /// RoPE membership; out-of-range layers apply rope (conservative default).
  [[nodiscard]] bool IsRopeLayer(std::uint32_t layer) const noexcept {
    return layer < rope_pattern.size() && rope_pattern[layer];
  }

  /// Returns true if this configuration conforms to the flint structural
  /// rules. Every rejection writes a message naming the offending metadata
  /// key so load failures are diagnosable without a debugger.
  bool IsValidFlint(std::string* error_msg = nullptr) const {
    const auto reject = [error_msg](const std::string& message) {
      if (error_msg != nullptr) {
        *error_msg = message;
      }
      return false;
    };
    if (num_layers == 0) {
      return reject("flint.block_count: must be positive");
    }
    if (hidden_size == 0 || intermediate_size == 0) {
      return reject(
          "flint.embedding_length and flint.feed_forward_length: must be "
          "positive");
    }
    if (num_attention_heads == 0 || num_key_value_heads == 0) {
      return reject(
          "flint.attention.head_count and flint.attention.head_count_kv: "
          "must be positive");
    }
    if (head_dim == 0) {
      return reject("flint.attention.key_length: must be positive");
    }
    if (vocab_size == 0 || context_length == 0) {
      return reject("flint.context_length: must be positive");
    }
    if (sliding_window_pattern.size() != num_layers) {
      return reject("flint.attention.sliding_window_pattern: expected " +
                    std::to_string(num_layers) + " entries, got " +
                    std::to_string(sliding_window_pattern.size()));
    }
    if (rope_pattern.size() != num_layers) {
      return reject("flint.attention.rope_pattern: expected " +
                    std::to_string(num_layers) + " entries, got " +
                    std::to_string(rope_pattern.size()));
    }
    if (IsMoe()) {
      if (expert_used_count == 0 || expert_used_count > expert_count) {
        return reject(
            "flint.expert_used_count: must be in (0, flint.expert_count]");
      }
      if (expert_feed_forward_length == 0) {
        return reject(
            "flint.expert_feed_forward_length: must be positive when "
            "flint.expert_count > 0");
      }
      if (expert_shared_count > 0 && expert_shared_feed_forward_length == 0) {
        return reject(
            "flint.expert_shared_feed_forward_length: must be positive when "
            "flint.expert_shared_count > 0");
      }
      if (expert_gating_func != 2) {
        // Fail closed: an unknown gating function must never fall back to
        // softmax silently.
        return reject(
            "flint.expert_gating_func: only 2 (sigmoid) is "
            "supported, got " +
            std::to_string(expert_gating_func));
      }
      if (leading_dense_block_count > num_layers) {
        return reject(
            "flint.leading_dense_block_count: exceeds flint.block_count");
      }
    }
    return true;
  }

  /// Returns true if this configuration conforms to the Qwen3.5/3.8 repeating
  /// block structure.
  [[nodiscard]] constexpr bool IsValidQwen() const noexcept {
    return head_dim > 0 && num_layers > 0 && hidden_size > 0 &&
           intermediate_size > 0 && num_attention_heads > 0 &&
           num_key_value_heads > 0 && vocab_size > 0 &&
           full_attention_interval == 4 &&
           (num_layers % full_attention_interval) == 0 && rotary_dim > 0 &&
           rotary_dim <= head_dim && (rotary_dim % 2) == 0 &&
           ssm_conv_kernel == 4 && ssm_state_size == 128 &&
           ssm_group_count > 0 && ssm_time_step_rank > 0 &&
           ssm_inner_size > 0 && (ssm_inner_size % ssm_time_step_rank) == 0 &&
           SsmValueSize() == 128 && is_text_only;
  }
};

}  // namespace gufo::core

#endif  // GUFO_CORE_MODEL_CONFIG_HPP_
