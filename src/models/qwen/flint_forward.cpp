#include "src/models/qwen/flint_forward.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/forward.hpp"

namespace gufo::models {
namespace {

/// flint.attention.layer_norm_rms_epsilon is 1e-6f bit-exact in the pinned
/// artifact (the GGUF f32 widens to 9.999999974752427e-07). The config does
/// not carry the key yet; when it lands, read it instead.
constexpr float kFlintRmsEps = 1e-6F;

/// One expert's contiguous slab inside a 3D [hidden, expert_ff, n_experts]
/// (or transposed) tensor. Slab sizes are block multiples, so the element
/// offset is block-aligned and EncodedSizeBytes gives the byte offset.
[[nodiscard]] QwenTensorRef ExpertSlab(const QwenTensorRef& tensor,
                                       std::size_t expert,
                                       std::size_t slab_elements) noexcept {
  QwenTensorRef slab;
  slab.type = tensor.type;
  slab.num_elements = slab_elements;
  const std::size_t byte_offset =
      quant::EncodedSizeBytes(tensor.type, expert * slab_elements);
  slab.data = static_cast<const std::uint8_t*>(tensor.data) + byte_offset;
  slab.available_bytes =
      tensor.available_bytes == std::numeric_limits<std::size_t>::max()
          ? tensor.available_bytes
          : tensor.available_bytes - byte_offset;
  return slab;
}

/// SwiGLU activation matching the ffn module's CPU backend exactly.
void FlintSwiglu(std::span<const float> gate, std::span<const float> up,
                 std::span<float> act) noexcept {
  for (std::size_t i = 0; i < act.size(); ++i) {
    const float g = gate[i];
    const float silu_g = g / (1.0F + std::exp(-g));
    act[i] = silu_g * up[i];
  }
}

/// Sigmoid router over the full expert set, top-k, renormalised, scaled;
/// expert stack accumulation plus the ungated shared expert at weight 1.0
/// (F-E1/F-E2/F-E3 CPU reference).
void FlintMoeForward(std::span<const float> x, const FlintLayerWeights& layer,
                     const core::ModelConfig& config,
                     FlintScratchArena& arena) noexcept {
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t expert_count = config.expert_count;
  const std::size_t used = config.expert_used_count;
  const std::size_t expert_ff = config.expert_feed_forward_length;
  const std::size_t shared_ff = config.expert_shared_feed_forward_length;
  const std::size_t expert_slab = expert_ff * hidden_size;

  // Router logits -> sigmoid over every expert (gating func 2).
  TensorGEMV(layer.ffn_gate_inp, x, expert_count, hidden_size,
             arena.router_probs);
  for (std::size_t e = 0; e < expert_count; ++e) {
    const float z = arena.router_probs[e];
    arena.router_probs[e] = 1.0F / (1.0F + std::exp(-z));
  }

  // Top-k selection: highest probability wins; ties go to the lower index
  // so the choice is deterministic. (Oracle tie-break confirmed at G3.)
  std::uint32_t selected[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float selected_weight[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  for (std::size_t e = 0; e < expert_count; ++e) {
    if (e < used) {
      selected[e] = static_cast<std::uint32_t>(e);
      selected_weight[e] = arena.router_probs[e];
      // Bubble the new entry down to keep the window sorted descending.
      for (std::size_t i = e; i > 0 && selected_weight[i] > selected_weight[i - 1];
           --i) {
        std::swap(selected_weight[i], selected_weight[i - 1]);
        std::swap(selected[i], selected[i - 1]);
      }
    } else if (arena.router_probs[e] > selected_weight[used - 1]) {
      selected[used - 1] = static_cast<std::uint32_t>(e);
      selected_weight[used - 1] = arena.router_probs[e];
      for (std::size_t i = used - 1;
           i > 0 && selected_weight[i] > selected_weight[i - 1]; --i) {
        std::swap(selected_weight[i], selected_weight[i - 1]);
        std::swap(selected[i], selected[i - 1]);
      }
    }
  }

  double weight_sum = 0.0;
  for (std::size_t i = 0; i < used; ++i) {
    weight_sum += selected_weight[i];
  }

  std::ranges::fill(arena.mlp_out, 0.0F);
  for (std::size_t i = 0; i < used; ++i) {
    const std::size_t expert = selected[i];
    double weight = selected_weight[i];
    if (config.expert_weights_norm && weight_sum > 0.0) {
      weight /= weight_sum;
    }
    weight *= static_cast<double>(config.expert_weights_scale);

    TensorGEMV(ExpertSlab(layer.ffn_gate_exps, expert, expert_slab), x,
               expert_ff, hidden_size, arena.expert_gate);
    TensorGEMV(ExpertSlab(layer.ffn_up_exps, expert, expert_slab), x, expert_ff,
               hidden_size, arena.expert_up);
    FlintSwiglu(arena.expert_gate.subspan(0, expert_ff),
                arena.expert_up.subspan(0, expert_ff),
                arena.expert_act.subspan(0, expert_ff));
    // down_exps is [expert_ff, hidden, n_experts]: rows of expert_ff.
    TensorGEMV(ExpertSlab(layer.ffn_down_exps, expert, expert_slab),
               arena.expert_act.subspan(0, expert_ff), hidden_size, expert_ff,
               arena.expert_out);
    for (std::size_t h = 0; h < hidden_size; ++h) {
      arena.mlp_out[h] = static_cast<float>(
          static_cast<double>(arena.mlp_out[h]) +
          weight * static_cast<double>(arena.expert_out[h]));
    }
  }

  // Shared expert: ungated, weight 1.0 (F-E3 — no synthetic gate tensor).
  ForwardFFN(x, layer.ffn_gate_shexp, layer.ffn_up_shexp,
             layer.ffn_down_shexp, hidden_size, shared_ff, arena.shared_gate,
             arena.shared_up, arena.shared_act, arena.expert_out);
  for (std::size_t h = 0; h < hidden_size; ++h) {
    arena.mlp_out[h] += arena.expert_out[h];
  }
}

void FlintWriteKv(const std::span<float>& k, const std::span<float>& v,
                  QwenKvCache& kv, std::uint32_t layer_idx,
                  std::uint32_t num_kv_heads, std::uint32_t head_dim,
                  std::uint32_t pos) noexcept {
  for (std::uint32_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
    const auto k_src = k.subspan(static_cast<std::size_t>(kv_h) * head_dim,
                                 head_dim);
    const auto v_src = v.subspan(static_cast<std::size_t>(kv_h) * head_dim,
                                 head_dim);
    std::ranges::copy(k_src, kv.GetKeySlice(layer_idx, kv_h, pos).begin());
    std::ranges::copy(v_src, kv.GetValueSlice(layer_idx, kv_h, pos).begin());
  }
}

}  // namespace

void FlintForwardLayer(std::span<float> hidden, const FlintLayerWeights& layer,
                       const core::ModelConfig& config, QwenKvCache& kv,
                       std::uint32_t layer_idx, std::uint32_t pos,
                       FlintScratchArena& arena) noexcept {
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;

  // 1. Pre-RMSNorm.
  ForwardRMSNorm(hidden, layer.attn_norm, kFlintRmsEps, arena.normed);

  // 2. Q/K/V projections (plain: flint has no query gating).
  TensorGEMV(layer.attn_q, arena.normed, attention_size, hidden_size, arena.q);
  TensorGEMV(layer.attn_k, arena.normed, kv_size, hidden_size, arena.k);
  TensorGEMV(layer.attn_v, arena.normed, kv_size, hidden_size, arena.v);

  // 3. Per-head Q/K RMSNorm.
  for (std::uint32_t h = 0; h < config.num_attention_heads; ++h) {
    auto q_head = arena.q.subspan(static_cast<std::size_t>(h) *
                                      config.head_dim,
                                  config.head_dim);
    ForwardRMSNorm(q_head, layer.attn_q_norm, kFlintRmsEps, q_head);
  }
  for (std::uint32_t h = 0; h < config.num_key_value_heads; ++h) {
    auto k_head = arena.k.subspan(static_cast<std::size_t>(h) *
                                      config.head_dim,
                                  config.head_dim);
    ForwardRMSNorm(k_head, layer.attn_k_norm, kFlintRmsEps, k_head);
  }

  // 4. RoPE only on rope_pattern layers (F-D3: NoPE on the others).
  if (layer.attrs.rope) {
    ForwardRoPE(arena.q, arena.k, config.num_attention_heads,
                config.num_key_value_heads, config.head_dim,
                config.rotary_dim, pos, config.rope_theta);
  }

  // 5. KV cache write, then sink + window attention (F-D1/F-D2, F-D6).
  FlintWriteKv(arena.k, arena.v, kv, layer_idx, config.num_key_value_heads,
               config.head_dim, pos);

  FlintAttentionPolicy policy;
  const auto sink_span = layer.attn_sinks.AsFloatSpan();
  policy.sinks = sink_span.empty() ? nullptr : sink_span.data();
  policy.num_heads = config.num_attention_heads;
  policy.num_kv_heads = config.num_key_value_heads;
  policy.head_dim = config.head_dim;
  policy.windowed = layer.attrs.windowed;
  policy.window = config.sliding_window;
  policy.rope = layer.attrs.rope;
  FlintAttentionReference(policy, arena.q.data(), kv, layer_idx, pos,
                          arena.attn_scores.data(), arena.attn_out.data());

  // 6. O-projection + residual.
  TensorGEMV(layer.attn_output, arena.attn_out, hidden_size, attention_size,
             arena.attn_proj);
  for (std::size_t i = 0; i < hidden_size; ++i) {
    hidden[i] += arena.attn_proj[i];
  }

  // 7. FFN: dense SwiGLU on the lead layers, MoE elsewhere.
  ForwardRMSNorm(hidden, layer.ffn_norm, kFlintRmsEps, arena.normed);
  if (layer.attrs.dense_ffn) {
    ForwardFFN(arena.normed, layer.ffn_gate, layer.ffn_up, layer.ffn_down,
               hidden_size, config.intermediate_size, arena.expert_gate,
               arena.expert_up, arena.expert_act, arena.mlp_out);
  } else {
    FlintMoeForward(arena.normed, layer, config, arena);
  }
  for (std::size_t i = 0; i < hidden_size; ++i) {
    hidden[i] += arena.mlp_out[i];
  }
}

void FlintForwardModel(std::uint32_t token_id, std::uint32_t pos,
                       const FlintModelWeights& weights, QwenKvCache& kv,
                       FlintScratchArena& arena,
                       std::span<float> logits_out) noexcept {
  ForwardEmbedding(token_id, weights.token_embd, weights.config.hidden_size,
                   arena.hidden);
  for (std::uint32_t l = 0; l < weights.config.num_layers; ++l) {
    FlintForwardLayer(arena.hidden, weights.layers[l], weights.config, kv, l,
                      pos, arena);
  }
  ForwardRMSNorm(arena.hidden, weights.output_norm, kFlintRmsEps,
                 arena.normed);
  TensorGEMV(weights.output, arena.normed, weights.config.vocab_size,
             weights.config.hidden_size, logits_out);
}

}  // namespace gufo::models
