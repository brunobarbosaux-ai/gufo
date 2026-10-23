#ifndef GUFO_MODELS_QWEN_FLINT_FORWARD_HPP_
#define GUFO_MODELS_QWEN_FLINT_FORWARD_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/core/model_config.hpp"
#include "src/models/qwen/flint_attention.hpp"
#include "src/models/qwen/state.hpp"

namespace gufo::models {

/// CPU reference forward for the flint stack (F-D5 attention wiring, F-E1
/// routing). Autoregressive only: no SSM state, no MTP, no draft. This is
/// the D4 slow-but-correct oracle the HIP paths are verified against.

/// One transformer layer: pre-norm, QKV, QK-norm, RoPE only when the layer's
/// policy says rope (F-D3 NoPE), KV write, sink+window attention, O-projection,
/// residual, then the dense SwiGLU FFN or the sigmoid top-k MoE with the
/// ungated shared expert.
void FlintForwardLayer(std::span<float> hidden, const FlintLayerWeights& layer,
                       const core::ModelConfig& config, QwenKvCache& kv,
                       std::uint32_t layer_idx, std::uint32_t pos,
                       FlintScratchArena& arena) noexcept;

/// Full forward for one token: embedding, all layers, final norm, LM head.
void FlintForwardModel(std::uint32_t token_id, std::uint32_t pos,
                       const FlintModelWeights& weights, QwenKvCache& kv,
                       FlintScratchArena& arena,
                       std::span<float> logits_out) noexcept;

}  // namespace gufo::models

#endif  // GUFO_MODELS_QWEN_FLINT_FORWARD_HPP_
