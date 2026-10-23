#ifndef GUFO_MODELS_QWEN_FLINT_ATTENTION_HPP_
#define GUFO_MODELS_QWEN_FLINT_ATTENTION_HPP_

#include <cstdint>

#include "src/models/qwen/state.hpp"

namespace gufo::models {

/// Per-layer flint attention policy carried by every execution path (F-D1,
/// F-D2, F-D3): the CPU reference, decode, split-k, graph, batched, tile,
/// WMMA, verify, and the fused norm/rope/kv kernel. One instance per layer;
/// every path must consume every field, and each path has a test asserting
/// the fields are honored.
///
/// The numerical contract (F-D6, pinned to the oracle):
///   scale -> sink -> softmax -> weighted sum
/// The sink logit joins the running max and the softmax denominator with a
/// weight of exp(sink - max) but contributes no value vector.
struct FlintAttentionPolicy {
  /// Learned f32 sink logit per query head ([num_heads]); null when the
  /// artifact carries none (never for flint, required by F-C1).
  const float* sinks = nullptr;
  std::uint32_t num_heads = 0;
  std::uint32_t num_kv_heads = 0;
  std::uint32_t head_dim = 0;

  /// sliding_window_pattern[i]: true bounds the key range to the window.
  bool windowed = false;
  /// flint.attention.sliding_window (513); only read when windowed.
  std::uint32_t window = 0;
  /// rope_pattern[i]: false means NoPE — RoPE must not be applied.
  bool rope = false;

  /// First key position a query at `query_pos` may attend to (F-D2).
  /// Global layers and degenerate windows see the whole prefix.
  [[nodiscard]] std::uint32_t KeyRangeStart(std::uint32_t query_pos) const {
    if (!windowed || window == 0) {
      return 0;
    }
    return query_pos + 1 > window ? query_pos + 1 - window : 0;
  }
};

/// CPU reference attention for the flint stack (F-D5): GQA with per-head
/// sinks, windowed key ranges, and the F-D6 numerical order. Writes the
/// attention output (pre-O-projection, num_heads * head_dim floats) into
/// `out`. `attn_scores_scratch` needs (pos - KeyRangeStart(pos) + 1) floats.
/// K/V for positions [KeyRangeStart(pos), pos] must already be in `kv`.
///
/// q: [num_heads * head_dim] (post q-norm, post-RoPE when policy.rope).
void FlintAttentionReference(const FlintAttentionPolicy& policy,
                             const float* q, const QwenKvCache& kv,
                             std::uint32_t layer, std::uint32_t pos,
                             float* attn_scores_scratch, float* out) noexcept;

}  // namespace gufo::models

#endif  // GUFO_MODELS_QWEN_FLINT_ATTENTION_HPP_
