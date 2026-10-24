#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_FLINT_PREFILL_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_FLINT_PREFILL_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/qwen/hip/detail/flint_decode_step.hpp"
#include "src/models/qwen/state.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Maximum tokens one chunked-prefill call consumes. The expert buckets grow
/// with the chunk (roughly chunk * used / expert_count rows each), which is
/// what makes the grouped expert kernels worth their launch: at 512 tokens a
/// bucket holds about twelve rows, so a decoded weight block feeds twelve
/// tokens instead of one.
inline constexpr std::size_t kFlintPrefillBatchCapacity = 512;

/// Upper bound on the slot tables for a chunk of `tokens`: the compact slot
/// count plus the 16-row bucket padding the grouped WMMA route requires (at
/// most 15 rows per expert), rounded up to a whole number of tiles so
/// `slot_capacity / 16` is the exact host bound on the tile count.
[[nodiscard]] inline constexpr std::size_t kFlintPrefillSlotCapacity(
    std::size_t tokens, std::size_t used, std::size_t expert_count) {
  const std::size_t raw = tokens * used + 15 * expert_count;
  return (raw + 15) & ~static_cast<std::size_t>(15);
}

/// Highest absolute position a batched chunk may cover. The batched attention
/// score row has to fit the gfx1151 LDS budget (64 KiB minus 256 floats), and
/// the global layers cover the full causal prefix; chunks that would cross
/// this position are handed to the token-by-token path instead.
inline constexpr std::size_t kFlintPrefillMaxPosition = 16000;

/// Chunk-prefill scratch. Sized once per session from the config; the decode
/// buffers (FlintGpuStepBuffers) are reused only for the KV cache, the logits
/// row and the argmax slot.
struct FlintPrefillBuffers {
  std::size_t batch_capacity = 0;  ///< rows this scratch holds
  std::size_t slot_capacity = 0;   ///< kFlintPrefillSlotCapacity(batch, used, E)

  float* hidden = nullptr;      ///< [cap, H] residual stream
  float* normed = nullptr;      ///< [cap, H] post-norm rows
  float* q = nullptr;           ///< [cap, A]
  float* k = nullptr;           ///< [cap, S]
  float* v = nullptr;           ///< [cap, S]
  float* attn_out = nullptr;    ///< [cap, A]
  float* attn_proj = nullptr;   ///< [cap, H]
  float* mlp_out = nullptr;     ///< [cap, H] FFN/MoE output before residual
  float* shared_out = nullptr;  ///< [cap, H] shared-expert output
  float* ffn_gate = nullptr;    ///< [cap, max(dense ff, shared ff)]
  float* ffn_up = nullptr;      ///< [cap, max(dense ff, shared ff)]
  float* router_logits = nullptr;  ///< [cap, E]

  std::uint32_t* selected = nullptr;  ///< [cap, used]
  float* sel_weights = nullptr;       ///< [cap, used]
  int* expert_counts = nullptr;       ///< [E]
  int* expert_offsets = nullptr;      ///< [E + 1] (padded starts; [E] = total)
  int* expert_cursors = nullptr;      ///< [E]
  int* slot_tokens = nullptr;         ///< [slots]
  float* slot_weights = nullptr;      ///< [slots]
  int* token_slots = nullptr;         ///< [cap, used]
  float* act_slots = nullptr;         ///< [slots, expert_ff]
  float* down_slots = nullptr;        ///< [slots, H]
  float* up_slots = nullptr;          ///< [slots, expert_ff] (WMMA route)

  int* tile_expert = nullptr;  ///< [slots / 16] tile -> expert (WMMA route)
  int* tile_slot0 = nullptr;   ///< [slots / 16] tile -> first slot
  int* tile_count = nullptr;   ///< [1] live tile count

  std::uint32_t* d_tokens = nullptr;  ///< [cap] device staging for token ids

  void* q8_act = nullptr;     ///< tiled Q8_1 activation staging
  void* q8_normed = nullptr;  ///< tiled Q8_1 of normed (shared by all readers)
  void* slot_q8 = nullptr;    ///< tiled Q8_1 gathered to slot order
  void* act_q8 = nullptr;     ///< tiled Q8_1 of the SwiGLU slot rows
};

/// True when every layer takes the batched routes (quantised projections,
/// one router type both stacks decode, grouped SwiGLU with matching gate/up
/// types). The runner falls back to the token-by-token path when false.
[[nodiscard]] bool FlintPrefillSupported(
    const models::FlintModelWeights& weights) noexcept;

/// Runs one chunked-prefill pass: embedding -> every layer over all `tokens`
/// -> final norm -> unembed (last row only). The KV cache is written position
/// by position into the caller's step buffers, so a following decode step at
/// `start_pos + tokens.size()` continues the same sequence.
///
/// `compute_logits` writes the last row's logits into `buffers.logits` and its
/// argmax into `buffers.d_out_token`, exactly like ExecuteFlintDecodeStep.
///
/// When `timings` is non-null (tuning item 0.3) each stage group is bracketed
/// with stream events and its GPU time accumulated into the matching field.
/// The bracketed launches are unchanged, but the event records serialize the
/// sections slightly, so request it only for profiling; the serve path passes
/// nullptr and pays nothing.
struct FlintPrefillTimings {
  double norm_ms = 0;       // RMSNorms + the shared Q8_1 quantize of normed
  double qkv_ms = 0;        // QKV projections
  double attention_ms = 0;  // QK norms + RoPE + batched attention
  double oproj_ms = 0;      // O projection + residual add
  double router_ms = 0;     // FFN norm + router + top-k + slot compaction
  double shared_ms = 0;     // shared-expert GEMMs + SwiGLU
  double routed_ms = 0;     // routed experts (gather/GEMMs/combine)
  double dense_ms = 0;      // dense-FFN layers
  double tail_ms = 0;       // embedding + final norm + unembed

  [[nodiscard]] double total_ms() const {
    return norm_ms + qkv_ms + attention_ms + oproj_ms + router_ms +
           shared_ms + routed_ms + dense_ms + tail_ms;
  }
};

void ExecuteFlintPrefillChunk(const models::FlintModelWeights& weights,
                              std::span<const std::uint32_t> tokens,
                              std::uint32_t start_pos, bool compute_logits,
                              const FlintGpuStepBuffers& buffers,
                              const FlintPrefillBuffers& prefill,
                              FlintPrefillTimings* timings = nullptr);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_FLINT_PREFILL_HPP_
