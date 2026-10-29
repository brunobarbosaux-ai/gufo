// Flint speculative target (7.7): adapts the flint arena + decode/prefill/verify
// path to the repository's ISpeculativeTargetExecutor interface, so the DFlash-2
// draft backend can drive it through SpeculativeVerifier exactly as it drives
// QwenGpuExecutor. The target owns no weights: it borrows the immutable
// FlintGpuModel and the per-session FlintGpuArena the runner already holds.
#ifndef GUFO_MODELS_QWEN_HIP_FLINT_SPECULATIVE_TARGET_HPP_
#define GUFO_MODELS_QWEN_HIP_FLINT_SPECULATIVE_TARGET_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "src/core/speculative/speculative_verifier.hpp"
#include "src/models/qwen/hip/flint_gpu_model.hpp"
#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)

namespace gufo::hip {

/// Target executor over one flint session. The arena owns the step/prefill
/// scratch and the fp32 KV cache; this class tracks the absolute position and
/// the DFlash-2 target-layer capture, and exposes the batched verify tail
/// (ExecuteFlintVerifyChunk) as ForwardVerificationChunk.
///
/// SaveState/RestoreState need no KV copy: the flint KV is append-only and
/// every attention launch is bounded by its explicit position, so a rejected
/// draft row is simply overwritten by the next forward pass. The ring cache's
/// windowed layers keep `sliding_window + prefill chunk` rows, so the <=16
/// draft rows written past the saved position never alias the live window.
class FlintGpuSpeculativeTarget final
    : public speculative::ISpeculativeTargetExecutor {
 public:
  FlintGpuSpeculativeTarget(
      const FlintGpuModel& model, FlintGpuArena& arena,
      std::shared_ptr<const tokenization::QwenTokenizer> tokenizer);

  void Reset() noexcept override;

  [[nodiscard]] tokenization::TokenId ForwardPromptBatch(
      std::span<const tokenization::TokenId> prompt_tokens) override;
  [[nodiscard]] tokenization::TokenId ForwardPromptSuffix(
      std::span<const tokenization::TokenId> tokens, std::uint32_t position,
      bool compute_logits = true) override;
  [[nodiscard]] tokenization::TokenId ForwardToken(
      tokenization::TokenId token_id, std::uint32_t pos,
      bool compute_logits = true) override;

  void SaveState(std::uint32_t valid_context) override;
  void RestoreState() override;
  void FinishVerification() override {}

  void SetPromptHiddenCapture(
      bool enabled,
      std::span<const std::uint32_t> target_layer_ids = {}) override;
  [[nodiscard]] std::span<const float> GetPromptHiddenStates()
      const noexcept override {
    return prompt_hidden_;
  }
  [[nodiscard]] std::span<const float> CopyLastHidden() override {
    return last_hidden_;
  }
  [[nodiscard]] std::span<const float> CopyLastLogits() override {
    return last_logits_;
  }
  [[nodiscard]] std::span<const float> CopyVerificationLogits(
      std::size_t row) override;

  [[nodiscard]] speculative::VerificationChunkResult ForwardVerificationChunk(
      std::span<const tokenization::TokenId> candidate_tokens,
      std::uint32_t start_pos, bool capture_hidden,
      bool capture_logits) override;
  void CommitVerificationChunk(
      std::span<const tokenization::TokenId> committed_tokens,
      std::uint32_t start_pos) override;

  [[nodiscard]] tokenization::TokenId GetEosTokenId() const noexcept override {
    return tokenizer_->GetEosTokenId();
  }
  [[nodiscard]] bool IsStopToken(
      tokenization::TokenId token) const noexcept override {
    return tokenizer_->IsStopToken(token);
  }
  [[nodiscard]] std::size_t VocabularySize() const noexcept override {
    return tokenizer_->GetVocabSize();
  }
  [[nodiscard]] std::string_view DecodeToken(
      tokenization::TokenId token_id) const noexcept override {
    return tokenizer_->DecodeToken(token_id);
  }

  [[nodiscard]] std::uint32_t position() const noexcept { return position_; }

 private:
  /// Runs the chunked prefill over `tokens` at `position`, capturing the
  /// target-layer residuals when enabled. Returns the last row's argmax.
  [[nodiscard]] tokenization::TokenId Prefill(
      std::span<const tokenization::TokenId> tokens, std::uint32_t position,
      bool compute_logits);
  /// Copies the last row's logits into `last_logits_` and returns its argmax.
  [[nodiscard]] tokenization::TokenId CaptureLogits();

  const FlintGpuModel& model_;
  FlintGpuArena& arena_;
  std::shared_ptr<const tokenization::QwenTokenizer> tokenizer_;
  std::uint32_t position_{0};
  std::uint32_t saved_position_{0};
  bool has_saved_{false};
  bool capture_hidden_{false};
  std::vector<std::uint32_t> target_layer_ids_;
  /// [row][tap][hidden] prompt capture, the draft's feature layout.
  std::vector<float> prompt_hidden_;
  /// [tap][hidden] last decode/verify row capture.
  std::vector<float> last_hidden_;
  std::vector<float> last_logits_;
  /// [row][vocab] verify logits, filled by ForwardVerificationChunk.
  std::vector<float> verification_logits_;
  std::size_t verification_rows_{0};
};

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_FLINT_SPECULATIVE_TARGET_HPP_
