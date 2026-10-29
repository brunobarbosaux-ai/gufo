// Flint speculative target (7.7). See the header for the design notes.
#include "src/models/qwen/hip/flint_speculative_target.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/flint_decode_step.hpp"
#include "src/models/qwen/hip/detail/flint_prefill.hpp"

namespace gufo::hip {

FlintGpuSpeculativeTarget::FlintGpuSpeculativeTarget(
    const FlintGpuModel& model, FlintGpuArena& arena,
    std::shared_ptr<const tokenization::QwenTokenizer> tokenizer)
    : model_(model), arena_(arena), tokenizer_(std::move(tokenizer)) {
  if (tokenizer_ == nullptr) {
    throw std::invalid_argument("flint speculative target needs a tokenizer");
  }
}

void FlintGpuSpeculativeTarget::Reset() noexcept {
  position_ = 0;
  has_saved_ = false;
  prompt_hidden_.clear();
  last_hidden_.clear();
  last_logits_.clear();
  verification_logits_.clear();
  verification_rows_ = 0;
}

tokenization::TokenId FlintGpuSpeculativeTarget::ForwardPromptBatch(
    std::span<const tokenization::TokenId> prompt_tokens) {
  return Prefill(prompt_tokens, 0, /*compute_logits=*/true);
}

tokenization::TokenId FlintGpuSpeculativeTarget::ForwardPromptSuffix(
    std::span<const tokenization::TokenId> tokens, std::uint32_t position,
    bool compute_logits) {
  return Prefill(tokens, position, compute_logits);
}

tokenization::TokenId FlintGpuSpeculativeTarget::Prefill(
    std::span<const tokenization::TokenId> tokens, std::uint32_t position,
    bool compute_logits) {
  const auto& weights = model_.GetWeights();
  const auto buffers = arena_.GetBuffers();
  const auto prefill = arena_.GetPrefillBuffers();
  const std::size_t hidden = weights.config.hidden_size;
  const std::size_t taps = target_layer_ids_.size();
  if (capture_hidden_) {
    prompt_hidden_.clear();
    prompt_hidden_.reserve(tokens.size() * taps * hidden);
  }
  std::size_t done = 0;
  while (done < tokens.size()) {
    const std::size_t take = std::min<std::size_t>(
        kFlintPrefillBatchCapacity, tokens.size() - done);
    const bool final_take = done + take == tokens.size();
    ExecuteFlintPrefillChunk(
        weights, tokens.subspan(done, take),
        position + static_cast<std::uint32_t>(done),
        compute_logits && final_take, buffers, prefill);
    HIP_CHECK(hipStreamSynchronize(buffers.stream));
    if (capture_hidden_) {
      const std::size_t count = take * taps * hidden;
      const std::size_t old = prompt_hidden_.size();
      prompt_hidden_.resize(old + count);
      HIP_CHECK(hipMemcpy(prompt_hidden_.data() + old,
                          prefill.target_layer_features,
                          count * sizeof(float), hipMemcpyDeviceToHost));
    }
    done += take;
  }
  position_ = position + static_cast<std::uint32_t>(tokens.size());
  if (!compute_logits) {
    return 0;
  }
  return CaptureLogits();
}

tokenization::TokenId FlintGpuSpeculativeTarget::CaptureLogits() {
  const auto buffers = arena_.GetBuffers();
  const std::size_t vocab = model_.GetConfig().vocab_size;
  last_logits_.resize(vocab);
  HIP_CHECK(hipMemcpy(last_logits_.data(), buffers.logits,
                      vocab * sizeof(float), hipMemcpyDeviceToHost));
  std::uint32_t token = 0;
  HIP_CHECK(hipMemcpy(&token, buffers.d_out_token, sizeof(std::uint32_t),
                      hipMemcpyDeviceToHost));
  return token;
}

tokenization::TokenId FlintGpuSpeculativeTarget::ForwardToken(
    tokenization::TokenId token_id, std::uint32_t pos, bool compute_logits) {
  const auto& weights = model_.GetWeights();
  const auto buffers = arena_.GetBuffers();
  ExecuteFlintDecodeStep(weights, token_id, pos, compute_logits, buffers,
                         /*timings=*/nullptr, arena_.GetGraphState());
  HIP_CHECK(hipStreamSynchronize(buffers.stream));
  if (capture_hidden_) {
    const std::size_t count =
        target_layer_ids_.size() * weights.config.hidden_size;
    last_hidden_.resize(count);
    HIP_CHECK(hipMemcpy(last_hidden_.data(), buffers.target_layer_features,
                        count * sizeof(float), hipMemcpyDeviceToHost));
  }
  position_ = pos + 1;
  if (!compute_logits) {
    return 0;
  }
  return CaptureLogits();
}

void FlintGpuSpeculativeTarget::SaveState(std::uint32_t valid_context) {
  saved_position_ = valid_context;
  has_saved_ = true;
}

void FlintGpuSpeculativeTarget::RestoreState() {
  if (!has_saved_) {
    throw std::logic_error("flint target state has not been saved");
  }
  position_ = saved_position_;
}

void FlintGpuSpeculativeTarget::SetPromptHiddenCapture(
    bool enabled, std::span<const std::uint32_t> target_layer_ids) {
  capture_hidden_ = enabled;
  target_layer_ids_.assign(target_layer_ids.begin(), target_layer_ids.end());
}

std::span<const float> FlintGpuSpeculativeTarget::CopyVerificationLogits(
    std::size_t row) {
  const std::size_t vocab = model_.GetConfig().vocab_size;
  if (row >= verification_rows_ ||
      verification_logits_.size() < (row + 1) * vocab) {
    return {};
  }
  return std::span<const float>(verification_logits_)
      .subspan(row * vocab, vocab);
}

speculative::VerificationChunkResult FlintGpuSpeculativeTarget::ForwardVerificationChunk(
    std::span<const tokenization::TokenId> candidate_tokens,
    std::uint32_t start_pos, bool capture_hidden, bool capture_logits) {
  const auto& weights = model_.GetWeights();
  const auto buffers = arena_.GetBuffers();
  const auto prefill = arena_.GetPrefillBuffers();
  const std::size_t rows = candidate_tokens.size();
  const std::size_t hidden = weights.config.hidden_size;
  const std::size_t taps = target_layer_ids_.size();
  const std::size_t vocab = weights.config.vocab_size;
  ExecuteFlintVerifyChunk(weights, candidate_tokens, start_pos, buffers,
                          prefill);
  HIP_CHECK(hipStreamSynchronize(buffers.stream));
  speculative::VerificationChunkResult result;
  result.predictions.resize(rows);
  HIP_CHECK(hipMemcpy(result.predictions.data(), prefill.verify_tokens,
                      rows * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
  if (capture_hidden) {
    result.hidden_width = taps * hidden;
    result.hidden_states.resize(rows * result.hidden_width);
    HIP_CHECK(hipMemcpy(result.hidden_states.data(),
                        prefill.target_layer_features,
                        result.hidden_states.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    last_hidden_.assign(result.hidden_states.end() - result.hidden_width,
                        result.hidden_states.end());
  }
  // The verify tail always computes per-row logits. Retain them so the
  // verifier can read the frontier row through CopyVerificationLogits even
  // when it did not request the whole chunk (greedy verification).
  result.vocab_size = vocab;
  verification_logits_.resize(rows * vocab);
  HIP_CHECK(hipMemcpy(verification_logits_.data(), prefill.verify_logits,
                      verification_logits_.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  verification_rows_ = rows;
  if (capture_logits) {
    result.logits = verification_logits_;
  }
  position_ = start_pos + static_cast<std::uint32_t>(rows);
  return result;
}

void FlintGpuSpeculativeTarget::CommitVerificationChunk(
    std::span<const tokenization::TokenId> committed_tokens,
    std::uint32_t start_pos) {
  // The verify tail uses batched-prefill arithmetic, which rounds differently
  // from the decode GEMV path. Replay the committed rows through the decode
  // step so the retained KV matches a plain autoregressive run exactly (GX).
  std::uint32_t pos = start_pos;
  for (const auto token : committed_tokens) {
    (void)ForwardToken(token, pos++, /*compute_logits=*/false);
  }
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)
