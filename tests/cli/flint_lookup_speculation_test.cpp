// Drives FlintTextRunner's prompt-lookup speculative DecodeStep (7.6). Env-gated
// on GUFO_FLINT_GGUF (skips with 77 when absent). Asserts determinism, draft
// acceptance, position bookkeeping, and first-token agreement with the
// non-speculative run. Token-level divergence is approximate (the verify path's
// Q8_1 activations differ from decode's f32 GEMV) and is reported only.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cli/serve/flint_runner_test_support.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen/hip/flint_gpu_model.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

using gufo::server::FlintFrontierLogitsForTest;
using gufo::server::MakeFlintTextRunnerForTest;
using gufo::server::TextDecodeStep;
using gufo::server::TextModelRunner;
using gufo::server::TextRunnerState;

void Expect(bool condition, const std::string& msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

struct DecodeRun {
  std::vector<std::uint32_t> tokens;
  std::size_t lookup_tokens{0};
  std::size_t lookup_accepted{0};
  std::size_t final_position{0};
  bool stopped{false};
};

// Greedy decode from a fresh state until `target` tokens are emitted or the
// step reports a stop. Speculative steps may emit several tokens at once.
DecodeRun Decode(const TextModelRunner& runner,
                 const std::vector<gufo::tokenization::TokenId>& prompt,
                 std::size_t target) {
  auto state = runner.CreateState();
  runner.Prefill(*state, prompt, 0, prompt.size());
  Expect(runner.CheckpointPosition(*state) == prompt.size(),
         "prefill must advance to the prompt length");

  gufo::sampling::SamplerState sampler(gufo::sampling::SamplingConfig{});
  DecodeRun run;
  while (run.tokens.size() < target && !run.stopped) {
    const std::size_t before = runner.CheckpointPosition(*state);
    const TextDecodeStep step =
        runner.DecodeStep(*state, target - run.tokens.size(), sampler);
    for (const auto& selection : step.selections) {
      Expect(!selection.stop, "emitted selections must not be stop markers");
      run.tokens.push_back(selection.token);
    }
    // Each emitted token advances the checkpoint by exactly one position.
    Expect(runner.CheckpointPosition(*state) ==
               before + step.selections.size(),
           "checkpoint must advance by the number of emitted tokens");
    run.lookup_tokens += step.lookup_tokens;
    run.lookup_accepted += step.lookup_accepted_tokens;
    if (step.stop) run.stopped = true;
    if (step.selections.empty()) break;
  }
  run.final_position = runner.CheckpointPosition(*state);
  Expect(run.final_position == prompt.size() + run.tokens.size(),
         "checkpoint must equal prompt length plus emitted tokens");
  return run;
}

}  // namespace

int main() {
  const char* env = std::getenv("GUFO_FLINT_GGUF");
  if (env == nullptr || *env == '\0') {
    std::cout << "SKIPPED: GUFO_FLINT_GGUF not set\n";
    return 77;
  }

  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(env, &error);
  Expect(reader != nullptr, "open GGUF: " + error);

  auto tokenizer_owned =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  Expect(tokenizer_owned != nullptr, "create tokenizer: " + error);
  std::shared_ptr<const gufo::tokenization::QwenTokenizer> tokenizer =
      std::move(tokenizer_owned);

  auto model = gufo::hip::FlintGpuModel::CreateFromGguf(
      std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)), &error);
  Expect(model != nullptr, "create GPU model: " + error);

  // Echo-heavy prompt: one paragraph repeated four times, so the anchor token
  // keeps matching earlier context and prompt lookup fires.
  const std::string paragraph =
      "The quick brown fox jumps over the lazy dog. Pack my box with five "
      "dozen liquor jugs. ";
  std::string text;
  for (int i = 0; i < 4; ++i) text += paragraph;
  const std::vector<gufo::tokenization::TokenId> prompt = tokenizer->Encode(text);
  Expect(prompt.size() >= 32, "prompt must tokenize to at least 32 tokens");

  constexpr std::uint32_t kMaxContext = 2048;
  constexpr std::size_t kTarget = 64;
  const std::shared_ptr<TextModelRunner> spec_runner =
      MakeFlintTextRunnerForTest(model, tokenizer, kMaxContext,
                                 /*lookup_speculation=*/true);
  const std::shared_ptr<TextModelRunner> plain_runner =
      MakeFlintTextRunnerForTest(model, tokenizer, kMaxContext,
                                 /*lookup_speculation=*/false);

  // Determinism: two speculative runs from fresh states must agree exactly.
  const DecodeRun spec_a = Decode(*spec_runner, prompt, kTarget);
  const DecodeRun spec_b = Decode(*spec_runner, prompt, kTarget);
  Expect(!spec_a.tokens.empty(), "speculative decode must emit tokens");
  Expect(spec_a.tokens == spec_b.tokens,
         "speculative decode must be deterministic across runs");

  // Acceptance: speculation must actually fire on the repetitive prompt.
  Expect(spec_a.lookup_tokens > 0, "prompt lookup must propose drafts");
  Expect(spec_a.lookup_accepted > 0,
         "prompt lookup must accept at least one draft on the echo prompt");
  const double accept_rate =
      100.0 * static_cast<double>(spec_a.lookup_accepted) /
      static_cast<double>(spec_a.lookup_tokens);

  // Plumbing sanity: both runners sample the first token from the same
  // prompt-prefill frontier logits, before any speculation, so they must agree
  // exactly. A mismatch here means a wiring bug, not numerical noise.
  const DecodeRun plain = Decode(*plain_runner, prompt, kTarget);
  Expect(!plain.tokens.empty(), "non-speculative decode must emit tokens");
  Expect(spec_a.tokens[0] == plain.tokens[0],
         "first token must match between speculative and plain runs");

  // Informational only. Speculative verify uses Q8_1 activations while decode
  // uses f32 GEMV, so the two paths differ numerically (measured at ~3.5% per
  // position by the `verify` smoke mode). One early flip changes the context
  // for every later token, so the token-by-token rate is not a correctness
  // bound and is deliberately not asserted.
  const std::size_t n = spec_a.tokens.size();
  const std::size_t common = std::min(n, plain.tokens.size());
  std::size_t divergence = 0;
  std::size_t first_divergence = common;
  for (std::size_t i = 0; i < common; ++i) {
    if (spec_a.tokens[i] != plain.tokens[i]) {
      ++divergence;
      first_divergence = std::min(first_divergence, i);
    }
  }
  divergence += std::max(n, plain.tokens.size()) - common;
  std::cout << "  first divergence index: " << first_divergence << " of " << n
            << "\n";
  std::cout << "  divergence vs non-speculative: " << divergence << "/" << n
            << " (" << (n > 0 ? 100.0 * divergence / n : 0.0) << "%)\n";

  std::cout << "flint lookup speculation: tokens=" << n
            << " lookup=" << spec_a.lookup_tokens
            << " accepted=" << spec_a.lookup_accepted
            << " accept_rate=" << accept_rate << "% first_divergence="
            << first_divergence << " divergence=" << divergence << "/" << n
            << " PASS\n";
  return 0;
}
