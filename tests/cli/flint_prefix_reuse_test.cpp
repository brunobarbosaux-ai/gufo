// F-I3 (5.4): prefix-reuse parity. A prefix restored from a snapshot and
// extended by prefill must produce the same frontier logits as a cold prefill
// of the whole sequence, for both KV layer classes and for a prefix longer
// than the 513-token sliding window (including one that wraps the ring).
// Env-gated on GUFO_FLINT_GGUF (skips with 77 when absent). The runner is
// built through the test-only factory in flint_runner_test_support.hpp.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/flint_runner_test_support.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/flint_gpu_model.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

using gufo::server::FlintFrontierLogitsForTest;
using gufo::server::MakeFlintTextRunnerForTest;
using gufo::server::TextModelRunner;
using gufo::tokenization::QwenTokenizer;
using gufo::tokenization::TokenId;

void Expect(bool condition, const std::string& msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

std::uint32_t Argmax(const std::vector<float>& logits) {
  std::uint32_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) best = static_cast<std::uint32_t>(i);
  }
  return best;
}

bool BitExact(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(),
                                   a.size() * sizeof(float)) == 0);
}

// Largest absolute difference, for a readable failure message.
float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  float worst = 0.0F;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i)
    worst = std::max(worst, std::abs(a[i] - b[i]));
  return worst;
}

// A prompt of exactly `tokens` tokens, built by repeating a sentence so the
// content is varied enough to avoid degenerate repetition.
std::vector<TokenId> MakePrompt(const QwenTokenizer& tokenizer,
                                std::size_t tokens) {
  static constexpr std::string_view kSentence =
      "The flint engine keeps a sliding window of keys for the local layers "
      "and a full cache for the global layers, so a reused prefix must carry "
      "exactly the keys the next query needs. ";
  std::vector<TokenId> prompt;
  while (prompt.size() < tokens) {
    const auto piece = tokenizer.Encode(kSentence);
    prompt.insert(prompt.end(), piece.begin(), piece.end());
  }
  prompt.resize(tokens);
  return prompt;
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
  std::shared_ptr<const QwenTokenizer> tokenizer = std::move(tokenizer_owned);

  auto model = gufo::hip::FlintGpuModel::CreateFromGguf(
      std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)), &error);
  Expect(model != nullptr, "create GPU model: " + error);

  // The parity claim is per layer class, so the artifact must carry both.
  const auto& config = model->GetConfig();
  std::size_t windowed = 0;
  for (std::uint32_t l = 0; l < config.num_layers; ++l)
    windowed += config.IsWindowedLayer(l) ? 1 : 0;
  const std::size_t global = config.num_layers - windowed;
  Expect(windowed > 0 && global > 0,
         "the parity test needs both windowed and global layers");
  std::cout << "layers: windowed=" << windowed << " global=" << global
            << " sliding_window=" << config.sliding_window << "\n";

  // Headroom above the longest case so the decode-parity steps stay in range.
  constexpr std::uint32_t kMaxContext = 4096;
  const std::shared_ptr<TextModelRunner> runner =
      MakeFlintTextRunnerForTest(model, tokenizer, kMaxContext);

  const std::vector<TokenId> prompt = MakePrompt(*tokenizer, kMaxContext);

  struct Case {
    const char* label;
    std::size_t total;
    std::size_t prefix;
  };
  // sub-window: prefix below the 513-token window.
  // over-window: prefix past the window but inside the ring.
  // ring-wrap: prefix past the ring modulus (window + prefill chunk).
  const Case cases[] = {
      {"sub-window", 512, 256},
      {"over-window", 1024, 600},
      {"ring-wrap", 2048, 1600},
  };

  for (const auto& c : cases) {
    const std::span<const TokenId> full(prompt.data(), c.total);

    // Cold: one prefill of the whole sequence.
    auto cold = runner->CreateState();
    runner->Prefill(*cold, full, 0, c.total);
    const std::vector<float> cold_logits = FlintFrontierLogitsForTest(*cold);
    Expect(!cold_logits.empty(), "cold prefill must leave frontier logits");

    // Reuse: prefill the prefix, snapshot it, restore into a fresh state,
    // then prefill the suffix. This is the serve path's cached-prefix reuse.
    auto source = runner->CreateState();
    runner->Prefill(*source, full, 0, c.prefix);
    auto snapshot = runner->Snapshot(*source);
    Expect(snapshot != nullptr, "prefix snapshot must be produced");
    auto reused = runner->CreateState();
    runner->RestoreOrFork(*reused, *snapshot);
    Expect(runner->CheckpointPosition(*reused) == c.prefix,
           "restore must land on the prefix frontier");
    runner->Prefill(*reused, full, c.prefix, c.total - c.prefix);
    const std::vector<float> reused_logits =
        FlintFrontierLogitsForTest(*reused);

    Expect(BitExact(cold_logits, reused_logits),
           std::string(c.label) +
               ": reused-prefix frontier logits differ from cold prefill "
               "(max|d|=" +
               std::to_string(MaxAbsDiff(cold_logits, reused_logits)) + ")");

    // Decode parity: the same token must advance both states identically,
    // which can only hold if the restored KV planes are complete for both
    // layer classes. Advance syncs per step, so no unsynced-launch buildup.
    std::uint32_t token = Argmax(cold_logits);
    for (int step = 0; step < 2; ++step) {
      runner->Advance(*cold, token);
      runner->Advance(*reused, token);
      const auto a = FlintFrontierLogitsForTest(*cold);
      const auto b = FlintFrontierLogitsForTest(*reused);
      Expect(BitExact(a, b),
             std::string(c.label) + ": decode step " + std::to_string(step) +
                 " logits differ (max|d|=" + std::to_string(MaxAbsDiff(a, b)) +
                 ")");
      token = Argmax(a);
    }

    std::cout << "case " << c.label << ": total=" << c.total
              << " prefix=" << c.prefix << " frontier=" << cold_logits.size()
              << " PASS\n";
  }

  std::cout << "flint prefix-reuse parity: "
            << (sizeof(cases) / sizeof(cases[0])) << " cases PASS\n";
  return 0;
}
