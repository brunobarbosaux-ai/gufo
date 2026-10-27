// Drives FlintTextRunner's snapshot/restore hooks directly (5.3c). Env-gated
// on GUFO_FLINT_GGUF (skips with 77 when absent). The runner is built through
// the test-only factory in flint_runner_test_support.hpp.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cli/serve/flint_runner_test_support.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/flint_gpu_model.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

using gufo::server::FlintFrontierLogitsForTest;
using gufo::server::FlintWindowClassForTest;
using gufo::server::MakeFlintTextRunnerForTest;
using gufo::server::TextModelRunner;
using gufo::server::TextRunnerSnapshot;
using gufo::server::TextRunnerState;

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

  // F-I5: the four flint EOG terminators must be in the stop set. The
  // non-conformant IQ4_XS omits tokenizer.ggml.eog_token_ids, so this also
  // exercises the tokenizer's EOG-fallback derivation.
  for (const std::uint32_t eog : {150001U, 150003U, 150005U, 150018U}) {
    Expect(tokenizer->IsStopToken(eog),
           "flint EOG id " + std::to_string(eog) + " must be a stop token");
  }

  auto model = gufo::hip::FlintGpuModel::CreateFromGguf(
      std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)), &error);
  Expect(model != nullptr, "create GPU model: " + error);

  // Prompt of exactly 16 tokens.
  std::vector<gufo::tokenization::TokenId> prompt = tokenizer->Encode(
      "The quick brown fox jumps over the lazy dog. The quick brown fox "
      "jumps over the lazy dog. The quick brown fox jumps over the lazy dog.");
  Expect(prompt.size() >= 16, "prompt must tokenize to at least 16 tokens");
  prompt.resize(16);

  constexpr std::uint32_t kMaxContext = 2048;
  const std::shared_ptr<TextModelRunner> runner =
      MakeFlintTextRunnerForTest(model, tokenizer, kMaxContext);
  auto state = runner->CreateState();

  runner->Prefill(*state, prompt, 0, prompt.size());
  const std::size_t pos0 = runner->CheckpointPosition(*state);
  Expect(pos0 == prompt.size(), "prefill must advance to the prompt length");

  const std::vector<float> logits0 = FlintFrontierLogitsForTest(*state);
  Expect(!logits0.empty(), "prefill must leave frontier logits");

  auto snap = runner->Snapshot(*state);
  Expect(snap != nullptr, "snapshot must be produced");
  Expect(runner->SnapshotPayloadBytes(*state) == snap->PayloadBytes(),
         "SnapshotPayloadBytes must match the snapshot payload");

  // Two decode steps past the snapshot, recording the logits they produce.
  const std::uint32_t t1 = Argmax(logits0);
  runner->Advance(*state, t1);
  const std::vector<float> logits1 = FlintFrontierLogitsForTest(*state);
  const std::uint32_t t2 = Argmax(logits1);
  runner->Advance(*state, t2);
  const std::vector<float> logits2 = FlintFrontierLogitsForTest(*state);
  Expect(runner->CheckpointPosition(*state) == pos0 + 2,
         "two advances must move the frontier by two");

  // Restore: position and frontier logits must match the snapshot exactly.
  runner->RestoreOrFork(*state, *snap);
  Expect(runner->CheckpointPosition(*state) == pos0,
         "restore must rewind the checkpoint position");
  const std::vector<float> restored0 = FlintFrontierLogitsForTest(*state);
  Expect(BitExact(restored0, logits0),
         "restored frontier logits must be bit-exact to the snapshot's");

  // Replaying the same two tokens must reproduce the same logits bit-exactly,
  // which can only hold if the KV planes were restored, not just the position.
  runner->Advance(*state, t1);
  Expect(BitExact(FlintFrontierLogitsForTest(*state), logits1),
         "replayed step 1 logits must be bit-exact after restore");
  runner->Advance(*state, t2);
  Expect(BitExact(FlintFrontierLogitsForTest(*state), logits2),
         "replayed step 2 logits must be bit-exact after restore");

  // F-I5 (5.5): the persistent (disk) round trip. Serialize the snapshot to
  // bytes, restore it into a fresh state as a restart would, and verify the
  // frontier logits, the window class and the position survive.
  const auto window_class = FlintWindowClassForTest(*state);
  const std::size_t payload_bytes =
      runner->PersistentSnapshotPayloadBytes(*snap);
  Expect(payload_bytes > 0, "persistent payload must be non-empty");
  std::vector<std::uint8_t> payload(payload_bytes);
  Expect(runner->SerializePersistentSnapshot(*snap, payload) == payload_bytes,
         "serialize must fill the exact payload size");

  auto restored = runner->CreateState();
  runner->RestorePersistentSnapshot(*restored, payload);
  Expect(runner->CheckpointPosition(*restored) == pos0,
         "persistent restore must land on the snapshot position");
  Expect(BitExact(FlintFrontierLogitsForTest(*restored), logits0),
         "persistent restore must reproduce the frontier logits");
  Expect(FlintWindowClassForTest(*restored) == window_class,
         "persistent restore must preserve the KV window class");

  // Replaying the same two tokens must reproduce the same logits, which can
  // only hold if the serialized KV planes were restored.
  runner->Advance(*restored, t1);
  Expect(BitExact(FlintFrontierLogitsForTest(*restored), logits1),
         "persistent replay step 1 logits must be bit-exact");
  runner->Advance(*restored, t2);
  Expect(BitExact(FlintFrontierLogitsForTest(*restored), logits2),
         "persistent replay step 2 logits must be bit-exact");

  // Negative: a state with a different max_context has another window class.
  const std::shared_ptr<TextModelRunner> other_runner =
      MakeFlintTextRunnerForTest(model, tokenizer, 1024);
  auto other_state = other_runner->CreateState();
  bool threw = false;
  try {
    other_runner->RestoreOrFork(*other_state, *snap);
  } catch (const std::invalid_argument& e) {
    threw = true;
    std::cout << "window-class mismatch rejected: " << e.what() << "\n";
  }
  Expect(threw, "restore into a different window class must throw "
                "std::invalid_argument");

  std::cout << "flint snapshot hooks: prompt=16 pos0=" << pos0
            << " payload_bytes=" << snap->PayloadBytes()
            << " persistent_bytes=" << payload_bytes
            << " frontier_logits=" << logits0.size() << " PASS\n";
  return 0;
}
