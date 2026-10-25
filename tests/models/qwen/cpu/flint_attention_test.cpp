// Flint CPU reference attention tests (F-D1 sinks, F-D2 window, F-D7
// boundaries, F-D6 order). Small deterministic shapes with exact expected
// values — the reference the HIP paths are verified against.

#include "src/models/qwen/flint_attention.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

using gufo::models::FlintAttentionPolicy;
using gufo::models::QwenKvCache;

void Expect(bool condition, const std::string& msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

void ExpectNear(float actual, float expected, float tol,
                const std::string& msg) {
  if (std::fabs(actual - expected) > tol) {
    std::cerr << "Assertion failed: " << msg << " (expected " << expected
              << ", got " << actual << ")\n";
    std::exit(1);
  }
}

constexpr std::uint32_t kHeads = 1;
constexpr std::uint32_t kKvHeads = 1;
constexpr std::uint32_t kHeadDim = 4;
constexpr float kScale = 0.5F;  // 1 / sqrt(4)

FlintAttentionPolicy BasePolicy() {
  FlintAttentionPolicy p;
  p.num_heads = kHeads;
  p.num_kv_heads = kKvHeads;
  p.head_dim = kHeadDim;
  return p;
}

// Fills the cache so that k[pos] = [1,0,0,0] and v[pos] = [pos+1,0,0,0].
// With q = [2,0,0,0] every score is 2 * 0.5 = 1.0.
void FillUniformCache(QwenKvCache& kv, std::uint32_t layer,
                      std::uint32_t positions) {
  for (std::uint32_t p = 0; p < positions; ++p) {
    auto k = kv.GetKeySlice(layer, 0, p);
    auto v = kv.GetValueSlice(layer, 0, p);
    k[0] = 1.0F;
    for (std::size_t d = 1; d < k.size(); ++d) k[d] = 0.0F;
    v[0] = static_cast<float>(p) + 1.0F;
    for (std::size_t d = 1; d < v.size(); ++d) v[d] = 0.0F;
  }
}

void TestNoSinkIsPlainSoftmax() {
  QwenKvCache kv(1, kKvHeads, 16, kHeadDim);
  FillUniformCache(kv, 0, 4);
  const std::vector<float> q = {2.0F, 0.0F, 0.0F, 0.0F};
  std::vector<float> scores(16, 0.0F);
  std::vector<float> out(kHeadDim, 0.0F);

  auto policy = BasePolicy();
  policy.sinks = nullptr;
  gufo::models::FlintAttentionReference(policy, q.data(), kv, 0, 3,
                                        scores.data(), out.data());
  // All four scores equal -> uniform weights -> mean of v = (1+2+3+4)/4.
  ExpectNear(out[0], 2.5F, 1e-5F, "plain softmax mean of values");
  std::cout << "TestNoSinkIsPlainSoftmax ok\n";
}

void TestSinkEqualScoreSharesMass() {
  QwenKvCache kv(1, kKvHeads, 16, kHeadDim);
  FillUniformCache(kv, 0, 4);
  const std::vector<float> q = {2.0F, 0.0F, 0.0F, 0.0F};
  std::vector<float> scores(16, 0.0F);
  std::vector<float> out(kHeadDim, 0.0F);

  const float sink = 1.0F;  // equal to every key score
  auto policy = BasePolicy();
  policy.sinks = &sink;
  gufo::models::FlintAttentionReference(policy, q.data(), kv, 0, 3,
                                        scores.data(), out.data());
  // Five equal entries (4 keys + sink): each weight 1/5, sink contributes no
  // value -> out = (1+2+3+4)/5 = 2.0.
  ExpectNear(out[0], 2.0F, 1e-5F, "sink shares the denominator (F-D1)");
  std::cout << "TestSinkEqualScoreSharesMass ok\n";
}

void TestSinkExtremes() {
  QwenKvCache kv(1, kKvHeads, 16, kHeadDim);
  FillUniformCache(kv, 0, 4);
  const std::vector<float> q = {2.0F, 0.0F, 0.0F, 0.0F};
  std::vector<float> scores(16, 0.0F);

  const float very_negative = -1000.0F;
  auto policy = BasePolicy();
  policy.sinks = &very_negative;
  std::vector<float> out(kHeadDim, 0.0F);
  gufo::models::FlintAttentionReference(policy, q.data(), kv, 0, 3,
                                        scores.data(), out.data());
  ExpectNear(out[0], 2.5F, 1e-5F, "very negative sink is inert");

  const float very_positive = 1000.0F;
  policy.sinks = &very_positive;
  gufo::models::FlintAttentionReference(policy, q.data(), kv, 0, 3,
                                        scores.data(), out.data());
  ExpectNear(out[0], 0.0F, 1e-5F, "dominant sink zeroes the output");
  std::cout << "TestSinkExtremes ok\n";
}

void TestKeyRangeEdges() {
  // F-D7: window 513 boundaries.
  auto policy = BasePolicy();
  policy.windowed = true;
  policy.window = 513;
  Expect(policy.KeyRangeStart(0) == 0, "pos 0 sees prefix");
  Expect(policy.KeyRangeStart(511) == 0, "pos 511 sees prefix");
  Expect(policy.KeyRangeStart(512) == 0, "pos 512 (window-1) sees prefix");
  Expect(policy.KeyRangeStart(513) == 1, "pos 513 drops position 0");
  Expect(policy.KeyRangeStart(1025) == 513, "pos 1025 (2*513-1)");
  Expect(policy.KeyRangeStart(1026) == 514, "pos 1026 (2*513)");

  policy.windowed = false;
  Expect(policy.KeyRangeStart(9999) == 0, "global layer sees everything");
  std::cout << "TestKeyRangeEdges ok\n";
}

void TestWindow513Multiples() {
  // F-D7 boundary cases from item 1.8: prompts shorter than the window,
  // the exact window multiples 513*k, and the 1024-token prefill chunk
  // edges. KeyRangeStart is pos+1-window once pos+1 exceeds the window.
  auto policy = BasePolicy();
  policy.windowed = true;
  policy.window = 513;
  Expect(policy.KeyRangeStart(1) == 0, "1-token sequence sees its prefix");
  Expect(policy.KeyRangeStart(513) == 1, "pos 513 = 513*1 drops key 0");
  Expect(policy.KeyRangeStart(1024) == 512,
         "pos 1024 (1024-chunk edge) keeps keys 512..1023");
  Expect(policy.KeyRangeStart(1537) == 1025, "pos 1537 (3*513-2)");
  Expect(policy.KeyRangeStart(1538) == 1026, "pos 1538 (3*513-1)");
  Expect(policy.KeyRangeStart(1539) == 1027, "pos 1539 (3*513)");
  std::cout << "TestWindow513Multiples ok\n";
}

void TestSingleTokenSequence() {
  // F-D7 degenerate: a 1-token sequence attends only itself; with a sink,
  // the sink still takes its mass share (F-D1 order is position-free).
  QwenKvCache kv(1, kKvHeads, 16, kHeadDim);
  FillUniformCache(kv, 0, 1);
  const std::vector<float> q = {2.0F, 0.0F, 0.0F, 0.0F};
  std::vector<float> scores(16, 0.0F);
  std::vector<float> out(kHeadDim, 0.0F);

  auto policy = BasePolicy();
  policy.windowed = true;
  policy.window = 513;
  gufo::models::FlintAttentionReference(policy, q.data(), kv, 0, 0,
                                        scores.data(), out.data());
  ExpectNear(out[0], 1.0F, 1e-5F, "1-token sequence attends only itself");

  const float sink = 1.0F;
  policy.sinks = &sink;
  gufo::models::FlintAttentionReference(policy, q.data(), kv, 0, 0,
                                        scores.data(), out.data());
  ExpectNear(out[0], 0.5F, 1e-5F, "sink shares the 1-key denominator");
  std::cout << "TestSingleTokenSequence ok\n";
}

void TestChunkCrossingWindow() {
  // F-D7 with the flint-real window across a 1024-token prefill chunk
  // boundary: the query at 1024 must see exactly keys [512, 1024] — the
  // previous chunk's keys stay reachable through the cache.
  QwenKvCache kv(1, kKvHeads, 2048, kHeadDim);
  FillUniformCache(kv, 0, 1025);
  const std::vector<float> q = {2.0F, 0.0F, 0.0F, 0.0F};
  std::vector<float> scores(513, 0.0F);
  std::vector<float> out(kHeadDim, 0.0F);

  auto policy = BasePolicy();
  policy.windowed = true;
  policy.window = 513;
  gufo::models::FlintAttentionReference(policy, q.data(), kv, 0, 1024,
                                        scores.data(), out.data());
  // Uniform scores over v[512..1024]: mean = ((512+1) + (1024+1)) / 2 = 769.
  // Float accumulation over 513 terms costs ~n*ulp(769) ≈ 0.05; missing the
  // window by one key shifts the mean ~0.5, so 0.1 still discriminates.
  ExpectNear(out[0], 769.0F, 0.1F, "chunk-crossing window range (F-D7)");
  std::cout << "TestChunkCrossingWindow ok\n";
}

void TestWindowIgnoresOldKeys() {
  QwenKvCache kv(1, kKvHeads, 16, kHeadDim);
  FillUniformCache(kv, 0, 6);
  const std::vector<float> q = {2.0F, 0.0F, 0.0F, 0.0F};
  std::vector<float> scores(16, 0.0F);
  std::vector<float> out(kHeadDim, 0.0F);

  auto policy = BasePolicy();
  policy.windowed = true;
  policy.window = 2;
  // pos 3 -> range [2,3]: mean of v[2], v[3] = (3+4)/2 = 3.5.
  gufo::models::FlintAttentionReference(policy, q.data(), kv, 0, 3,
                                        scores.data(), out.data());
  ExpectNear(out[0], 3.5F, 1e-5F, "window bounds the key range (F-D2)");
  std::cout << "TestWindowIgnoresOldKeys ok\n";
}

void TestGqaGrouping() {
  // 2 query heads, 1 kv head: both heads read the same k/v and produce the
  // same output for the same q.
  QwenKvCache kv(1, 1, 16, kHeadDim);
  FillUniformCache(kv, 0, 4);
  const std::vector<float> q = {2.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F,
                                0.0F};
  std::vector<float> scores(16, 0.0F);
  std::vector<float> out(2 * kHeadDim, 0.0F);

  auto policy = BasePolicy();
  policy.num_heads = 2;
  gufo::models::FlintAttentionReference(policy, q.data(), kv, 0, 3,
                                        scores.data(), out.data());
  ExpectNear(out[0], 2.5F, 1e-5F, "head 0 uniform mean");
  ExpectNear(out[kHeadDim], 2.5F, 1e-5F, "head 1 shares the kv head");
  std::cout << "TestGqaGrouping ok\n";
}

void TestPerHeadSinksDiffer() {
  // Two heads, same q, different sinks: the sinked head is damped.
  QwenKvCache kv(1, 1, 16, kHeadDim);
  FillUniformCache(kv, 0, 4);
  const std::vector<float> q = {2.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F,
                                0.0F};
  std::vector<float> scores(16, 0.0F);
  std::vector<float> out(2 * kHeadDim, 0.0F);
  const float sinks[2] = {1.0F, -1000.0F};

  auto policy = BasePolicy();
  policy.num_heads = 2;
  policy.sinks = sinks;
  gufo::models::FlintAttentionReference(policy, q.data(), kv, 0, 3,
                                        scores.data(), out.data());
  ExpectNear(out[0], 2.0F, 1e-5F, "head 0 sink shares mass");
  ExpectNear(out[kHeadDim], 2.5F, 1e-5F, "head 1 sink inert");
  std::cout << "TestPerHeadSinksDiffer ok\n";
}

}  // namespace

int main() {
  TestNoSinkIsPlainSoftmax();
  TestSinkEqualScoreSharesMass();
  TestSinkExtremes();
  TestKeyRangeEdges();
  TestWindow513Multiples();
  TestSingleTokenSequence();
  TestChunkCrossingWindow();
  TestWindowIgnoresOldKeys();
  TestGqaGrouping();
  TestPerHeadSinksDiffer();
  std::cout << "flint_attention_test: all tests passed\n";
  return 0;
}
