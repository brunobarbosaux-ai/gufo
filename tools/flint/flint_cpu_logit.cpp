// Flint CPU reference log-prob dump: runs the D4 CPU forward over a probe
// text and writes plain f32 log-prob rows (one row per next-token position,
// vocab-wide) for comparison against the llama.cpp oracle dumps.
//
// Usage: flint_cpu_logit MODEL.gguf PROBE.txt OUT.raw [--max N]
// Emits rows for positions 1..min(N, tokens-1); prints the token count and
// the greedy argmax chain for eyeball checks against the oracle.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/flint_forward.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

void LogSoftmax(std::vector<float>& logits) {
  const auto best = static_cast<float>(
      *std::max_element(logits.begin(), logits.end()));
  double sum = 0.0;
  for (float& v : logits) {
    v = std::exp(v - best);
    sum += v;
  }
  const float inv = static_cast<float>(1.0 / sum);
  for (float& v : logits) {
    v = std::log(v * inv);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: flint_cpu_logit MODEL.gguf PROBE.txt OUT.raw "
                 "[--max N]\n");
    return 2;
  }
  std::string err;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &err);
  if (reader == nullptr) {
    std::fprintf(stderr, "open failed: %s\n", err.c_str());
    return 1;
  }
  std::ifstream in(argv[2]);
  const std::string text{(std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>()};

  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
  if (tokenizer == nullptr) {
    std::fprintf(stderr, "tokenizer failed: %s\n", err.c_str());
    return 1;
  }
  const auto tokens = tokenizer->Encode(text);
  std::fprintf(stderr, "tokens: %zu\n", tokens.size());

  auto weights = gufo::models::FlintModelWeights::LoadFromGguf(*reader, &err);
  if (!weights.has_value()) {
    std::fprintf(stderr, "weights failed: %s\n", err.c_str());
    return 1;
  }

  std::size_t max_rows = tokens.size();
  for (int i = 4; i < argc - 1; ++i) {
    if (std::strcmp(argv[i], "--max") == 0) {
      max_rows = std::min(max_rows, static_cast<std::size_t>(atol(argv[i + 1])));
    }
  }

  const std::uint32_t vocab = weights->config.vocab_size;
  gufo::models::QwenKvCache kv(weights->config.num_layers,
                               weights->config.num_key_value_heads,
                               weights->config.context_length,
                               weights->config.head_dim);
  gufo::models::FlintScratchArena arena(weights->config);
  std::vector<float> logits(vocab, 0.0F);
  std::FILE* out = std::fopen(argv[3], "wb");
  if (out == nullptr) {
    std::fprintf(stderr, "cannot write %s\n", argv[3]);
    return 1;
  }

  std::fprintf(stderr, "greedy argmax chain: ");
  for (std::size_t i = 0; i < max_rows; ++i) {
    gufo::models::FlintForwardModel(tokens[i], static_cast<std::uint32_t>(i),
                                    *weights, kv, arena, logits);
    std::size_t argmax = 0;
    float best = logits[0];
    for (std::size_t v = 1; v < vocab; ++v) {
      if (logits[v] > best) {
        best = logits[v];
        argmax = v;
      }
    }
    if (i < 8) std::fprintf(stderr, "%zu ", argmax);
    LogSoftmax(logits);
    std::fwrite(logits.data(), sizeof(float), vocab, out);
    if ((i + 1) % 8 == 0) {
      std::fprintf(stderr, "\rrow %zu/%zu", i + 1, max_rows);
    }
  }
  std::fprintf(stderr, "\n");
  std::fclose(out);
  return 0;
}
