// Per-layer diagnostics for the flint CPU forward: hidden-state norms, the
// attention vs FFN contribution split, and router statistics. Debug aid for
// the oracle-divergence investigation.
//
// Usage: flint_debug_layer MODEL.gguf TOKEN_ID [TOKEN_ID ...]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/flint_forward.hpp"
#include "src/models/qwen/forward.hpp"

namespace {

double L2Norm(std::span<const float> x) {
  return std::sqrt(std::accumulate(x.begin(), x.end(), 0.0,
                                   [](double a, float v) { return a + v * v; }));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: flint_debug_layer MODEL.gguf TOK [TOK...]\n");
    return 2;
  }
  std::string err;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &err);
  if (reader == nullptr) {
    std::fprintf(stderr, "open: %s\n", err.c_str());
    return 1;
  }
  auto weights = gufo::models::FlintModelWeights::LoadFromGguf(*reader, &err);
  if (!weights.has_value()) {
    std::fprintf(stderr, "weights: %s\n", err.c_str());
    return 1;
  }
  const auto& config = weights->config;
  gufo::models::QwenKvCache kv(config.num_layers, config.num_key_value_heads,
                               4096, config.head_dim);
  gufo::models::FlintScratchArena arena(config);

  for (int t = 2; t < argc; ++t) {
    const std::uint32_t tok = static_cast<std::uint32_t>(atoi(argv[t]));
    const std::uint32_t pos = static_cast<std::uint32_t>(t - 2);
    gufo::models::ForwardEmbedding(tok, weights->token_embd, config.hidden_size,
                           arena.hidden);
    std::vector<float> hidden(arena.hidden.begin(), arena.hidden.end());
    std::printf("== token %u pos %u emb_norm %.4f\n", tok, pos,
                L2Norm(hidden));
    for (std::uint32_t l = 0; l < config.num_layers; ++l) {
      const auto before = hidden;
      gufo::models::FlintForwardLayer(arena.hidden, weights->layers[l], config, kv, l,
                              pos, arena);
      hidden.assign(arena.hidden.begin(), arena.hidden.end());
      // attn contribution = attn_proj (residual add #1), ffn = mlp_out.
      std::printf(
          "  L%-2u norm %8.3f  attn_norm %8.3f  mlp_norm %8.3f%s%s  "
          "max|h| %7.2f\n",
          l, L2Norm(hidden), L2Norm(arena.attn_proj), L2Norm(arena.mlp_out),
          weights->layers[l].attrs.windowed ? " win" : " gbl",
          weights->layers[l].attrs.rope ? " rope" : " nope",
          *std::max_element(hidden.begin(), hidden.end(),
                            [](float a, float b) { return std::abs(a) < std::abs(b); }));
    }
  }
  return 0;
}
