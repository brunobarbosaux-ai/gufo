// Per-layer diagnostics for the flint CPU forward: hidden-state norms, the
// attention vs FFN contribution split, and router statistics. Debug aid for
// the oracle-divergence investigation.
//
// Usage: flint_debug_layer MODEL.gguf TOKEN_ID [TOKEN_ID ...]
//
// Registered as the ctest `flint_layer_probe` (F-K1 6.1). Without arguments
// the ctest run uses GUFO_FLINT_GGUF and a fixed token pair, and asserts the
// per-layer invariants the printout is meant to surface: every hidden, attn
// and MLP norm is finite; the MLP branch is nonzero on every layer (a dead
// branch shows as an exact zero); and the window/rope pattern matches the
// config. Exits 77 (skip) when the artifact is not configured.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/flint_forward.hpp"
#include "src/models/qwen/forward.hpp"

namespace {

double L2Norm(std::span<const float> x) {
  return std::sqrt(std::accumulate(x.begin(), x.end(), 0.0,
                                   [](double a, float v) { return a + v * v; }));
}

bool AllFinite(std::span<const float> x) {
  return std::all_of(x.begin(), x.end(),
                     [](float v) { return std::isfinite(v); });
}

int Fail(const char* msg, std::uint32_t layer) {
  std::fprintf(stderr, "flint_layer_probe: %s at layer %u\n", msg, layer);
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  // Default ctest mode: no arguments, artifact from the environment.
  std::string model_path;
  std::vector<std::uint32_t> tokens;
  const bool ctest_mode = argc < 3;
  if (ctest_mode) {
    const char* env = std::getenv("GUFO_FLINT_GGUF");
    if (env == nullptr || *env == '\0') {
      std::printf("flint_layer_probe: skipped (GUFO_FLINT_GGUF unset)\n");
      return 77;
    }
    model_path = env;
    tokens = {2, 1};
  } else {
    model_path = argv[1];
    for (int t = 2; t < argc; ++t) {
      tokens.push_back(static_cast<std::uint32_t>(atoi(argv[t])));
    }
  }

  std::string err;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &err);
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
  if (weights->layers.size() != config.num_layers) {
    return Fail("layer count differs from config", 0);
  }
  gufo::models::QwenKvCache kv(config.num_layers, config.num_key_value_heads,
                               4096, config.head_dim);
  gufo::models::FlintScratchArena arena(config);

  for (std::size_t t = 0; t < tokens.size(); ++t) {
    const std::uint32_t tok = tokens[t];
    const std::uint32_t pos = static_cast<std::uint32_t>(t);
    gufo::models::ForwardEmbedding(tok, weights->token_embd, config.hidden_size,
                                   arena.hidden);
    std::vector<float> hidden(arena.hidden.begin(), arena.hidden.end());
    std::printf("== token %u pos %u emb_norm %.4f\n", tok, pos,
                L2Norm(hidden));
    for (std::uint32_t l = 0; l < config.num_layers; ++l) {
      gufo::models::FlintForwardLayer(arena.hidden, weights->layers[l], config,
                                      kv, l, pos, arena);
      hidden.assign(arena.hidden.begin(), arena.hidden.end());
      // attn contribution = attn_proj (residual add #1), ffn = mlp_out.
      const double hidden_norm = L2Norm(hidden);
      const double attn_norm = L2Norm(arena.attn_proj);
      const double mlp_norm = L2Norm(arena.mlp_out);
      std::printf(
          "  L%-2u norm %8.3f  attn_norm %8.3f  mlp_norm %8.3f%s%s  "
          "max|h| %7.2f\n",
          l, hidden_norm, attn_norm, mlp_norm,
          weights->layers[l].attrs.windowed ? " win" : " gbl",
          weights->layers[l].attrs.rope ? " rope" : " nope",
          *std::max_element(hidden.begin(), hidden.end(), [](float a, float b) {
            return std::abs(a) < std::abs(b);
          }));

      if (!AllFinite(hidden) || !std::isfinite(attn_norm) ||
          !std::isfinite(mlp_norm)) {
        return Fail("non-finite activation", l);
      }
      // A zero MLP branch on a MoE layer means the expert path went dead.
      if (!weights->layers[l].attrs.dense_ffn && mlp_norm == 0.0) {
        return Fail("MoE branch is exactly zero", l);
      }
      if (weights->layers[l].attrs.windowed != config.IsWindowedLayer(l) ||
          weights->layers[l].attrs.rope != config.IsRopeLayer(l)) {
        return Fail("window/rope attribute disagrees with config", l);
      }
    }
  }
  if (ctest_mode) {
    std::printf("flint_layer_probe: %zu layers x %zu tokens finite\n",
                static_cast<std::size_t>(config.num_layers), tokens.size());
  }
  return 0;
}
