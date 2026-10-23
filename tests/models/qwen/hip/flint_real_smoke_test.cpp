// Flint real-artifact GPU smoke test (F-D4 dispatch entry, real shapes).
// Env-gated on GUFO_FLINT_GGUF (skips with 77 when absent): loads the pinned
// flint GGUF once, runs the same greedy token sequence through the CPU
// reference (FlintForwardModel, the D4 oracle) and the GPU decode step
// (ExecuteFlintDecodeStep over the mapped-registration weights of
// FlintGpuModel), and compares logits step by step in the G3 format
// (cosine >= 0.999, max |delta log p| <= 5e-3, argmax identity). The first
// pass warms the UMM pages; the second pass is timed for decode t/s.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/flint_forward.hpp"
#include "src/models/qwen/hip/detail/flint_decode_step.hpp"
#include "src/models/qwen/hip/flint_gpu_model.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

struct StepParity {
  double cosine{0.0};
  double max_logp_diff{0.0};
  std::uint32_t argmax_cpu{0};
  std::uint32_t argmax_gpu{0};
};

/// G3 format: cosine on raw logits, max |delta log p| over the softmax,
/// and the greedy argmax on both sides.
StepParity CompareLogits(const std::vector<float>& cpu,
                         const std::vector<float>& gpu) {
  StepParity parity;
  double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    dot += static_cast<double>(cpu[i]) * gpu[i];
    norm_a += static_cast<double>(cpu[i]) * cpu[i];
    norm_b += static_cast<double>(gpu[i]) * gpu[i];
    if (cpu[i] > cpu[parity.argmax_cpu]) parity.argmax_cpu = static_cast<std::uint32_t>(i);
    if (gpu[i] > gpu[parity.argmax_gpu]) parity.argmax_gpu = static_cast<std::uint32_t>(i);
  }
  parity.cosine = dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-30);

  double max_cpu = -1e30, max_gpu = -1e30;
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    max_cpu = std::max(max_cpu, static_cast<double>(cpu[i]));
    max_gpu = std::max(max_gpu, static_cast<double>(gpu[i]));
  }
  double sum_cpu = 0.0, sum_gpu = 0.0;
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    sum_cpu += std::exp(static_cast<double>(cpu[i]) - max_cpu);
    sum_gpu += std::exp(static_cast<double>(gpu[i]) - max_gpu);
  }
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    const double logp_cpu =
        static_cast<double>(cpu[i]) - max_cpu - std::log(sum_cpu);
    const double logp_gpu =
        static_cast<double>(gpu[i]) - max_gpu - std::log(sum_gpu);
    parity.max_logp_diff =
        std::max(parity.max_logp_diff, std::abs(logp_cpu - logp_gpu));
  }
  return parity;
}

}  // namespace

int main(int argc, char** argv) {
  const char* env = std::getenv("GUFO_FLINT_GGUF");
  if (env == nullptr || *env == '\0') {
    std::cout << "SKIPPED: GUFO_FLINT_GGUF not set\n";
    return 77;
  }

  std::uint32_t max_context = 4096;
  std::uint32_t steps = 8;
  if (argc > 1) max_context = static_cast<std::uint32_t>(std::atoi(argv[1]));
  if (argc > 2) steps = static_cast<std::uint32_t>(std::atoi(argv[2]));
  Expect(steps > 0, "steps must be positive");

  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(env, &error);
  Expect(reader != nullptr, "open GGUF: " + error);

  // CPU reference weights: host pointers straight from the reader mapping.
  // The reader must outlive both weight sets.
  auto cpu_weights = gufo::models::FlintModelWeights::LoadFromGguf(*reader, &error);
  Expect(cpu_weights.has_value(), "load CPU weights: " + error);
  const auto& config = cpu_weights->config;
  Expect(config.num_layers > 0, "model must have layers");

  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  Expect(tokenizer != nullptr, "create tokenizer: " + error);
  const std::string prompt = "The quick brown fox jumps over the lazy dog.";
  std::vector<gufo::tokenization::TokenId> prompt_tokens =
      tokenizer->Encode(prompt);
  Expect(!prompt_tokens.empty(), "prompt must tokenize to at least one token");

  // GPU model: registered mapped copies, tensors re-pointed at device space.
  auto gpu_model = gufo::hip::FlintGpuModel::CreateFromGguf(
      std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)), &error);
  Expect(gpu_model != nullptr, "create GPU model: " + error);

  std::cout << "flint real smoke: " << env << "\n"
            << "  layers=" << config.num_layers
            << " hidden=" << config.hidden_size
            << " heads=" << config.num_attention_heads << "/"
            << config.num_key_value_heads << " head_dim=" << config.head_dim
            << " vocab=" << config.vocab_size
            << " experts=" << config.expert_count << " top"
            << config.expert_used_count
            << " window=" << config.sliding_window << "\n"
            << "  weight regions=" << gpu_model->GetWeightRegionCount()
            << " resident=" << gpu_model->GetResidentBytes() / (1ULL << 20)
            << " MiB\n"
            << "  prompt(" << prompt_tokens.size() << " tokens): " << prompt
            << "\n  max_context=" << max_context << " steps=" << steps << "\n";

  gufo::models::QwenKvCache kv(config.num_layers,
                               config.num_key_value_heads, max_context,
                               config.head_dim);
  gufo::models::FlintScratchArena arena(config);
  std::vector<float> cpu_logits(config.vocab_size);

  gufo::hip::FlintGpuArena gpu_arena(config, max_context);
  const auto buffers = gpu_arena.GetBuffers();
  std::vector<float> gpu_logits(config.vocab_size);
  std::cout << "  arena scratch=" << gpu_arena.GetScratchBytes() / (1ULL << 20)
            << " MiB\n";

  for (int pass = 0; pass < 2; ++pass) {
    // The KV cache is only read at [0, pos] and every position is rewritten
    // before it is read, so the second pass re-runs cleanly without a reset.
    std::uint32_t token = prompt_tokens[0];
    double total_ms = 0.0;
    std::cout << (pass == 0 ? "  pass 1 (warm):" : "  pass 2 (timed):");
    for (std::uint32_t step = 0; step < steps; ++step) {
      gufo::models::FlintForwardModel(token, step, *cpu_weights, kv, arena,
                                      cpu_logits);

      const auto t0 = std::chrono::steady_clock::now();
      gufo::hip::ExecuteFlintDecodeStep(gpu_model->GetWeights(), token, step,
                                        /*compute_logits=*/true, buffers);
      HIP_CHECK(hipStreamSynchronize(buffers.stream));
      const auto t1 = std::chrono::steady_clock::now();
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

      HIP_CHECK(hipMemcpy(gpu_logits.data(), buffers.logits,
                          gpu_logits.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      const auto parity = CompareLogits(cpu_logits, gpu_logits);
      Expect(parity.cosine >= 0.999,
             "cosine at step " + std::to_string(step) + " = " +
                 std::to_string(parity.cosine));
      Expect(parity.max_logp_diff <= 5e-3,
             "max |delta log p| at step " + std::to_string(step) + " = " +
                 std::to_string(parity.max_logp_diff));
      Expect(parity.argmax_cpu == parity.argmax_gpu,
             "argmax mismatch at step " + std::to_string(step) + ": cpu " +
                 std::to_string(parity.argmax_cpu) + " gpu " +
                 std::to_string(parity.argmax_gpu));
      std::cout << " s" << step << " tok=" << parity.argmax_gpu
                << " cos=" << parity.cosine
                << " dlogp=" << parity.max_logp_diff
                << " (" << total_ms / (step + 1) << " ms/step avg)";
      token = parity.argmax_cpu;
    }
    std::cout << "\n";
    if (pass == 1) {
      const double tps = steps * 1000.0 / total_ms;
      std::cout << "  decode: " << total_ms / steps << " ms/step = " << tps
                << " t/s\n";
    }
  }

  std::cout << "flint_real_smoke_test ok (" << steps
            << " steps x 2 passes, G3 format, real artifact)\n";
  return 0;
}
