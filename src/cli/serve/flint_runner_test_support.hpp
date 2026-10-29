#ifndef GUFO_SERVER_FLINT_RUNNER_TEST_SUPPORT_HPP_
#define GUFO_SERVER_FLINT_RUNNER_TEST_SUPPORT_HPP_

#include <cstdint>
#include <memory>
#include <vector>

#include "src/cli/serve/text_model_runner.hpp"
#include "src/models/qwen/hip/flint_gpu_model.hpp"

namespace gufo::hip {
class FlintGpuModel;
}  // namespace gufo::hip

namespace gufo::tokenization {
class QwenTokenizer;
}  // namespace gufo::tokenization

namespace gufo::server {

#if defined(ENGINE_ENABLE_HIP)
/// Test-only: builds the flint text runner so snapshot/restore hooks can be
/// driven directly. Production code reaches the runner through the backend.
std::shared_ptr<TextModelRunner> MakeFlintTextRunnerForTest(
    std::shared_ptr<const hip::FlintGpuModel> model,
    std::shared_ptr<const tokenization::QwenTokenizer> tokenizer,
    std::uint32_t max_context, bool lookup_speculation = false);

/// Test-only: copy of the frontier logits of a flint state, or empty when the
/// state has no frontier.
std::vector<float> FlintFrontierLogitsForTest(const TextRunnerState& state);

/// Test-only: the KV window class (F-I2 ring geometry) of a flint state.
hip::FlintKvWindowClass FlintWindowClassForTest(const TextRunnerState& state);
#endif

}  // namespace gufo::server

#endif  // GUFO_SERVER_FLINT_RUNNER_TEST_SUPPORT_HPP_
