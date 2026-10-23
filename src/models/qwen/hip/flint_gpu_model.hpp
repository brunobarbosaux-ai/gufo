#ifndef GUFO_MODELS_QWEN_HIP_FLINT_GPU_MODEL_HPP_
#define GUFO_MODELS_QWEN_HIP_FLINT_GPU_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/detail/flint_decode_step.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/state.hpp"

#if defined(ENGINE_ENABLE_HIP)

namespace gufo::hip {

/// Immutable GPU-visible flint model: the exact flint tensor inventory
/// (F-C1) re-pointed at registered, GPU-visible copies of the GGUF shards —
/// the same mapped-registration path QwenGpuModel uses, so no per-tensor
/// device copies. The reader is retained because the region offsets are
/// relative to its original mappings.
class FlintGpuModel {
 public:
  FlintGpuModel(std::shared_ptr<const core::GgufReader> reader,
                models::FlintModelWeights weights,
                std::vector<QwenGpuWeightRegion> weight_regions);
  ~FlintGpuModel();

  FlintGpuModel(const FlintGpuModel&) = delete;
  FlintGpuModel& operator=(const FlintGpuModel&) = delete;
  FlintGpuModel(FlintGpuModel&&) = delete;
  FlintGpuModel& operator=(FlintGpuModel&&) = delete;

  /// Loads the flint inventory from `reader` and re-points every tensor at
  /// a registered GPU-visible copy. Fails closed when any tensor does not
  /// fall inside a mapped shard (F-C4).
  [[nodiscard]] static std::shared_ptr<const FlintGpuModel> CreateFromGguf(
      std::shared_ptr<const core::GgufReader> reader,
      std::string* error_msg = nullptr);

  [[nodiscard]] const models::FlintModelWeights& GetWeights() const noexcept {
    return weights_;
  }
  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }
  [[nodiscard]] std::size_t GetWeightRegionCount() const noexcept {
    return weight_regions_.size();
  }
  [[nodiscard]] std::size_t GetResidentBytes() const noexcept;

 private:
  std::shared_ptr<const core::GgufReader> reader_;
  models::FlintModelWeights weights_;
  std::vector<QwenGpuWeightRegion> weight_regions_;
};

/// Preallocated per-session GPU scratch for flint decode steps: one
/// FlintGpuStepBuffers set sized from the config plus the fp32 KV cache for
/// `max_context` positions (qwen layout), and one stream. The KV cache is
/// written position by position and only ever read at [0, pos], so a
/// re-run of a sequence from position 0 is self-consistent without a reset.
class FlintGpuArena {
 public:
  explicit FlintGpuArena(const core::ModelConfig& config,
                         std::uint32_t max_context = 4096);
  ~FlintGpuArena();

  FlintGpuArena(const FlintGpuArena&) = delete;
  FlintGpuArena& operator=(const FlintGpuArena&) = delete;
  FlintGpuArena(FlintGpuArena&&) noexcept;
  FlintGpuArena& operator=(FlintGpuArena&&) noexcept;

  [[nodiscard]] FlintGpuStepBuffers GetBuffers() const noexcept {
    return buffers_;
  }
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return max_context_;
  }
  /// Device bytes owned by this arena (scratch + KV cache), for the
  /// memory-budget accounting in §9.
  [[nodiscard]] std::size_t GetScratchBytes() const noexcept {
    return scratch_bytes_;
  }

 private:
  void FreeAll() noexcept;

  core::ModelConfig config_;
  std::uint32_t max_context_;
  std::size_t scratch_bytes_{0};
  FlintGpuStepBuffers buffers_{};
};

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_FLINT_GPU_MODEL_HPP_
