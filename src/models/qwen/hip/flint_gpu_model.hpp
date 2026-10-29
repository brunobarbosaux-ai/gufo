#ifndef GUFO_MODELS_QWEN_HIP_FLINT_GPU_MODEL_HPP_
#define GUFO_MODELS_QWEN_HIP_FLINT_GPU_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/detail/flint_decode_step.hpp"
#include "src/models/qwen/hip/detail/flint_prefill.hpp"
#include "src/models/qwen/hip/dflash_target.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/state.hpp"

#if defined(ENGINE_ENABLE_HIP)

namespace gufo::hip {

/// Immutable GPU-visible flint model: the exact flint tensor inventory
/// (F-C1) re-pointed at registered, GPU-visible copies of the GGUF shards —
/// the same mapped-registration path QwenGpuModel uses, so no per-tensor
/// device copies. The reader is retained because the region offsets are
/// relative to its original mappings.
class FlintGpuModel : public DFlashTargetWeights {
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
  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept override {
    return weights_.config;
  }
  [[nodiscard]] const models::QwenTensorRef& GetTokenEmbedding()
      const noexcept override {
    return weights_.token_embd;
  }
  [[nodiscard]] const models::QwenTensorRef& GetOutputHead()
      const noexcept override {
    return weights_.output;
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

/// Identity of the flint KV geometry (F-I2 ring cache). Two arenas with equal
/// classes lay out their KV planes identically, so a snapshot taken under one
/// class can be restored under the other. `rows_checksum` is an FNV-1a hash
/// over every layer's (windowed flag, row modulus), so it covers the full
/// sliding-window pattern at any layer count, not just the first 64 layers.
struct FlintKvWindowClass {
  std::uint32_t sliding_window{0};
  std::uint32_t num_layers{0};
  std::uint32_t max_context{0};
  std::uint64_t rows_checksum{0};

  friend bool operator==(const FlintKvWindowClass&,
                         const FlintKvWindowClass&) = default;
};

/// Preallocated per-session GPU scratch for flint decode steps: one
/// FlintGpuStepBuffers set sized from the config plus the fp32 KV cache for
/// `max_context` positions (qwen layout), and one stream. The KV cache is
/// written position by position and only ever read at [0, pos], so a
/// re-run of a sequence from position 0 is self-consistent without a reset.
class FlintGpuArena {
 public:
  explicit FlintGpuArena(const core::ModelConfig& config,
                         std::uint32_t max_context = 4096,
                         std::span<const std::uint32_t> target_layer_ids = {});
  ~FlintGpuArena();

  FlintGpuArena(const FlintGpuArena&) = delete;
  FlintGpuArena& operator=(const FlintGpuArena&) = delete;
  FlintGpuArena(FlintGpuArena&&) noexcept;
  FlintGpuArena& operator=(FlintGpuArena&&) noexcept;

  [[nodiscard]] FlintGpuStepBuffers GetBuffers() const noexcept {
    return buffers_;
  }
  /// Capture-once decode graph state (3.1); owned by the arena, released in
  /// FreeAll. Callers pass it to ExecuteFlintDecodeStep to opt into graph
  /// replay.
  [[nodiscard]] FlintDecodeGraphState* GetGraphState() noexcept {
    return &graph_state_;
  }
  [[nodiscard]] FlintPrefillBuffers GetPrefillBuffers() const noexcept {
    return prefill_;
  }
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return max_context_;
  }
  /// F-I2 ring cache: per-layer KV plane element offsets (K and V share
  /// them), indexed by layer_idx. Windowed layers hold fewer rows than
  /// max_context.
  [[nodiscard]] const std::vector<std::size_t>& GetKvLayerBase() const noexcept {
    return kv_layer_base_;
  }
  /// F-I2 ring cache: per-layer KV row count (the row modulus), indexed by
  /// layer_idx. Windowed layers hold fewer rows than max_context.
  [[nodiscard]] const std::vector<std::uint32_t>& GetKvLayerRows() const noexcept {
    return kv_layer_rows_;
  }
  /// Element count of one fp32 KV plane (K and V are each this many floats).
  [[nodiscard]] std::size_t GetKvPlaneElements() const noexcept {
    return kv_plane_elements_;
  }
  /// Device bytes owned by this arena (scratch + KV cache), for the
  /// memory-budget accounting in §9.
  [[nodiscard]] std::size_t GetScratchBytes() const noexcept {
    return scratch_bytes_;
  }
  /// KV geometry identity of this arena (F-I2 ring cache), for snapshot
  /// compatibility checks.
  [[nodiscard]] FlintKvWindowClass WindowClass() const noexcept;
  /// True when a snapshot taken under `other` can be restored into this
  /// arena. Rejects any KV geometry mismatch (5.3c restore gate).
  [[nodiscard]] bool CompatibleWith(
      const FlintKvWindowClass& other) const noexcept {
    return WindowClass() == other;
  }

 private:
  void FreeAll() noexcept;

  core::ModelConfig config_;
  std::uint32_t max_context_;
  std::size_t scratch_bytes_{0};
  std::vector<std::size_t> kv_layer_base_;
  std::vector<std::uint32_t> kv_layer_rows_;
  std::size_t kv_plane_elements_{0};
  /// DFlash-2 target-layer taps (7.7). Empty when DFlash is off, which leaves
  /// the capture buffers null and the decode/prefill paths unchanged.
  std::vector<std::uint32_t> target_layer_ids_;
  FlintGpuStepBuffers buffers_{};
  FlintPrefillBuffers prefill_{};
  FlintDecodeGraphState graph_state_{};
};

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_FLINT_GPU_MODEL_HPP_
