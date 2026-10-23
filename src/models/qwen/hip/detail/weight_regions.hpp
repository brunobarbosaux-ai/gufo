#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_WEIGHT_REGIONS_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_WEIGHT_REGIONS_HPP_

#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/state.hpp"

#if defined(ENGINE_ENABLE_HIP)

namespace gufo::hip::detail {

/// Model-agnostic weight-region machinery shared by every GPU model that
/// serves weights straight from mapped GGUF shards (QwenGpuModel,
/// FlintGpuModel): one registered, GPU-visible private copy per mapped
/// shard, plus the tensor re-pointing that swaps a ref's reader address for
/// its device address.

/// Unregisters and unmaps every region; idempotent.
void ReleaseWeightRegions(std::vector<QwenGpuWeightRegion>& regions) noexcept;

/// Creates one mapped, GPU-visible registered copy per mapped GGUF shard
/// (private mmap + chunked copy + hipHostRegister(Mapped|ReadOnly)). On
/// failure the vector is left empty and `error_msg` is set.
[[nodiscard]] bool CreateWeightRegions(
    const core::GgufReader& reader,
    std::vector<QwenGpuWeightRegion>& weight_regions, std::string* error_msg);

/// Re-points `tensor` from its reader mapping to the matching registered
/// region's device pointer. Empty tensors are a no-op; returns false when
/// the tensor does not fall inside any region's storage.
[[nodiscard]] bool RemapTensor(models::QwenTensorRef& tensor,
                               std::span<const QwenGpuWeightRegion> regions) noexcept;

}  // namespace gufo::hip::detail

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_WEIGHT_REGIONS_HPP_
