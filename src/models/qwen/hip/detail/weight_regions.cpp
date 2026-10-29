#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/detail/weight_regions.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/hip/heap_placement.hpp"
#include "src/core/hip/weight_upload.hpp"
#include "src/core/platform/pages.hpp"
#include "src/core/platform/prefault.hpp"

namespace gufo::hip::detail {
namespace {

[[nodiscard]] hipError_t MapRegisteredRegion(
    const core::GgufMappedRegion& source, QwenGpuWeightRegion& destination,
    bool& source_was_read) {
  source_was_read = false;
  void* host_copy =
      platform::CloneMappedWeightsToAnonymousPages(source.data, source.size);
  if (host_copy == nullptr)
    return hipErrorOutOfMemory;
  source_was_read = true;
  const auto unmap = [&](void* pointer) {
    platform::ReleaseCommittedPages(pointer, source.size);
  };
  std::unique_ptr<void, decltype(unmap)> owned_copy(host_copy, unmap);
  const auto register_error = hipHostRegister(
      host_copy, source.size, hipHostRegisterMapped | hipHostRegisterReadOnly);
  if (register_error != hipSuccess)
    return register_error;

  void* device_data = nullptr;
  const auto pointer_error =
      hipHostGetDevicePointer(&device_data, host_copy, 0);
  if (pointer_error != hipSuccess) {
    (void)hipHostUnregister(host_copy);
    return pointer_error;
  }

  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .host_copy = owned_copy.release(),
                 .size = source.size};
  return hipSuccess;
}

/// Device-resident placement: one hipMalloc copy in (on this APU target)
/// carve-out memory, filled from the cached mapping or by FillDeviceRegions.
[[nodiscard]] hipError_t AllocateDeviceRegion(
    const core::GgufMappedRegion& source, QwenGpuWeightRegion& destination) {
  void* device_data = nullptr;
  const auto status = hipMalloc(&device_data, source.size);
  if (status != hipSuccess)
    return status;
  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .host_copy = nullptr,
                 .size = source.size,
                 .owns_device_allocation = true};
  return hipSuccess;
}

/// Streams one region from the read-only mapping into device memory. Where
/// the platform can drop the mapping's pages as it goes, eight workers each
/// copy 64 MiB chunks through their own pinned staging buffer, so a shard in
/// the page cache uploads at memory speed. A hipMemcpy straight from the
/// pageable mapping pins it page by page and runs at a small fraction of
/// that. Elsewhere one staging buffer streams the prefaulted mapping; that is
/// the fallback for sources WeightUpload cannot read, such as in-memory images
/// without a file.
[[nodiscard]] hipError_t CopyMappedRegion(const core::GgufMappedRegion& source,
                                          void* device_data) {
  const auto free_staging = [](void* pointer) { (void)hipHostFree(pointer); };
  using Staging = std::unique_ptr<void, decltype(free_staging)>;
  if (platform::CanVisitMappedChunks()) {
    constexpr std::size_t kChunkBytes = 64ULL << 20;
    constexpr std::size_t kWorkers = 8;
    std::vector<Staging> staging;
    staging.reserve(kWorkers);
    for (std::size_t i = 0; i < kWorkers; ++i)
      staging.emplace_back(nullptr, free_staging);
    std::atomic<hipError_t> failure{hipSuccess};
    platform::VisitMappedChunks(
        source.data, source.size, kChunkBytes, kWorkers,
        [&](std::size_t worker, std::size_t offset, const void* chunk,
            std::size_t size) {
          auto& buffer = staging[worker];
          if (!buffer) {
            void* pointer = nullptr;
            if (const auto status = hipHostMalloc(&pointer, kChunkBytes);
                status != hipSuccess) {
              failure.store(status, std::memory_order_relaxed);
              return false;
            }
            buffer.reset(pointer);
          }
          std::memcpy(buffer.get(), chunk, size);
          const auto status =
              hipMemcpy(static_cast<std::uint8_t*>(device_data) + offset,
                        buffer.get(), size, hipMemcpyHostToDevice);
          if (status != hipSuccess) {
            failure.store(status, std::memory_order_relaxed);
            return false;
          }
          return true;
        });
    return failure.load();
  }

  platform::PrefaultMappedRange(source.data, source.size);

  constexpr std::size_t kChunkBytes = 16ULL << 20;
  void* staging = nullptr;
  hipError_t status = hipHostMalloc(&staging, kChunkBytes);
  if (status != hipSuccess)
    return status;
  Staging owned_staging(staging, free_staging);

  const auto chunks =
      source.size / kChunkBytes + (source.size % kChunkBytes != 0);
  for (std::size_t index = 0; index < chunks; ++index) {
    const auto offset = index * kChunkBytes;
    const auto bytes = std::min(kChunkBytes, source.size - offset);
    std::memcpy(staging, static_cast<const std::uint8_t*>(source.data) + offset,
                bytes);
    status = hipMemcpy(static_cast<std::uint8_t*>(device_data) + offset,
                       staging, bytes, hipMemcpyHostToDevice);
    if (status != hipSuccess)
      return status;
  }
  return hipSuccess;
}

/// Fills device-resident regions from their shard files without touching the
/// mappings. Copying through a mapping faults the whole shard into host memory
/// first; on Windows those pages also stay in the process working set, so a
/// 16 GiB target needs as much free RAM beside the GPU carve-out. WeightUpload
/// keeps host residency at its staging buffers.
[[nodiscard]] bool FillDeviceRegions(
    std::span<const core::GgufMappedRegion> sources,
    std::span<QwenGpuWeightRegion> destinations,
    std::span<const std::size_t> indices, std::string* error_msg) {
  std::string upload_error;
  if (auto uploader = WeightUpload::Create(sources, &upload_error)) {
    for (const auto index : indices) {
      if (!uploader->Copy(static_cast<std::uint32_t>(index), 0,
                          sources[index].size, destinations[index].device_data,
                          error_msg))
        break;
    }
    // Finish waits for every queued copy, also after a failed Copy, so the
    // destinations stay valid until the uploader is gone.
    return uploader->Finish(error_msg);
  }
  for (const auto index : indices) {
    const auto status =
        CopyMappedRegion(sources[index], destinations[index].device_data);
    if (status != hipSuccess) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to upload GGUF shard " + std::to_string(index) +
                     ": " + hipGetErrorString(status);
      }
      return false;
    }
  }
  return true;
}

}  // namespace

void ReleaseWeightRegions(std::vector<QwenGpuWeightRegion>& regions) noexcept {
  for (auto& region : regions) {
    if (region.owns_device_allocation && region.device_data != nullptr) {
      (void)hipFree(region.device_data);
    } else if (region.host_copy != nullptr) {
      (void)hipHostUnregister(region.host_copy);
      platform::ReleaseCommittedPages(region.host_copy, region.size);
    }
    region = {};
  }
}

bool CreateWeightRegions(const core::GgufReader& reader,
                         std::vector<QwenGpuWeightRegion>& weight_regions,
                         std::string* error_msg) {
  const auto source_regions = reader.GetMappedRegions();
  if (source_regions.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader has no mapped weight regions";
    }
    return false;
  }

  weight_regions.resize(source_regions.size());
  std::vector<std::size_t> device_regions;
  for (std::size_t i = 0; i < source_regions.size(); ++i) {
    const auto& source = source_regions[i];
    auto& destination = weight_regions[i];
    hipError_t map_error;
    try {
      // Device memory is not on the kernel's page LRU, so compaction never
      // migrates it. A registered anonymous copy is: every migration makes
      // the driver evict and restore the GPU mapping, and kernels reading it
      // stall for seconds.
      // Read through the page cache where the platform allows it, so a
      // warm reload does not wait on the disk; otherwise (Windows) the
      // bounded file pipeline fills the region without touching the mapping.
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess &&
          HeapFitsAboveReserve(free_bytes, source.size) &&
          AllocateDeviceRegion(source, destination) == hipSuccess) {
        if (platform::CanVisitMappedChunks())
          map_error = CopyMappedRegion(source, destination.device_data);
        else {
          map_error = hipSuccess;
          device_regions.push_back(i);
        }
      } else {
        (void)hipGetLastError();
        destination = {};
        bool source_was_read = false;
        map_error = MapRegisteredRegion(source, destination, source_was_read);
        if (map_error != hipSuccess) {
          // The device-resident upload preserves single-copy residency
          // wherever registration is refused.
          destination = {};
          map_error = AllocateDeviceRegion(source, destination);
          if (map_error == hipSuccess) {
            if (source_was_read) {
              // A successful clone has already read the mapping. Reuse those
              // cached bytes if registration fails instead of rereading the
              // shard with direct I/O. When cloning is unavailable (Windows),
              // keep the bounded file pipeline without touching the mapping.
              map_error = CopyMappedRegion(source, destination.device_data);
            } else {
              device_regions.push_back(i);
            }
          }
        }
      }
    } catch (const std::exception& e) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg)
        *error_msg = e.what();
      return false;
    }
    if (map_error != hipSuccess || destination.device_data == nullptr) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg != nullptr) {
        *error_msg = "Failed to make GGUF shard " + std::to_string(i) +
                     " GPU-visible: " + hipGetErrorString(map_error);
      }
      return false;
    }
  }
  if (!device_regions.empty() &&
      !FillDeviceRegions(source_regions, weight_regions, device_regions,
                         error_msg)) {
    ReleaseWeightRegions(weight_regions);
    return false;
  }
  return true;
}

bool RemapTensor(models::QwenTensorRef& tensor,
                 std::span<const QwenGpuWeightRegion> regions) noexcept {
  if (tensor.empty()) {
    return true;
  }
  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor.data);
  for (const auto& region : regions) {
    const auto region_address =
        reinterpret_cast<std::uintptr_t>(region.host_data);
    if (tensor_address >= region_address) {
      const auto offset = tensor_address - region_address;
      const std::size_t encoded_bytes = tensor.EncodedSizeBytes();
      if (offset < region.size && encoded_bytes != 0 &&
          encoded_bytes <= region.size - offset) {
        tensor.data =
            static_cast<const std::uint8_t*>(region.device_data) + offset;
        tensor.available_bytes = region.size - offset;
        return true;
      }
    }
  }
  return false;
}

}  // namespace gufo::hip::detail
#endif  // defined(ENGINE_ENABLE_HIP)
