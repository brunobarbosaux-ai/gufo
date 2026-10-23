#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/detail/weight_regions.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <system_error>
#include <thread>
#include <vector>

namespace gufo::hip::detail {
namespace {

void CopyMappedWeights(const core::GgufMappedRegion& source, void* copy) {
  constexpr std::size_t kChunkBytes = 16ULL << 20;
  const auto chunks =
      source.size / kChunkBytes + (source.size % kChunkBytes != 0);
  std::atomic<std::size_t> next{0};
  std::atomic<int> failure{0};
  auto copy_chunk = [&] {
    while (failure.load(std::memory_order_relaxed) == 0) {
      const auto index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= chunks)
        break;
      const auto offset = index * kChunkBytes;
      const auto bytes = std::min(kChunkBytes, source.size - offset);
      auto* input = const_cast<std::uint8_t*>(
                        static_cast<const std::uint8_t*>(source.data)) +
                    offset;
      if (madvise(input, bytes, MADV_POPULATE_READ) != 0) {
        failure.store(errno, std::memory_order_relaxed);
        break;
      }
      std::memcpy(static_cast<std::uint8_t*>(copy) + offset, input, bytes);
      // The immutable copy now owns these resident bytes. Keep the original
      // mapping valid for the reader without retaining its populated PTEs.
      (void)madvise(input, bytes, MADV_DONTNEED);
    }
  };
  {
    std::vector<std::jthread> workers;
    for (std::size_t i = 1; i < std::min<std::size_t>(16, chunks); ++i)
      workers.emplace_back(copy_chunk);
    copy_chunk();
  }
  if (const auto error = failure.load(); error != 0)
    throw std::system_error(error, std::generic_category(),
                            "cannot read mapped Qwen weights");
}

[[nodiscard]] hipError_t MapRegisteredRegion(
    const core::GgufMappedRegion& source, QwenGpuWeightRegion& destination) {
  void* host_copy = mmap(nullptr, source.size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (host_copy == MAP_FAILED)
    return hipErrorOutOfMemory;
  const auto unmap = [&](void* pointer) { (void)munmap(pointer, source.size); };
  std::unique_ptr<void, decltype(unmap)> owned_copy(host_copy, unmap);
  (void)madvise(host_copy, source.size, MADV_HUGEPAGE);
  CopyMappedWeights(source, host_copy);
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

}  // namespace

void ReleaseWeightRegions(std::vector<QwenGpuWeightRegion>& regions) noexcept {
  for (auto& region : regions) {
    if (region.host_copy != nullptr) {
      (void)hipHostUnregister(region.host_copy);
      (void)munmap(region.host_copy, region.size);
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
  for (std::size_t i = 0; i < source_regions.size(); ++i) {
    const auto& source = source_regions[i];
    auto& destination = weight_regions[i];
    hipError_t map_error;
    try {
      map_error = MapRegisteredRegion(source, destination);
    } catch (const std::exception& e) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg)
        *error_msg = e.what();
      return false;
    }
    if (destination.device_data == nullptr) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg != nullptr) {
        *error_msg = "Failed to make GGUF shard " + std::to_string(i) +
                     " GPU-visible through mapped registration: " +
                     hipGetErrorString(map_error);
      }
      return false;
    }
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
