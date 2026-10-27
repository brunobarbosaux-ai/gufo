#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_FLINT_KV_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_FLINT_KV_HPP_

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>

namespace gufo::hip::detail {

/// Flint KV-cache element load (f16-KV plan): the fp32 plane is read raw,
/// the f16 twin converts on load. Templated so one kernel body serves both
/// planes and the dispatch picks the instantiation from the arena's cache
/// type.
template <typename KVT>
__device__ __forceinline__ float FlintKvLoad(const KVT* __restrict__ ptr) {
  if constexpr (std::is_same_v<KVT, float>) {
    return *ptr;
  } else {
    return __half2float(*ptr);
  }
}

/// Ring row for a KV position. `rows` is the per-layer plane row count:
/// 0 = legacy/uniform plane and `rows == max_context` = a global layer, both
/// index the position directly; a windowed ring is allocated at a power of
/// two, so the wrap is a mask rather than a divide (the F-J4 decode
/// regression was a runtime modulo per key access).
__device__ __forceinline__ std::size_t FlintKvRow(
    std::size_t position, std::uint32_t rows, std::uint32_t max_context) {
  if (rows == 0 || rows == max_context) {
    return position;
  }
  return position & (static_cast<std::size_t>(rows) - 1U);
}

/// Flint KV-cache addressing per plane: fp32 is head-major
/// (layer_base + (kv_h * rows + row) * head_dim); the f16 twin is token-major
/// (layer_base + row * kv_width + kv_h * head_dim) — the layout
/// WriteKVCacheKernel / WriteKVCachePtrKernel / WriteBatchedKVCacheKernel
/// already write for the f16 planes and the qwen batched/half kernels already
/// read.
///
/// F-I2 ring cache: `rows` is the per-layer plane row count (0 = legacy /
/// uniform plane; `max_context` = global layer; a power of two < max_context
/// = windowed ring). `layer_base` is the element offset of this layer's
/// plane; K and V planes share the same per-layer base and row count, so one
/// value serves both.
template <typename KVT>
__device__ __forceinline__ std::size_t FlintKvOffset(
    std::size_t layer_base, std::uint32_t kv_h, std::size_t position,
    std::uint32_t rows, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::size_t stride_rows =
      (rows == 0) ? static_cast<std::size_t>(max_context)
                  : static_cast<std::size_t>(rows);
  const std::size_t row = FlintKvRow(position, rows, max_context);
  if constexpr (std::is_same_v<KVT, float>) {
    return layer_base +
           (static_cast<std::size_t>(kv_h) * stride_rows + row) * head_dim;
  } else {
    return layer_base +
           row * (static_cast<std::size_t>(num_kv_heads) * head_dim) +
           static_cast<std::size_t>(kv_h) * head_dim;
  }
}

}  // namespace gufo::hip::detail
#endif  // ENGINE_ENABLE_HIP

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_FLINT_KV_HPP_
