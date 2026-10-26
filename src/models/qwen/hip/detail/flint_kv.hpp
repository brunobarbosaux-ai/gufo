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

/// Ring row for a KV position: windowed layers wrap at `min(max_context,
/// window)` rows, global layers index the position directly.
__device__ __forceinline__ std::size_t FlintKvRow(
    std::size_t position, std::uint32_t window, std::uint32_t max_context) {
  if (window == 0) {
    return position;
  }
  const std::uint32_t rows = window < max_context ? window : max_context;
  return position % rows;
}

/// Flint KV-cache addressing per plane: fp32 is head-major
/// (layer_base + (kv_h * rows + row) * head_dim); the f16 twin is token-major
/// (layer_base + row * kv_width + kv_h * head_dim) — the layout
/// WriteKVCacheKernel / WriteKVCachePtrKernel / WriteBatchedKVCacheKernel
/// already write for the f16 planes and the qwen batched/half kernels already
/// read.
///
/// F-I2 ring cache: `window` is 0 for a global layer (row = position, rows =
/// max_context) and the sliding window (513) for a windowed layer, whose KV is
/// a ring of `min(max_context, window)` rows per sequence (row = position %
/// rows). `layer_base` is the element offset of this layer's plane; K and V
/// planes share the same per-layer base and row count, so one value serves
/// both.
template <typename KVT>
__device__ __forceinline__ std::size_t FlintKvOffset(
    std::size_t layer_base, std::uint32_t kv_h, std::size_t position,
    std::uint32_t window, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::uint32_t rows =
      window != 0 ? (window < max_context ? window : max_context)
                  : max_context;
  const std::size_t row = window != 0 ? (position % rows) : position;
  if constexpr (std::is_same_v<KVT, float>) {
    return layer_base +
           (static_cast<std::size_t>(kv_h) * rows + row) * head_dim;
  } else {
    return layer_base +
           row * (static_cast<std::size_t>(num_kv_heads) * head_dim) +
           static_cast<std::size_t>(kv_h) * head_dim;
  }
}

}  // namespace gufo::hip::detail
#endif  // ENGINE_ENABLE_HIP

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_FLINT_KV_HPP_
