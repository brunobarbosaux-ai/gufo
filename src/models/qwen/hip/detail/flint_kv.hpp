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

/// Flint KV-cache addressing per plane: fp32 is head-major
/// ((layer*kv_heads + kv_h) * max_context + pos) * head_dim; the f16 twin is
/// token-major ((layer*max_context + pos) * kv_width) + kv_h*head_dim — the
/// layout WriteKVCacheKernel / WriteKVCachePtrKernel /
/// WriteBatchedKVCacheKernel already write for the f16 planes and the qwen
/// batched/half kernels already read.
template <typename KVT>
__device__ __forceinline__ std::size_t FlintKvOffset(
    std::size_t layer_idx, std::uint32_t kv_h, std::size_t position,
    std::uint32_t num_kv_heads, std::uint32_t max_context,
    std::uint32_t head_dim) {
  if constexpr (std::is_same_v<KVT, float>) {
    return ((layer_idx * num_kv_heads + kv_h) * max_context + position) *
           head_dim;
  } else {
    return (layer_idx * max_context + position) *
               (static_cast<std::size_t>(num_kv_heads) * head_dim) +
           static_cast<std::size_t>(kv_h) * head_dim;
  }
}

}  // namespace gufo::hip::detail
#endif  // ENGINE_ENABLE_HIP

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_FLINT_KV_HPP_
