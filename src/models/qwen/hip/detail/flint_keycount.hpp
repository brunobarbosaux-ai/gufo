#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_FLINT_KEYCOUNT_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_FLINT_KEYCOUNT_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "src/core/hip/hip_utils.hpp"

namespace gufo::hip::detail {

// F-J3 visited-key counter. One slot per layer; each attention kernel records
// the max number of keys any query in the launch visited. Gated by a device
// flag: production runs (flag 0) skip the atomic, so the graph-captured
// kernels are unchanged when off.
//
// Without -fgpu-rdc every .hip translation unit owns a private copy of any
// __device__ variable (verified: inline and weak linkage do not merge them),
// so each TU that holds instrumented kernels defines its own counter and
// set/max-into functions with GUFO_FLINT_KEYCOUNT_TU(suffix); the public
// FlintKeyCountSet/Read in attention_batched.hip fan out to all of them.
inline constexpr std::uint32_t kFlintKeyCountMaxLayers = 128;
static __device__ unsigned int g_flint_key_count[kFlintKeyCountMaxLayers];
static __device__ unsigned int g_flint_keycount_on;

__device__ __forceinline__ void FlintKeyCountAdd(std::uint32_t layer,
                                                 std::uint32_t span) {
  if (g_flint_keycount_on != 0u && layer < kFlintKeyCountMaxLayers) {
    atomicMax(&g_flint_key_count[layer], span);
  }
}

}  // namespace gufo::hip::detail

#define GUFO_FLINT_KEYCOUNT_TU(suffix)                                        \
  void FlintKeyCountSet_##suffix(bool on, hipStream_t stream) {               \
    const unsigned int flag = on ? 1u : 0u;                                   \
    HIP_CHECK(hipMemcpyToSymbolAsync(                                         \
        HIP_SYMBOL(detail::g_flint_keycount_on), &flag, sizeof(flag), 0,      \
        hipMemcpyHostToDevice, stream));                                      \
    const unsigned int zeros[detail::kFlintKeyCountMaxLayers] = {};           \
    HIP_CHECK(hipMemcpyToSymbolAsync(                                         \
        HIP_SYMBOL(detail::g_flint_key_count), zeros, sizeof(zeros), 0,       \
        hipMemcpyHostToDevice, stream));                                      \
  }                                                                           \
  void FlintKeyCountMaxInto_##suffix(unsigned int* acc, std::size_t layers,   \
                                     hipStream_t stream) {                    \
    unsigned int local[detail::kFlintKeyCountMaxLayers] = {};                 \
    const std::size_t n =                                                     \
        std::min<std::size_t>(layers, detail::kFlintKeyCountMaxLayers);       \
    HIP_CHECK(hipMemcpyFromSymbolAsync(                                       \
        local, HIP_SYMBOL(detail::g_flint_key_count),                         \
        n * sizeof(unsigned int), 0, hipMemcpyDeviceToHost, stream));         \
    HIP_CHECK(hipStreamSynchronize(stream));                                  \
    for (std::size_t i = 0; i < n; ++i) {                                     \
      acc[i] = std::max(acc[i], local[i]);                                    \
    }                                                                         \
  }

#endif  // ENGINE_ENABLE_HIP
#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_FLINT_KEYCOUNT_HPP_
