#ifndef GUFO_CORE_FLINT_OPTIONS_HPP_
#define GUFO_CORE_FLINT_OPTIONS_HPP_

#include <cstdint>
#include <string>

namespace gufo::core {

/// Default served context when the caller does not pin `-c` (F-I1): 32k keeps
/// the fp32 KV arena inside the 45 GiB budget at the default session count.
inline constexpr std::uint32_t kFlintDefaultContext = 32768;

/// Served-context caps (F-I1/F-I2). The window-aware ring cache (F-I2) holds
/// only `sliding_window + prefill chunk` rows for the windowed layers, so the
/// fp32 KV arena fits 262144 positions inside the 45 GiB budget; a full-KV
/// layout (no sliding window) stays at 32768.
inline constexpr std::uint32_t kFlintFullKvContextCap = 32768;
inline constexpr std::uint32_t kFlintRingContextCap = 262144;

/// Layout-dependent served-context cap: the ring cap when the model has a
/// sliding window (the ring is active), the full-KV cap otherwise.
[[nodiscard]] inline constexpr std::uint32_t FlintServedContextCap(
    std::uint32_t sliding_window) noexcept {
  return sliding_window != 0 ? kFlintRingContextCap : kFlintFullKvContextCap;
}

/// Closes every runtime option the flint architecture cannot serve (F-B3,
/// F-B4): no MTP/nextn, no DSpark drafting, and no vision projector. DFlash2
/// drafting is served (7.7) through the flint speculative target. Callers pass
/// exactly what the CLI received so each rejection names the offending flag;
/// the first violation is reported.
[[nodiscard]] inline bool ValidateFlintOptions(bool has_mmproj,
                                               bool speculative_requested,
                                               bool speculative_lookup,
                                               bool speculative_dflash,
                                               bool has_mtp_model,
                                               bool has_dspark_model,
                                               std::string* error_msg) {
  const auto reject = [error_msg](const std::string& message) {
    if (error_msg != nullptr) {
      *error_msg = message;
    }
    return false;
  };
  if (has_mmproj) {
    return reject(
        "flint does not support --mmproj: the artifact carries no vision "
        "tensors");
  }
  if (has_mtp_model) {
    return reject(
        "flint has no MTP/nextn layers; --mtp-model is not supported");
  }
  if (has_dspark_model) {
    return reject(
        "flint does not support --dspark-model: DSpark drafting is closed "
        "for this architecture");
  }
  if (speculative_requested && !speculative_lookup && !speculative_dflash) {
    return reject(
        "flint does not support --speculative: only the lookup and DFlash2 "
        "backends are available");
  }
  return true;
}

}  // namespace gufo::core

#endif  // GUFO_CORE_FLINT_OPTIONS_HPP_
