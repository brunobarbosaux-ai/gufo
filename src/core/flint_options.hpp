#ifndef GUFO_CORE_FLINT_OPTIONS_HPP_
#define GUFO_CORE_FLINT_OPTIONS_HPP_

#include <string>

namespace gufo::core {

/// Closes every runtime option the flint architecture cannot serve (F-B3,
/// F-B4): no MTP/nextn, no DFlash2, no DSpark drafting, and no vision
/// projector. Callers pass exactly what the CLI received so each rejection
/// names the offending flag; the first violation is reported.
[[nodiscard]] inline bool ValidateFlintOptions(bool has_mmproj,
                                               bool speculative_requested,
                                               bool has_mtp_model,
                                               bool has_dflash_model,
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
  if (has_dflash_model) {
    return reject(
        "flint does not support --dflash-model: DFlash2 drafting is closed "
        "for this architecture");
  }
  if (has_dspark_model) {
    return reject(
        "flint does not support --dspark-model: DSpark drafting is closed "
        "for this architecture");
  }
  if (speculative_requested) {
    return reject(
        "flint does not support --speculative: autoregressive-only build");
  }
  return true;
}

}  // namespace gufo::core

#endif  // GUFO_CORE_FLINT_OPTIONS_HPP_
