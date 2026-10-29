#ifndef GUFO_MODELS_QWEN_HIP_DFLASH_TARGET_HPP_
#define GUFO_MODELS_QWEN_HIP_DFLASH_TARGET_HPP_

#include "src/core/model_config.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::hip {

/// The target-model surface a DFlash-2 draft needs: the tied token embedding
/// and LM head, plus the hidden/vocab geometry. Implemented by every GPU target
/// model that can host a DFlash draft (Qwen and Flint), so the draft no longer
/// depends on a concrete target type. The draft keeps a shared_ptr to this
/// interface, which also keeps the target's device weights alive.
class DFlashTargetWeights {
 public:
  virtual ~DFlashTargetWeights() = default;

  [[nodiscard]] virtual const core::ModelConfig& GetConfig() const noexcept = 0;
  [[nodiscard]] virtual const tokenization::QwenTokenizer& GetTokenizer()
      const noexcept = 0;
  [[nodiscard]] virtual const models::QwenTensorRef& GetTokenEmbedding()
      const noexcept = 0;
  [[nodiscard]] virtual const models::QwenTensorRef& GetOutputHead()
      const noexcept = 0;
};

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_DFLASH_TARGET_HPP_
