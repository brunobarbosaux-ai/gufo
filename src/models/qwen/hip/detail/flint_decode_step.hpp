#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_FLINT_DECODE_STEP_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_FLINT_DECODE_STEP_HPP_

#include <cstdint>
#include <limits>

#include "src/models/qwen/state.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Device buffers for one flint decode step. All pointers are device memory
/// owned by the caller (arena / test harness); the step reuses them across
/// layers and tokens without reallocation. Sizing (H = hidden_size,
/// A = num_heads * head_dim, S = kv_heads * head_dim,
/// F = max(dense ffn, expert ffn, shared ffn), E = expert_count):
///   hidden, normed, attn_proj, expert_out, mlp_out : H
///   q : A,  k, v : S,  attn_out : A,  act : F
///   router_logits : E,  selected / expert_weights : 8
///   logits : vocab_size
///   k_cache, v_cache : fp32 qwen layout
///     ((layer * kv_heads + kv) * max_context + pos) * head_dim + d
struct FlintGpuStepBuffers {
  std::uint32_t* d_out_token = nullptr;

  float* hidden = nullptr;
  float* normed = nullptr;
  float* q = nullptr;
  float* k = nullptr;
  float* v = nullptr;
  float* attn_out = nullptr;
  float* attn_proj = nullptr;
  float* act = nullptr;
  float* act_gate = nullptr;  ///< gate half for the f32 SwiGLU fallback
  float* expert_out = nullptr;
  float* mlp_out = nullptr;
  float* router_logits = nullptr;
  std::uint32_t* selected = nullptr;
  float* expert_weights = nullptr;
  float* logits = nullptr;

  float* k_cache = nullptr;
  float* v_cache = nullptr;
  std::uint32_t max_context = 0;

  hipStream_t stream = nullptr;
};

/// Runs one full flint decode step (embedding -> 49 attention layers ->
/// final norm -> unembed -> argmax) on the GPU. Every layer is attention
/// (no SSM); per-layer switches come from FlintLayerAttributes: RoPE is
/// skipped on NoPE layers (F-D3), windowed layers pass the sliding window
/// through FlintKernelPolicy (F-D2), and layers below
/// leading_dense_block_count use a dense SwiGLU FFN while the rest run the
/// sigmoid top-k MoE with one ungated shared expert (F-E1/F-E3).
///
/// The router runs on device; the selected expert ids and weights are read
/// back once per MoE layer so the host can offset the expert slabs for the
/// GEMV launches. That readback is the known optimization point (a fused
/// device-side MoE dispatch removes it); correctness first.
///
/// `pos` is the position of the token being decoded. `compute_logits` skips
/// the unembed + argmax when false.
void ExecuteFlintDecodeStep(const models::FlintModelWeights& weights,
                            std::uint32_t token_id, std::uint32_t pos,
                            bool compute_logits,
                            const FlintGpuStepBuffers& buffers);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_FLINT_DECODE_STEP_HPP_
