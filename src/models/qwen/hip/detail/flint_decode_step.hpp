#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_FLINT_DECODE_STEP_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_FLINT_DECODE_STEP_HPP_

#include <cstdint>
#include <limits>

#include "src/models/qwen/state.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/detail/hip_graph_decode_executor.hpp"

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
  /// Fused-dispatch (F-J1): renormalised weight per expert, zero for
  /// non-selected, shared expert at slot expert_count. expert_count + 1.
  float* expert_weights_full = nullptr;
  /// Fused-dispatch (F-J1): per-expert SwiGLU activations,
  /// expert_count * expert_ff + shared_ff floats.
  float* act_moe = nullptr;
  /// Fused-dispatch (F-J1): compacted job list written by the router
  /// (selected ids in top-k order, shared expert last, -1 padded),
  /// kFlintMoeMaxJobs ints. The fused kernels index experts through it so
  /// their grids span only real jobs.
  int* expert_job_list = nullptr;
  /// Deterministic MoE accumulation (3.6): per-job-slot down-projection
  /// partials, kFlintMoeMaxJobs * hidden floats. The down kernel writes the
  /// disjoint slot rows; FlintFusedMoEReduceKernel sums them in job order
  /// (the atomicAdd it replaces ordered float adds by wave schedule —
  /// run-to-run noise that flipped fragile argmax ties).
  float* moe_partials = nullptr;
  float* logits = nullptr;

  float* k_cache = nullptr;
  float* v_cache = nullptr;
  /// Token-major f16 twin planes (GUFO_FLINT_KV_F16, default on): written
  /// alongside fp32 by the KV-write kernels, read by the f16 attention
  /// instantiations — half the KV bytes on the global-layer decode steps.
  __half* k_cache_f16 = nullptr;
  __half* v_cache_f16 = nullptr;
  std::uint32_t max_context = 0;

  hipStream_t stream = nullptr;
};

/// Capture-once decode graph state (3.1). Owned by the arena; a pointer is
/// passed to ExecuteFlintDecodeStep. On the first step the step function
/// decides eligibility (fp32 KV plane, argmax output, every MoE layer on the
/// fused sync-free dispatch) and lazily instantiates one exec per tail
/// variant (with/without logits); every later step is one control-scalar
/// launch + one hipGraphLaunch. Per-step scalars (token id, position) reach
/// the kernels through d_control, written by LaunchFlintWriteControl OUTSIDE
/// the graph with the values as kernel arguments — enqueue-time semantics
/// keep queued-ahead replays race-free. Graph-mode attention is the
/// fixed-grid/fixed-LDS online-tile kernel at every position, so one capture
/// serves the whole context (the legacy baseline's seq-sized LDS score row
/// is not capturable and caps at 16128).
struct FlintDecodeGraphState {
  detail::HipGraphDecodeExecutor exec_logits;  ///< capture includes the tail
  detail::HipGraphDecodeExecutor exec_hidden;  ///< capture stops at layers
  std::uint32_t* d_control = nullptr;          ///< [0] = token id, [1] = position
  bool tried = false;      ///< eligibility decided (first step)
  bool supported = false;  ///< graph mode active

  void Release() noexcept;
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
/// the unembed + argmax when false. When `timings` is non-null the step is
/// instrumented with stream-ordered event marks (~500 event creates per
/// call — profiling only; see FlintDecodeTimings below).
struct FlintDecodeTimings {
  double embed_ms = 0;
  double attn_norm_qkv_ms = 0;   ///< RMSNorm + Q/K/V GEMVs
  double head_rope_ms = 0;       ///< per-head norms + RoPE
  double attention_ms = 0;       ///< decode attention incl. KV write
  double oproj_ms = 0;           ///< O-projection + residual
  double ffn_norm_router_ms = 0; ///< FFN norm + router GEMV
  double moe_ms = 0;             ///< MoE branch (or dense FFN)
  double residual_ms = 0;        ///< post-FFN residual add
  double tail_ms = 0;            ///< final norm + unembed + argmax
  double total_ms = 0;
  std::size_t moe_layers = 0;
  std::size_t dense_layers = 0;
};

void ExecuteFlintDecodeStep(const models::FlintModelWeights& weights,
                            std::uint32_t token_id, std::uint32_t pos,
                            bool compute_logits,
                            const FlintGpuStepBuffers& buffers,
                            FlintDecodeTimings* timings = nullptr,
                            FlintDecodeGraphState* graph = nullptr);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_FLINT_DECODE_STEP_HPP_
