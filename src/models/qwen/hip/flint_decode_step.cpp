// Flint GPU decode step (F-D dispatch entry): one full autoregressive step
// over the attention-only 49-layer stack. Mirrors the validated CPU
// reference (flint_forward.cpp) op for op: RMSNorm -> QKV GEMV -> per-head
// QK norm -> RoPE (NoPE layers skip) -> sink+window attention -> O-proj ->
// residual -> FFN (dense SwiGLU lead layers, sigmoid top-k MoE + ungated
// shared expert elsewhere) -> residual; final norm -> unembed -> argmax.
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/detail/flint_decode_step.hpp"

#include <cstdint>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/hip/ops/token.hpp"

namespace gufo::hip {

namespace {

constexpr float kFlintRmsEps = 1e-6F;
constexpr std::size_t kMaxSelectedExperts = 8;

/// Device-side twin of the CPU ExpertSlab: one expert's contiguous slab
/// inside a 3D stack tensor, offset by encoded bytes (block-aligned).
[[nodiscard]] const void* ExpertSlabData(const models::QwenTensorRef& tensor,
                                         std::size_t expert,
                                         std::size_t slab_elements) noexcept {
  const std::size_t byte_offset =
      quant::EncodedSizeBytes(tensor.type, expert * slab_elements);
  return static_cast<const std::uint8_t*>(tensor.data) + byte_offset;
}

/// QwenTensorRef view of one expert slab (for helpers taking refs).
[[nodiscard]] models::QwenTensorRef SlabRef(const models::QwenTensorRef& tensor,
                                            std::size_t expert,
                                            std::size_t slab_elements) noexcept {
  auto slab = tensor;
  slab.data = ExpertSlabData(tensor, expert, slab_elements);
  slab.num_elements = slab_elements;
  return slab;
}

/// SwiGLU for one [intermediate, hidden] gate/up pair. The fused kernel
/// covers BF16 and quantized weights; f32 pairs (synthetic tests, future
/// f32 exports) fall back to two GEMVs + the batched activation.
void RunSwiGLU(const models::QwenTensorRef& gate,
               const models::QwenTensorRef& up, const float* x, float* act,
               float* gate_scratch, std::size_t intermediate_size,
               std::size_t hidden_size, hipStream_t stream) {
  if (gate.type == core::GgmlType::kF32 && up.type == core::GgmlType::kF32) {
    LaunchGEMV(gate.data, gate.type, x, gate_scratch, intermediate_size,
               hidden_size, stream);
    LaunchGEMV(up.data, up.type, x, act, intermediate_size, hidden_size,
               stream);
    LaunchBatchedSwiGLUActivation(gate_scratch, act, act, nullptr,
                                  intermediate_size, stream);
    return;
  }
  LaunchFusedSwiGLUGEMV(gate.data, gate.type, up.data, up.type, x, act,
                        intermediate_size, hidden_size, stream);
}

/// Types the fused MoE kernels can decode: exactly the switch in
/// QuantWarpBlockDot (Q8_0 plus the sub-16 decoded set). Anything else
/// (f32 fixtures, BF16, Q8_K) must stay on the legacy readback path.
[[nodiscard]] bool FusedMoESupported(core::GgmlType t) noexcept {
  return t == core::GgmlType::kQ8_0 || t == core::GgmlType::kQ4_K ||
         t == core::GgmlType::kQ5_K || t == core::GgmlType::kQ6_K ||
         t == core::GgmlType::kQ3_K || t == core::GgmlType::kIQ4_NL ||
         t == core::GgmlType::kIQ4_XS || t == core::GgmlType::kIQ3_S;
}

void RunFlintLayer(const models::FlintModelWeights& weights,
                   const models::FlintLayerWeights& layer,
                   std::uint32_t layer_idx, std::uint32_t pos,
                   const FlintGpuStepBuffers& b,
                   FlintDecodeTimings* timings,
                   std::vector<hipEvent_t>* marks,
                   char* is_moe_layer) {
  const auto& config = weights.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t attention_size =
      static_cast<std::size_t>(config.num_attention_heads) * config.head_dim;
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;

  // Event marks are stream-ordered; the caller resolves stage sums after
  // one final sync. Eight boundaries per layer: [0]=norm+qkv start,
  // [1]=head-norm start, [2]=attention start, [3]=oproj start, [4]=ffn
  // norm+router start, [5]=moe start, [6]=residual start, [7]=layer end.
  auto mark = [&](std::size_t i) {
    if (marks != nullptr) {
      HIP_CHECK(hipEventRecord((*marks)[layer_idx * 8 + i], b.stream));
    }
  };
  mark(0);
  // 1. Pre-attention RMSNorm.
  LaunchRMSNorm(b.hidden, static_cast<const float*>(layer.attn_norm.data),
                b.normed, hidden_size, kFlintRmsEps, b.stream);

  // 2. Q/K/V projections (flint has no interleaved attention gate).
  LaunchGEMV(layer.attn_q.data, layer.attn_q.type, b.normed, b.q,
             attention_size, hidden_size, b.stream);
  LaunchGEMV(layer.attn_k.data, layer.attn_k.type, b.normed, b.k, kv_size,
             hidden_size, b.stream);
  LaunchGEMV(layer.attn_v.data, layer.attn_v.type, b.normed, b.v, kv_size,
             hidden_size, b.stream);

  mark(1);
  // 3. Per-head Q/K RMSNorm.
  LaunchPerHeadRMSNorm(b.q, static_cast<const float*>(layer.attn_q_norm.data),
                       b.q, config.num_attention_heads, config.head_dim,
                       kFlintRmsEps, b.stream);
  LaunchPerHeadRMSNorm(b.k, static_cast<const float*>(layer.attn_k_norm.data),
                       b.k, config.num_key_value_heads, config.head_dim,
                       kFlintRmsEps, b.stream);

  // 4. RoPE on rope layers; NoPE layers keep position-invariant Q/K (F-D3).
  if (layer.attrs.rope) {
    LaunchRoPE(b.q, b.k, config.num_attention_heads, config.num_key_value_heads,
               config.head_dim, config.rotary_dim, pos, config.rope_theta,
               b.stream);
  }

  mark(2);
  // 5. Attention with sinks + optional sliding window (F-D1/F-D2). The
  // policy gates dispatch to the sink-capable baseline kernels.
  const detail::FlintKernelPolicy policy{
      static_cast<const float*>(layer.attn_sinks.data),
      layer.attrs.windowed ? config.sliding_window : 0U};
  LaunchAttention(b.q, b.k, b.v, /*gate=*/nullptr, b.k_cache, b.v_cache,
                  /*k_cache_f16=*/nullptr, /*v_cache_f16=*/nullptr, b.attn_out,
                  layer_idx, pos, b.max_context, config.num_attention_heads,
                  config.num_key_value_heads, config.head_dim, b.stream,
                  /*split_k_scratch=*/nullptr, /*skip_kv_write=*/false,
                  &policy);

  // 6. O-projection + residual.
  mark(3);
  LaunchGEMV(layer.attn_output.data, layer.attn_output.type, b.attn_out,
             b.attn_proj, hidden_size, attention_size, b.stream);
  LaunchResidualAdd(b.hidden, b.attn_proj, b.hidden, hidden_size, b.stream);

  // 7. FFN: dense SwiGLU on the lead layers, MoE elsewhere.
  mark(4);
  LaunchRMSNorm(b.hidden, static_cast<const float*>(layer.ffn_norm.data),
                b.normed, hidden_size, kFlintRmsEps, b.stream);
  if (layer.attrs.dense_ffn) {
    if (is_moe_layer != nullptr) {
      *is_moe_layer = 0;
    }
    mark(5);
    RunSwiGLU(layer.ffn_gate, layer.ffn_up, b.normed, b.act, b.act_gate,
              config.intermediate_size, hidden_size, b.stream);
    LaunchGEMV(layer.ffn_down.data, layer.ffn_down.type, b.act, b.mlp_out,
               hidden_size, config.intermediate_size, b.stream);
  } else {
    const std::size_t expert_ff = config.expert_feed_forward_length;
    const std::size_t expert_slab = expert_ff * hidden_size;
    const std::size_t shared_ff = config.expert_shared_feed_forward_length;
    const std::uint32_t used = config.expert_used_count;

    LaunchGEMV(layer.ffn_gate_inp.data, layer.ffn_gate_inp.type, b.normed,
               b.router_logits, config.expert_count, hidden_size, b.stream);

    if (is_moe_layer != nullptr) {
      *is_moe_layer = 1;
    }
    mark(5);

    // F-J1 fused dispatch: the router writes the full weight array on
    // device and compacts the selected ids into the job list; the two
    // fused kernels index experts through the job list (shared expert
    // included) — no readback sync, no per-selected-expert launches, and
    // grids spanning only real jobs. Four launches for the whole branch.
    // Per-kernel type consistency only: the SwiGLU kernel decodes both
    // routed stacks with one type (gate == up), the down kernel uses the
    // down type, and each shared matrix decodes with its own type.
    // UD-Q6_K_XL is mixed like this on every MoE layer (gate/up Q6_K,
    // down Q8_0, shared Q8_0).
    const bool fused =
        FusedMoESupported(layer.ffn_gate_exps.type) &&
        FusedMoESupported(layer.ffn_up_exps.type) &&
        FusedMoESupported(layer.ffn_down_exps.type) &&
        FusedMoESupported(layer.ffn_gate_shexp.type) &&
        FusedMoESupported(layer.ffn_up_shexp.type) &&
        FusedMoESupported(layer.ffn_down_shexp.type) &&
        used + 1 <= kFlintMoeMaxJobs &&
        b.expert_weights_full != nullptr && b.act_moe != nullptr &&
        b.expert_job_list != nullptr;
    if (fused) {
      const std::size_t gate_slab_bytes =
          quant::EncodedSizeBytes(layer.ffn_gate_exps.type, expert_slab);
      const std::size_t up_slab_bytes =
          quant::EncodedSizeBytes(layer.ffn_up_exps.type, expert_slab);
      const std::size_t down_slab_bytes =
          quant::EncodedSizeBytes(layer.ffn_down_exps.type, expert_slab);
      LaunchFlintRouterTopKFull(b.router_logits, config.expert_count, used,
                                config.expert_weights_norm,
                                config.expert_weights_scale,
                                b.expert_weights_full, 1.0F, b.mlp_out,
                                hidden_size, b.expert_job_list, b.stream);
      LaunchFlintFusedMoESwiGLU(layer.ffn_gate_exps.data,
                                layer.ffn_up_exps.data,
                                layer.ffn_gate_shexp.data,
                                layer.ffn_up_shexp.data,
                                layer.ffn_gate_exps.type,
                                layer.ffn_up_exps.type,
                                layer.ffn_gate_shexp.type,
                                layer.ffn_up_shexp.type, b.normed, b.act_moe,
                                b.expert_weights_full, b.expert_job_list,
                                used + 1, config.expert_count,
                                expert_ff, shared_ff, hidden_size,
                                gate_slab_bytes, up_slab_bytes, b.stream);
      LaunchFlintFusedMoEDown(layer.ffn_down_exps.data,
                              layer.ffn_down_shexp.data,
                              layer.ffn_down_exps.type,
                              layer.ffn_down_shexp.type, b.act_moe, b.mlp_out,
                              b.expert_weights_full, b.expert_job_list,
                              used + 1, config.expert_count,
                              expert_ff, shared_ff, hidden_size,
                              down_slab_bytes, b.stream);
    } else {
      // Legacy readback path (f32 fixtures, mixed types): the selection is
      // read back so the host can offset expert slabs for the per-expert
      // GEMV launches.
      LaunchFlintRouterTopK(b.router_logits, config.expert_count, used,
                            config.expert_weights_norm,
                            config.expert_weights_scale, b.selected,
                            b.expert_weights, b.stream);
      std::vector<std::uint32_t> h_selected(kMaxSelectedExperts);
      std::vector<float> h_weights(kMaxSelectedExperts);
      HIP_CHECK(hipMemcpyAsync(h_selected.data(), b.selected,
                               used * sizeof(std::uint32_t),
                               hipMemcpyDeviceToHost, b.stream));
      HIP_CHECK(hipMemcpyAsync(h_weights.data(), b.expert_weights,
                               used * sizeof(float), hipMemcpyDeviceToHost,
                               b.stream));
      HIP_CHECK(hipStreamSynchronize(b.stream));

      LaunchFlintZero(b.mlp_out, hidden_size, b.stream);
      for (std::uint32_t i = 0; i < used; ++i) {        const std::size_t expert = h_selected[i];
        RunSwiGLU(SlabRef(layer.ffn_gate_exps, expert, expert_slab),
                  SlabRef(layer.ffn_up_exps, expert, expert_slab), b.normed,
                  b.act, b.act_gate, expert_ff, hidden_size, b.stream);
        LaunchGEMV(ExpertSlabData(layer.ffn_down_exps, expert, expert_slab),
                   layer.ffn_down_exps.type, b.act, b.expert_out, hidden_size,
                   expert_ff, b.stream);
        LaunchFlintExpertAccumulate(b.mlp_out, b.expert_out, h_weights[i],
                                    hidden_size, b.stream);
      }

      // Shared expert: ungated, unit weight (F-E3).
      RunSwiGLU(layer.ffn_gate_shexp, layer.ffn_up_shexp, b.normed, b.act,
                b.act_gate, shared_ff, hidden_size, b.stream);
      LaunchGEMV(layer.ffn_down_shexp.data, layer.ffn_down_shexp.type, b.act,
                 b.expert_out, hidden_size, shared_ff, b.stream);
      LaunchFlintExpertAccumulate(b.mlp_out, b.expert_out, 1.0F, hidden_size,
                                  b.stream);
    }
  }
  mark(6);
  LaunchResidualAdd(b.hidden, b.mlp_out, b.hidden, hidden_size, b.stream);
  mark(7);
}

/// Resolve the stream-ordered event marks into the stage sums. Boundaries
/// per layer: 0 norm+qkv start, 1 head-norm start, 2 attention start,
/// 3 oproj start, 4 ffn-norm start, 5 moe start, 6 residual start, 7 end.
void ResolveDecodeTimings(FlintDecodeTimings& timings,
                          const std::vector<hipEvent_t>& marks,
                          const std::vector<char>& moe_flags,
                          hipEvent_t embed_start, hipEvent_t embed_end,
                          hipEvent_t tail_start, hipEvent_t step_end,
                          hipStream_t stream) {
  HIP_CHECK(hipEventSynchronize(step_end));
  const auto elapsed = [&](hipEvent_t a, hipEvent_t b) {
    float ms = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    return static_cast<double>(ms);
  };
  timings.embed_ms = elapsed(embed_start, embed_end);
  const std::size_t layers = moe_flags.size();
  for (std::size_t i = 0; i < layers; ++i) {
    const hipEvent_t* e = marks.data() + i * 8;
    timings.attn_norm_qkv_ms += elapsed(e[0], e[1]);
    timings.head_rope_ms += elapsed(e[1], e[2]);
    timings.attention_ms += elapsed(e[2], e[3]);
    timings.oproj_ms += elapsed(e[3], e[4]);
    timings.ffn_norm_router_ms += elapsed(e[4], e[5]);
    timings.moe_ms += elapsed(e[5], e[6]);
    timings.residual_ms += elapsed(e[6], e[7]);
    if (moe_flags[i] != 0) {
      ++timings.moe_layers;
    } else {
      ++timings.dense_layers;
    }
  }
  if (tail_start != nullptr) {
    timings.tail_ms = elapsed(tail_start, step_end);
  }
  timings.total_ms = elapsed(embed_start, step_end);
}

}  // namespace

void ExecuteFlintDecodeStep(const models::FlintModelWeights& weights,
                            std::uint32_t token_id, std::uint32_t pos,
                            bool compute_logits,
                            const FlintGpuStepBuffers& b,
                            FlintDecodeTimings* timings) {
  const auto& config = weights.config;
  const std::size_t hidden_size = config.hidden_size;

  // Profiling scaffolding: 8 marks per layer + embed/tail boundaries.
  // All events are stream-ordered; stage sums resolve after one final
  // sync, so the step's enqueue shape is untouched.
  std::vector<hipEvent_t> marks;
  hipEvent_t embed_start = nullptr, embed_end = nullptr;
  hipEvent_t tail_start = nullptr, step_end = nullptr;
  if (timings != nullptr) {
    marks.resize(config.num_layers * 8);
    for (auto& e : marks) {
      HIP_CHECK(hipEventCreate(&e));
    }
    HIP_CHECK(hipEventCreate(&embed_start));
    HIP_CHECK(hipEventCreate(&embed_end));
    HIP_CHECK(hipEventCreate(&tail_start));
    HIP_CHECK(hipEventCreate(&step_end));
    HIP_CHECK(hipEventRecord(embed_start, b.stream));
  }

  LaunchEmbeddingLookup(weights.token_embd.data, weights.token_embd.type,
                        token_id, b.hidden, hidden_size, b.stream);
  if (timings != nullptr) {
    HIP_CHECK(hipEventRecord(embed_end, b.stream));
  }

  std::vector<char> moe_flags(config.num_layers, 0);
  for (std::uint32_t layer_idx = 0; layer_idx < config.num_layers;
       ++layer_idx) {
    RunFlintLayer(weights, weights.layers[layer_idx], layer_idx, pos, b,
                  timings, timings != nullptr ? &marks : nullptr,
                  timings != nullptr ? moe_flags.data() + layer_idx : nullptr);
  }

  if (!compute_logits) {
    if (timings != nullptr) {
      HIP_CHECK(hipEventRecord(step_end, b.stream));
      HIP_CHECK(hipEventSynchronize(step_end));
      ResolveDecodeTimings(*timings, marks, moe_flags, embed_start, embed_end,
                           nullptr, step_end, b.stream);
      for (auto& e : marks) {
        HIP_CHECK(hipEventDestroy(e));
      }
      HIP_CHECK(hipEventDestroy(embed_start));
      HIP_CHECK(hipEventDestroy(embed_end));
      HIP_CHECK(hipEventDestroy(tail_start));
      HIP_CHECK(hipEventDestroy(step_end));
    }
    return;
  }
  if (timings != nullptr) {
    HIP_CHECK(hipEventRecord(tail_start, b.stream));
  }
  LaunchRMSNorm(b.hidden, static_cast<const float*>(weights.output_norm.data),
                b.normed, hidden_size, kFlintRmsEps, b.stream);
  LaunchGEMV(weights.output.data, weights.output.type, b.normed, b.logits,
             config.vocab_size, hidden_size, b.stream);
  LaunchGPUArgmax(b.logits, b.d_out_token, config.vocab_size, b.stream);
  if (timings != nullptr) {
    HIP_CHECK(hipEventRecord(step_end, b.stream));
    HIP_CHECK(hipEventSynchronize(step_end));
    ResolveDecodeTimings(*timings, marks, moe_flags, embed_start, embed_end,
                         tail_start, step_end, b.stream);
    for (auto& e : marks) {
      HIP_CHECK(hipEventDestroy(e));
    }
    HIP_CHECK(hipEventDestroy(embed_start));
    HIP_CHECK(hipEventDestroy(embed_end));
    HIP_CHECK(hipEventDestroy(tail_start));
    HIP_CHECK(hipEventDestroy(step_end));
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
