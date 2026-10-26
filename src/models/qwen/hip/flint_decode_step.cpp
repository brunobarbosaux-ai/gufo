// Flint GPU decode step (F-D dispatch entry): one full autoregressive step
// over the attention-only 49-layer stack. Mirrors the validated CPU
// reference (flint_forward.cpp) op for op: RMSNorm -> QKV GEMV -> per-head
// QK norm -> RoPE (NoPE layers skip) -> sink+window attention -> O-proj ->
// residual -> FFN (dense SwiGLU lead layers, sigmoid top-k MoE + ungated
// shared expert elsewhere) -> residual; final norm -> unembed -> argmax.
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/detail/flint_decode_step.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
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
                   char* is_moe_layer,
                   const std::uint32_t* d_control = nullptr) {
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
  // Graph mode reads the position from d_control ([1]) so the captured
  // nodes are position-independent.
  if (layer.attrs.rope) {
    if (d_control != nullptr) {
      LaunchRoPE(b.q, b.k, config.num_attention_heads,
                 config.num_key_value_heads, config.head_dim,
                 config.rotary_dim, d_control + 1, config.rope_theta,
                 b.stream);
    } else {
      LaunchRoPE(b.q, b.k, config.num_attention_heads,
                 config.num_key_value_heads, config.head_dim,
                 config.rotary_dim, pos, config.rope_theta, b.stream);
    }
  }

  mark(2);
  // 5. Attention with sinks + optional sliding window (F-D1/F-D2). The
  // policy gates dispatch to the sink-capable baseline kernels. Graph mode
  // routes to the fixed-grid/fixed-LDS online-tile kernel reading d_pos —
  // the baseline's seq-sized LDS score row is not capturable and caps the
  // context at 16128.
  const detail::FlintKernelPolicy policy{
      static_cast<const float*>(layer.attn_sinks.data),
      layer.attrs.windowed ? config.sliding_window : 0U,
      b.kv_layer_rows[layer_idx]};
  if (d_control != nullptr) {
    LaunchAttention(b.q, b.k, b.v, /*gate=*/nullptr, b.k_cache, b.v_cache,
                    b.k_cache_f16, b.v_cache_f16,
                    b.attn_out, layer_idx, d_control + 1, b.max_context,
                    config.num_attention_heads, config.num_key_value_heads,
                    config.head_dim, b.stream,
                    /*skip_kv_write=*/false, &policy,
                    b.kv_layer_base[layer_idx]);
  } else {
    LaunchAttention(b.q, b.k, b.v, /*gate=*/nullptr, b.k_cache, b.v_cache,
                    b.k_cache_f16, b.v_cache_f16,
                    b.attn_out, layer_idx, pos, b.max_context,
                    config.num_attention_heads, config.num_key_value_heads,
                    config.head_dim, b.stream,
                    /*split_k_scratch=*/nullptr, /*skip_kv_write=*/false,
                    &policy, b.kv_layer_base[layer_idx]);
  }

  // 6. O-projection + residual (3.3): the residual-add is the GEMV's
  // epilogue — hidden[m] += attn_out[m]·W_o[m] — one launch instead of two.
  mark(3);
  LaunchGEMVResidual(layer.attn_output.data, layer.attn_output.type,
                     b.attn_out, b.hidden, hidden_size, attention_size,
                     b.stream);

  // 7. FFN: dense SwiGLU on the lead layers, MoE elsewhere. Every branch
  // owns its residual: dense and fused-MoE fold it into their final kernel
  // epilogue (3.3), the legacy readback path keeps the explicit add.
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
    // Dense down GEMV with fused residual (3.3): hidden += act·W_down.
    LaunchGEMVResidual(layer.ffn_down.data, layer.ffn_down.type, b.act,
                       b.hidden, hidden_size, config.intermediate_size,
                       b.stream);
    mark(6);
    mark(7);
    return;
  }
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
      b.expert_job_list != nullptr && b.moe_partials != nullptr;
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
    // The reduce kernel folds the FFN residual (3.3): hidden += MoE sum.
    LaunchFlintFusedMoEDown(layer.ffn_down_exps.data,
                            layer.ffn_down_shexp.data,
                            layer.ffn_down_exps.type,
                            layer.ffn_down_shexp.type, b.act_moe,
                            b.moe_partials, b.mlp_out, b.hidden,
                            b.expert_weights_full, b.expert_job_list,
                            used + 1, config.expert_count,
                            expert_ff, shared_ff, hidden_size,
                            down_slab_bytes, b.stream);
    mark(6);
    mark(7);
    return;
  }
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
  for (std::uint32_t i = 0; i < used; ++i) {
    const std::size_t expert = h_selected[i];
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

/// Graph-mode eligibility (3.1): fp32 KV cache plane (the online-tile Ptr
/// kernel's format), the argmax output, and every MoE layer on the fused
/// sync-free dispatch (a readback sync cannot be captured). GUFO_FLINT_
/// DECODE_GRAPH=0 forces the legacy path (A/B profiling).
[[nodiscard]] bool DecodeGraphSupported(const models::FlintModelWeights& weights,
                                        const FlintGpuStepBuffers& b) {
  const char* env = std::getenv("GUFO_FLINT_DECODE_GRAPH");
  if (env != nullptr && std::string_view(env) == "0") {
    return false;
  }
  if (b.k_cache == nullptr || b.v_cache == nullptr ||
      b.d_out_token == nullptr || b.expert_weights_full == nullptr ||
      b.act_moe == nullptr || b.expert_job_list == nullptr ||
      b.moe_partials == nullptr) {
    return false;
  }
  for (const auto& layer : weights.layers) {
    if (layer.attrs.dense_ffn) {
      continue;
    }
    if (!(FusedMoESupported(layer.ffn_gate_exps.type) &&
          FusedMoESupported(layer.ffn_up_exps.type) &&
          FusedMoESupported(layer.ffn_down_exps.type) &&
          FusedMoESupported(layer.ffn_gate_shexp.type) &&
          FusedMoESupported(layer.ffn_up_shexp.type) &&
          FusedMoESupported(layer.ffn_down_shexp.type))) {
      return false;
    }
  }
  return true;
}

/// The captured step: embedding (token from d_control) -> 49 layers ->
/// [final norm -> unembed -> argmax]. Kernels are position- and
/// token-independent; per-step scalars arrive through d_control written by
/// LaunchFlintWriteControl outside the graph. Capture runs the same
/// enqueue sequence as the legacy path with device-arg variants.
void CaptureFlintDecodeGraph(const models::FlintModelWeights& weights,
                             const FlintGpuStepBuffers& b, bool compute_logits,
                             FlintDecodeGraphState* graph) {
  const auto& config = weights.config;
  const std::size_t hidden_size = config.hidden_size;
  const auto capture_fn = [&]() {
    LaunchEmbeddingLookup(weights.token_embd.data, weights.token_embd.type,
                          graph->d_control, b.hidden, hidden_size, b.stream);
    for (std::uint32_t layer_idx = 0; layer_idx < config.num_layers;
         ++layer_idx) {
      RunFlintLayer(weights, weights.layers[layer_idx], layer_idx,
                    /*pos=*/0, b, nullptr, nullptr, nullptr,
                    graph->d_control);
    }
    if (compute_logits) {
      LaunchRMSNorm(b.hidden,
                    static_cast<const float*>(weights.output_norm.data),
                    b.normed, hidden_size, kFlintRmsEps, b.stream);
      LaunchGEMV(weights.output.data, weights.output.type, b.normed, b.logits,
                 config.vocab_size, hidden_size, b.stream);
      LaunchGPUArgmax(b.logits, b.d_out_token, config.vocab_size, b.stream);
    }
  };
  auto& executor = compute_logits ? graph->exec_logits : graph->exec_hidden;
  if (!executor.TryCapture(b.stream, detail::HipGraphCaptureKey{},
                           capture_fn)) {
    throw std::runtime_error(
        "flint decode graph capture failed (see graph dispatch telemetry)");
  }
}

}  // namespace

void FlintDecodeGraphState::Release() noexcept {
  exec_logits.Reset();
  exec_hidden.Reset();
  if (d_control != nullptr) {
    (void)hipFree(d_control);
    d_control = nullptr;
  }
  tried = false;
  supported = false;
}

void ExecuteFlintDecodeStep(const models::FlintModelWeights& weights,
                            std::uint32_t token_id, std::uint32_t pos,
                            bool compute_logits,
                            const FlintGpuStepBuffers& b,
                            FlintDecodeTimings* timings,
                            FlintDecodeGraphState* graph) {
  const auto& config = weights.config;
  const std::size_t hidden_size = config.hidden_size;

  // KV-plane debug hash (f16-KV bring-up): FNV-1a over layer 0's logical
  // rows [0, min(pos+1, 64)) in BOTH planes after the step. Print when
  // GUFO_FLINT_KV_DEBUG=1 — cross-request diffs expose stale/divergent
  // cache content directly.
  static const bool kv_debug = [] {
    const char* v = std::getenv("GUFO_FLINT_KV_DEBUG");
    return v != nullptr && v[0] == '1';
  }();
  auto kv_hash = [](const void* data, std::size_t bytes) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    std::uint32_t h = 2166136261u;
    for (std::size_t i = 0; i < bytes; ++i) {
      h = (h ^ p[i]) * 16777619u;
    }
    return h;
  };
  if (kv_debug) {
    // The instrument reads with hipMemcpy (legacy NULL stream); the arena
    // stream is a blocking stream so the copies serialize after it — but
    // only if we sync first: queued (not yet scheduled) work would otherwise
    // let the copies run BEFORE the step's kernels land (the night-3
    // "frozen zeros" readings were exactly this lag, void).
    HIP_CHECK(hipStreamSynchronize(b.stream));
    // Cross-request probe: rows [0, pos) — every row in that window was
    // written by THIS request (prefill rows [0, prompt) + decode rows
    // [prompt, pos)), so for identical requests the per-step hash sequences
    // must match bit-exactly. (The old [pos-63, pos] window mixed in the
    // not-yet-rewritten tail — stale rows from the previous request — and
    // could not be compared across runs.)
    const std::uint32_t lo = 0;
    const std::uint32_t n = pos;
    const std::size_t kv_width =
        static_cast<std::size_t>(config.num_key_value_heads) *
        config.head_dim;
    // F-I2 ring: layer 0's plane base and row count (windowed layers wrap).
    const std::size_t layer0_base = b.kv_layer_base[0];
    const std::uint32_t layer0_rows = b.kv_layer_rows[0];
    const bool layer0_windowed = layer0_rows != b.max_context;
    const auto kv_row = [&](std::uint32_t p_abs) {
      return layer0_windowed ? (p_abs % layer0_rows) : p_abs;
    };
    std::vector<float> f32_rows(n * kv_width);
    std::vector<std::uint16_t> f16_rows(n * kv_width);
    for (std::uint32_t kv_h = 0; kv_h < config.num_key_value_heads;
         ++kv_h) {
      for (std::uint32_t p = 0; p < n; ++p) {
        const std::uint32_t p_abs = lo + p;
        const std::size_t f32_off =
            layer0_base +
            (static_cast<std::size_t>(kv_h) * layer0_rows + kv_row(p_abs)) *
                config.head_dim;
        const std::size_t f16_off =
            layer0_base + static_cast<std::size_t>(kv_row(p_abs)) * kv_width +
            static_cast<std::size_t>(kv_h) * config.head_dim;
        HIP_CHECK(hipMemcpy(
            f32_rows.data() + static_cast<std::size_t>(p) * kv_width +
                static_cast<std::size_t>(kv_h) * config.head_dim,
            b.k_cache + f32_off, config.head_dim * sizeof(float),
            hipMemcpyDeviceToHost));
        if (b.k_cache_f16 != nullptr) {
          HIP_CHECK(hipMemcpy(
              f16_rows.data() + static_cast<std::size_t>(p) * kv_width +
                  static_cast<std::size_t>(kv_h) * config.head_dim,
              b.k_cache_f16 + f16_off, config.head_dim * sizeof(std::uint16_t),
              hipMemcpyDeviceToHost));
        }
      }
    }
    std::uint32_t f32_h = 0;
    for (const float v : f32_rows) {
      f32_h = kv_hash(&v, sizeof(v)) ^ (f32_h << 1);
    }
    std::uint32_t f16_h = 0;
    for (const auto v : f16_rows) {
      f16_h = kv_hash(&v, sizeof(v)) ^ (f16_h << 1);
    }
    // V plane, same rows.
    std::vector<float> v32_rows(n * kv_width);
    for (std::uint32_t kv_h = 0; kv_h < config.num_key_value_heads;
         ++kv_h) {
      for (std::uint32_t p = 0; p < n; ++p) {
        const std::uint32_t p_abs = lo + p;
        const std::size_t f32_off =
            layer0_base +
            (static_cast<std::size_t>(kv_h) * layer0_rows + kv_row(p_abs)) *
                config.head_dim;
        HIP_CHECK(hipMemcpy(
            v32_rows.data() + static_cast<std::size_t>(p) * kv_width +
                static_cast<std::size_t>(kv_h) * config.head_dim,
            b.v_cache + f32_off, config.head_dim * sizeof(float),
            hipMemcpyDeviceToHost));
      }
    }
    std::uint32_t v32_h = 0;
    for (const float v : v32_rows) {
      v32_h = kv_hash(&v, sizeof(v)) ^ (v32_h << 1);
    }
    static const void* prev_k = nullptr;
    std::uint32_t ctl[2] = {};
    if (graph != nullptr && graph->d_control != nullptr) {
      HIP_CHECK(hipMemcpy(ctl, graph->d_control, sizeof(ctl),
                          hipMemcpyDeviceToHost));
    }
    // Previous-step final hidden + logits hashes: b.hidden/b.logits still
    // hold the last step's outputs at this point (the step hasn't run).
    // Cross-run diffs quantify device-side numeric noise directly.
    std::uint32_t hid_h = 0;
    std::uint32_t lg_h = 0;
    {
      std::vector<float> buf(hidden_size);
      HIP_CHECK(hipMemcpy(buf.data(), b.hidden,
                          hidden_size * sizeof(float),
                          hipMemcpyDeviceToHost));
      for (const float v : buf) {
        hid_h = kv_hash(&v, sizeof(v)) ^ (hid_h << 1);
      }
      std::vector<float> lg(config.vocab_size);
      HIP_CHECK(hipMemcpy(lg.data(), b.logits,
                          config.vocab_size * sizeof(float),
                          hipMemcpyDeviceToHost));
      for (const float v : lg) {
        lg_h = kv_hash(&v, sizeof(v)) ^ (lg_h << 1);
      }
    }
    std::fprintf(stderr,
                 "kvdbg pos=%u ctl=[%u,%u] kc=%p k16c=%p same=%d "
                 "k32=%08x k16=%08x v32=%08x H=%08x LG=%08x\n",
                 pos, ctl[0], ctl[1], static_cast<const void*>(b.k_cache),
                 static_cast<const void*>(b.k_cache_f16),
                 prev_k == static_cast<const void*>(b.k_cache), f32_h,
                 f16_h, v32_h, hid_h, lg_h);
    prev_k = b.k_cache;
  }

  // Graph mode (3.1): eligibility is decided once; one exec per tail
  // variant is captured lazily on its first step; every step (including
  // the capture one — stream capture records without executing) then runs
  // as one control-scalar launch + one hipGraphLaunch.
  if (graph != nullptr && timings == nullptr) {
    if (!graph->tried) {
      graph->tried = true;
      graph->supported = DecodeGraphSupported(weights, b);
      if (graph->supported) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&graph->d_control),
                            2 * sizeof(std::uint32_t)));
      }
    }
    if (graph->supported) {
      auto& executor = compute_logits ? graph->exec_logits
                                      : graph->exec_hidden;
      if (!executor.IsCaptured()) {
        try {
          CaptureFlintDecodeGraph(weights, b, compute_logits, graph);
        } catch (const std::exception&) {
          graph->supported = false;  // permanent legacy fallback
        }
      }
      if (graph->supported && executor.IsCaptured()) {
        LaunchFlintWriteControl(graph->d_control, token_id, pos, b.stream);
        if (!executor.Launch(b.stream, detail::HipGraphCaptureKey{})) {
          // A failed graph launch would silently reuse the previous step's
          // logits/hidden — the exact "wandering outputs" signature. Make it
          // impossible to miss while debugging.
          std::fprintf(stderr,
                       "flint: decode graph launch FAILED (pos=%u) — step "
                       "not computed\n",
                       pos);
        }
        return;
      }
    }
  }

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
