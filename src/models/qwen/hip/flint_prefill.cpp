// Flint chunked prefill (F-L4): the whole layer stack over a chunk of tokens.
//
// The v1 prefill ran ExecuteFlintDecodeStep once per prompt token, which is
// batch-1 GEMV everywhere: every token re-read the ~850 MB of expert weights
// it selects and paid ~590 kernel launches, giving 35 t/s on a 640-token
// prompt and 16 t/s at 4.9K. This path keeps llama.cpp's reuse structure:
// every projection runs as one quantised WMMA GEMM over the chunk, the router
// runs as one batched GEMM plus one top-k block per row, and the MoE runs as
// two expert-grouped kernels whose blocks decode a weight tile once and feed
// it to a whole expert bucket (see kernels/flint_moe_batch.hip).
//
// Attention reuses the batch-decode pipeline with the flint policy: batched
// per-head QK norm, batched RoPE (NoPE layers skip it, F-D3), the sink and
// sliding-window policy (F-D1/F-D2), and the per-chunk KV write.
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/detail/flint_prefill.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
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

const bool debug_slots = std::getenv("GUFO_FLINT_PREFILL_DEBUG") != nullptr;

// A/B override: force the scalar grouped kernels even when the WMMA route is
// available (W1 measurements compare the two on the same binary).
const bool debug_force_scalar =
    std::getenv("GUFO_FLINT_MOE_SCALAR") != nullptr;

/// Types the batched expert kernels decode (Q8_0 plus the sub-16 formats).
[[nodiscard]] bool BatchExpertSupported(core::GgmlType t) noexcept {
  return t == core::GgmlType::kQ8_0 || t == core::GgmlType::kQ4_K ||
         t == core::GgmlType::kQ5_K || t == core::GgmlType::kQ6_K ||
         t == core::GgmlType::kQ3_K || t == core::GgmlType::kIQ4_NL ||
         t == core::GgmlType::kIQ4_XS || t == core::GgmlType::kIQ3_S;
}

/// Types the projected-tensor WMMA route accepts: Q8_0 or a block format with
/// a QuantizedBlockElements entry (anything else silently no-ops).
[[nodiscard]] bool BatchGemmSupported(core::GgmlType t) noexcept {
  return t == core::GgmlType::kQ8_0 || quant::QuantizedBlockElements(t) != 0;
}

[[nodiscard]] bool BatchRouterSupported(core::GgmlType t) noexcept {
  return t == core::GgmlType::kF32 || t == core::GgmlType::kF16 ||
         t == core::GgmlType::kBF16;
}

void Project(const models::QwenTensorRef& w, const float* x, std::size_t batch,
             std::size_t m, std::size_t k, void* q8_scratch, float* out,
             hipStream_t stream) {
  LaunchQuantizeActivationQ8_1FromFp32(x, q8_scratch, batch, k, stream);
  LaunchBatchedQuantGEMMPreQuantized(w.type, w.data, q8_scratch, out, batch, m,
                                     k, stream);
}

/// Projection from an activation that is already in the tiled Q8_1 layout.
/// Five of the layer's projections read `normed` (q, k, v and the shared
/// expert's gate/up, all k = hidden); quantizing it once per layer into a
/// dedicated buffer instead of once per Project call is F-L4 tuning item 2.4.
void ProjectFromQ8(const models::QwenTensorRef& w, const void* q8_x,
                   std::size_t batch, std::size_t m, std::size_t k, float* out,
                   hipStream_t stream) {
  LaunchBatchedQuantGEMMPreQuantized(w.type, w.data, q8_x, out, batch, m, k,
                                     stream);
}

}  // namespace

std::size_t FlintPrefillPositionCap(std::uint32_t max_context) {
  static const bool attn_scalar =
      std::getenv("GUFO_FLINT_ATTN_SCALAR") != nullptr;
  return attn_scalar ? kFlintPrefillMaxPosition
                     : static_cast<std::size_t>(max_context);
}

bool FlintPrefillSupported(const models::FlintModelWeights& weights) noexcept {
  const auto& config = weights.config;
  if (config.expert_used_count == 0 || config.expert_count == 0) {
    return false;
  }
  if (!BatchGemmSupported(weights.token_embd.type) ||
      !BatchGemmSupported(weights.output.type)) {
    return false;
  }
  for (const auto& layer : weights.layers) {
    if (!BatchGemmSupported(layer.attn_q.type) ||
        !BatchGemmSupported(layer.attn_k.type) ||
        !BatchGemmSupported(layer.attn_v.type) ||
        !BatchGemmSupported(layer.attn_output.type)) {
      return false;
    }
    if (layer.attrs.dense_ffn) {
      if (!BatchGemmSupported(layer.ffn_gate.type) ||
          !BatchGemmSupported(layer.ffn_up.type) ||
          !BatchGemmSupported(layer.ffn_down.type)) {
        return false;
      }
      continue;
    }
    if (!BatchRouterSupported(layer.ffn_gate_inp.type)) {
      return false;
    }
    // gate/up/down may each carry their own type (bartowski IQ4_XS: blk.2 has
    // gate IQ4_XS + up Q5_K, blk.48 is Q5_K/Q5_K/Q6_K). Layers where
    // gate == up in a no-offset WMMA format take the grouped WMMA route; the
    // rest take the scalar grouped pair, which decodes each stack with its
    // own type.
    if (!BatchExpertSupported(layer.ffn_gate_exps.type) ||
        !BatchExpertSupported(layer.ffn_up_exps.type) ||
        !BatchExpertSupported(layer.ffn_down_exps.type)) {
      return false;
    }
    if (!BatchGemmSupported(layer.ffn_gate_shexp.type) ||
        !BatchGemmSupported(layer.ffn_up_shexp.type) ||
        !BatchGemmSupported(layer.ffn_down_shexp.type)) {
      return false;
    }
  }
  return true;
}

void ExecuteFlintPrefillChunk(const models::FlintModelWeights& weights,
                              std::span<const std::uint32_t> tokens,
                              std::uint32_t start_pos, bool compute_logits,
                              const FlintGpuStepBuffers& buffers,
                              const FlintPrefillBuffers& prefill,
                              FlintPrefillTimings* timings) {
  const auto& config = weights.config;
  const std::size_t n = tokens.size();
  if (std::getenv("GUFO_FLINT_KV_DEBUG") != nullptr) {
    std::fprintf(stderr, "prefill dbg start=%u n=%zu kc=%p k16c=%p\n",
                 start_pos, n, static_cast<const void*>(buffers.k_cache),
                 static_cast<const void*>(buffers.k_cache_f16));
  }
  if (n == 0) {
    return;
  }
  if (n > prefill.batch_capacity ||
      n * config.expert_used_count > prefill.slot_capacity) {
    throw std::invalid_argument("flint prefill chunk exceeds its scratch");
  }
  if (!FlintPrefillSupported(weights)) {
    throw std::invalid_argument(
        "flint prefill chunk requires the quantised batched routes");
  }

  hipStream_t stream = buffers.stream;
  const std::size_t hidden = config.hidden_size;
  const std::size_t attention_size =
      static_cast<std::size_t>(config.num_attention_heads) * config.head_dim;
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::uint32_t used = config.expert_used_count;
  const std::size_t slots = n * used;

  void* q8 = prefill.q8_act;

  // Stage-grouped event marks (tuning item 0.3). Each interval between two
  // marks is attributed to the bucket recorded at its closing mark, so the
  // per-layer residual add folds into the next layer's norm interval.
  enum Bucket : int {
    kNone = -1,
    kTail = 0,  // embedding, final norm, unembed
    kNorm,      // RMSNorm + the shared Q8_1 quantize
    kQkv,
    kAttention,
    kOproj,
    kRouter,  // FFN norm + router + top-k + compaction
    kShared,
    kRouted,
    kDense,
    kBucketCount
  };
  std::vector<hipEvent_t> marks;
  std::vector<int> mark_buckets;
  const auto mark = [&](int bucket) {
    if (timings == nullptr) {
      return;
    }
    hipEvent_t e = nullptr;
    HIP_CHECK(hipEventCreate(&e));
    HIP_CHECK(hipEventRecord(e, stream));
    marks.push_back(e);
    mark_buckets.push_back(bucket);
  };
  const auto collect = [&]() {
    if (timings == nullptr || marks.size() < 2) {
      return;
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    double* fields[kBucketCount] = {
        &timings->tail_ms,  &timings->norm_ms, &timings->qkv_ms,
        &timings->attention_ms, &timings->oproj_ms, &timings->router_ms,
        &timings->shared_ms,    &timings->routed_ms, &timings->dense_ms};
    for (std::size_t i = 1; i < marks.size(); ++i) {
      if (mark_buckets[i] < 0) {
        continue;
      }
      float ms = 0.0F;
      HIP_CHECK(hipEventElapsedTime(&ms, marks[i - 1], marks[i]));
      *fields[static_cast<std::size_t>(mark_buckets[i])] += ms;
    }
    for (auto& e : marks) {
      (void)hipEventDestroy(e);
    }
    marks.clear();
    mark_buckets.clear();
  };
  mark(kNone);

  // Token ids reach the embedding kernel through device staging: the caller's
  // span is host memory, and a kernel dereferencing it faults the context.
  HIP_CHECK(hipMemcpyAsync(prefill.d_tokens, tokens.data(),
                           n * sizeof(std::uint32_t), hipMemcpyHostToDevice,
                           stream));
  if (debug_slots) {
    // Verify the staging copy landed before the kernel consumes it.
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<std::uint32_t> back(n);
    HIP_CHECK(hipMemcpy(back.data(), prefill.d_tokens, n * sizeof(int),
                        hipMemcpyDeviceToHost));
    std::cout << "[prefill-dbg] chunk n=" << n << " start_pos=" << start_pos
              << " d_tokens=" << static_cast<const void*>(prefill.d_tokens)
              << " hidden=" << static_cast<const void*>(prefill.hidden)
              << " table=" << weights.token_embd.data << " tokens:";
    for (auto v : back) {
      std::cout << " " << v;
    }
    std::cout << std::endl;
  }
  LaunchBatchedEmbeddingLookup(weights.token_embd.data, weights.token_embd.type,
                               prefill.d_tokens, prefill.hidden, n, hidden,
                               stream);
  mark(kTail);

  for (std::uint32_t layer_idx = 0; layer_idx < config.num_layers;
       ++layer_idx) {
    const auto& layer = weights.layers[layer_idx];

    // --- attention ---------------------------------------------------------
    LaunchBatchedRMSNorm(prefill.hidden,
                         static_cast<const float*>(layer.attn_norm.data),
                         prefill.normed, nullptr, n, hidden, kFlintRmsEps,
                         stream);
    // normed (attention norm) feeds q/k/v — quantize it once for the three
    // readers (2.4). The FFN-side readers re-quantize after the FFN norm.
    LaunchQuantizeActivationQ8_1FromFp32(prefill.normed, prefill.q8_normed, n,
                                         hidden, stream);
    mark(kNorm);
    ProjectFromQ8(layer.attn_q, prefill.q8_normed, n, attention_size, hidden,
                  prefill.q, stream);
    ProjectFromQ8(layer.attn_k, prefill.q8_normed, n, kv_size, hidden,
                  prefill.k, stream);
    ProjectFromQ8(layer.attn_v, prefill.q8_normed, n, kv_size, hidden,
                  prefill.v, stream);
    mark(kQkv);

    LaunchBatchedPerHeadRMSNorm(
        prefill.q, static_cast<const float*>(layer.attn_q_norm.data),
        prefill.q, n, config.num_attention_heads, config.head_dim, kFlintRmsEps,
        stream);
    LaunchBatchedPerHeadRMSNorm(
        prefill.k, static_cast<const float*>(layer.attn_k_norm.data),
        prefill.k, n, config.num_key_value_heads, config.head_dim,
        kFlintRmsEps, stream);

    if (layer.attrs.rope) {
      LaunchBatchedRoPE(prefill.q, prefill.k, n, config.num_attention_heads,
                        config.num_key_value_heads, config.head_dim,
                        config.rotary_dim, start_pos, config.rope_theta,
                        stream, nullptr);
    }

    const detail::FlintKernelPolicy policy{
        static_cast<const float*>(layer.attn_sinks.data),
        layer.attrs.windowed ? config.sliding_window : 0U,
        buffers.kv_layer_rows[layer_idx]};
    static const bool attn_scalar = std::getenv("GUFO_FLINT_ATTN_SCALAR") !=
                                    nullptr;
    // Layer-range A/B for divergence bisecting: flash only for layers in
    // [GUFO_FLINT_FLASH_FROM, GUFO_FLINT_FLASH_TO) (defaults: all).
    static const int flash_from = [] {
      const char* v = std::getenv("GUFO_FLINT_FLASH_FROM");
      return v != nullptr ? std::atoi(v) : 0;
    }();
    static const int flash_to = [] {
      const char* v = std::getenv("GUFO_FLINT_FLASH_TO");
      return v != nullptr ? std::atoi(v) : 1000;
    }();
    bool wmma_attention = false;
    if (!attn_scalar && layer_idx >= flash_from && layer_idx < flash_to) {
      // 2.6 flash route: the WMMA kernel reads the cache, so stage this
      // chunk's K/V first (the batched kernel would otherwise do it).
      LaunchBatchedKvWrite(prefill.k, prefill.v, buffers.k_cache,
                           buffers.v_cache, buffers.k_cache_f16,
                           buffers.v_cache_f16, layer_idx, start_pos, n,
                           buffers.max_context, config.num_key_value_heads,
                           config.head_dim, stream,
                           buffers.kv_layer_base[layer_idx],
                           buffers.kv_layer_rows[layer_idx]);
      wmma_attention = LaunchFlintWmmaAttention(
          prefill.q, buffers.k_cache, buffers.v_cache, prefill.attn_out,
          layer_idx, start_pos, n, buffers.max_context, stream, &policy,
          buffers.kv_layer_base[layer_idx]);
    }
    if (!wmma_attention) {
      LaunchBatchedAttention(
          prefill.q, prefill.k, prefill.v, /*gate=*/nullptr, buffers.k_cache,
          buffers.v_cache, buffers.k_cache_f16, buffers.v_cache_f16,
          prefill.attn_out, layer_idx, start_pos, n, buffers.max_context,
          config.num_attention_heads, config.num_key_value_heads,
          config.head_dim, stream, /*skip_kv_write=*/false,
          /*out_context_bf16=*/nullptr, &policy,
          buffers.kv_layer_base[layer_idx]);
    }
    mark(kAttention);

    Project(layer.attn_output, prefill.attn_out, n, hidden, attention_size, q8,
            prefill.attn_proj, stream);
    LaunchBatchedResidualAdd(prefill.hidden, prefill.attn_proj, prefill.hidden,
                             n, hidden, stream);
    mark(kOproj);

    // --- feed forward ------------------------------------------------------
    LaunchBatchedRMSNorm(prefill.hidden,
                         static_cast<const float*>(layer.ffn_norm.data),
                         prefill.normed, nullptr, n, hidden, kFlintRmsEps,
                         stream);

    if (layer.attrs.dense_ffn) {
      const std::size_t intermediate = config.intermediate_size;
      Project(layer.ffn_gate, prefill.normed, n, intermediate, hidden, q8,
              prefill.ffn_gate, stream);
      Project(layer.ffn_up, prefill.normed, n, intermediate, hidden, q8,
              prefill.ffn_up, stream);
      LaunchBatchedSwiGLUActivation(prefill.ffn_gate, prefill.ffn_up,
                                    prefill.ffn_gate, nullptr, n * intermediate,
                                    stream);
      Project(layer.ffn_down, prefill.ffn_gate, n, hidden, intermediate, q8,
              prefill.mlp_out, stream);
      mark(kDense);
    } else {
      const std::size_t expert_ff = config.expert_feed_forward_length;
      const std::size_t shared_ff = config.expert_shared_feed_forward_length;
      const std::size_t expert_slab = expert_ff * hidden;

      // The FFN norm overwrote normed, so the FFN-side q8_1 readers (shared
      // expert gate/up and the WMMA MoE gather) need codes from THIS norm —
      // re-quantize here. The QKV readers consumed the attention-norm codes
      // earlier in the layer, which is why one shared buffer is not enough.
      LaunchQuantizeActivationQ8_1FromFp32(prefill.normed, prefill.q8_normed, n,
                                           hidden, stream);
      // Router and device-side slot compaction.
      LaunchFlintRouterBatch(prefill.normed, n, layer.ffn_gate_inp.data,
                             layer.ffn_gate_inp.type, config.expert_count,
                             hidden, prefill.router_logits, stream);
      LaunchFlintRouterTopKBatch(
          prefill.router_logits, n, config.expert_count, used,
          config.expert_weights_norm, config.expert_weights_scale,
          prefill.selected, prefill.sel_weights, stream);
      LaunchFlintMoeSlots(prefill.selected, prefill.sel_weights, n, used,
                          config.expert_count, prefill.expert_counts,
                          prefill.expert_offsets, prefill.expert_cursors,
                          prefill.slot_tokens, prefill.slot_weights,
                          prefill.token_slots, prefill.tile_expert,
                          prefill.tile_slot0, prefill.tile_count, stream);
      mark(kRouter);

      // GUFO_FLINT_PREFILL_DEBUG=1 dumps the first MoE layer's routing state
      // (first chunk only) to isolate slot-table corruption.
      if (debug_slots && layer_idx == config.leading_dense_block_count &&
          start_pos == 0) {
        HIP_CHECK(hipStreamSynchronize(stream));
        const std::size_t dump = std::min<std::size_t>(slots, 24);
        std::vector<std::uint32_t> sel(dump);
        std::vector<int> tsl(dump);
        std::vector<int> cnt(config.expert_count);
        std::vector<int> offs(config.expert_count + 1);
        HIP_CHECK(hipMemcpy(sel.data(), prefill.selected, dump * sizeof(int),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(tsl.data(), prefill.token_slots,
                            dump * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(cnt.data(), prefill.expert_counts,
                            cnt.size() * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(offs.data(), prefill.expert_offsets,
                            offs.size() * sizeof(int), hipMemcpyDeviceToHost));
        std::cout << "[prefill-dbg] layer " << layer_idx << " slots=" << slots
                  << std::endl;
        std::cout << "  selected:";
        for (auto v : sel) {
          std::cout << " " << v;
        }
        std::cout << std::endl;
        std::cout << "  token_slots:";
        for (auto v : tsl) {
          std::cout << " " << v;
        }
        std::cout << std::endl;
        std::cout << "  counts[0..16):";
        for (int i = 0; i < 16; ++i) {
          std::cout << " " << cnt[i];
        }
        std::cout << "  total="
                  << std::accumulate(cnt.begin(), cnt.end(), 0L) << std::endl;
        std::cout << "  offsets[0..16):";
        for (int i = 0; i < 16; ++i) {
          std::cout << " " << offs[i];
        }
        std::cout << "  last=" << offs.back() << std::endl;
        std::vector<float> logits_row(config.expert_count);
        HIP_CHECK(hipMemcpy(logits_row.data(), prefill.router_logits,
                            logits_row.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        int nan_count = 0;
        float first = 0.0F;
        for (std::size_t i = 0; i < logits_row.size(); ++i) {
          if (!std::isfinite(logits_row[i])) {
            ++nan_count;
            if (nan_count == 1) {
              first = logits_row[i];
            }
          }
        }
        std::cout << "  router row0: nonfinite=" << nan_count
                  << " first=" << first
                  << " l[0..8):";
        for (int i = 0; i < 8; ++i) {
          std::cout << " " << logits_row[i];
        }
        std::cout << std::endl;
        std::vector<float> normed_row(hidden);
        HIP_CHECK(hipMemcpy(normed_row.data(), prefill.normed,
                            normed_row.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        int normed_bad = 0;
        float normed_first_bad = 0.0F;
        float normed_absmax = 0.0F;
        for (std::size_t i = 0; i < normed_row.size(); ++i) {
          if (!std::isfinite(normed_row[i])) {
            ++normed_bad;
            if (normed_bad == 1) {
              normed_first_bad = normed_row[i];
            }
          }
          normed_absmax = std::max(normed_absmax, std::abs(normed_row[i]));
        }
        std::cout << "  normed row0: nonfinite=" << normed_bad
                  << " first_bad=" << normed_first_bad
                  << " absmax=" << normed_absmax << std::endl;
      }

      // Shared expert: three full-chunk GEMMs.
      ProjectFromQ8(layer.ffn_gate_shexp, prefill.q8_normed, n, shared_ff,
                    hidden, prefill.ffn_gate, stream);
      ProjectFromQ8(layer.ffn_up_shexp, prefill.q8_normed, n, shared_ff, hidden,
                    prefill.ffn_up, stream);
      LaunchBatchedSwiGLUActivation(prefill.ffn_gate, prefill.ffn_up,
                                    prefill.ffn_gate, nullptr, n * shared_ff,
                                    stream);
      Project(layer.ffn_down_shexp, prefill.ffn_gate, n, hidden, shared_ff, q8,
              prefill.shared_out, stream);
      mark(kShared);

      // Routed experts. The grouped WMMA route runs the buckets on the same
      // int8 tensor cores as the projections; the scalar grouped pair stays
      // for formats the WMMA kernels do not cover.
      const std::size_t slab_gate =
          quant::EncodedSizeBytes(layer.ffn_gate_exps.type, expert_slab);
      const std::size_t slab_up =
          quant::EncodedSizeBytes(layer.ffn_up_exps.type, expert_slab);
      const std::size_t slab_down =
          quant::EncodedSizeBytes(layer.ffn_down_exps.type, expert_slab);
      const bool wmma_route =
          FlintGroupedWmmaSupported(layer.ffn_gate_exps.type,
                                    layer.ffn_up_exps.type,
                                    layer.ffn_down_exps.type, expert_ff,
                                    hidden) &&
          !debug_force_scalar;
      if (wmma_route) {
        const std::size_t tile_bound = prefill.slot_capacity / 16;
        LaunchFlintGatherQ8Act(prefill.q8_normed, prefill.slot_q8,
                               prefill.slot_tokens, prefill.expert_offsets,
                               config.expert_count, prefill.slot_capacity,
                               hidden, stream);
        LaunchFlintMoeGroupedGateUpWmma(
            layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
            layer.ffn_gate_exps.type, prefill.slot_q8, prefill.act_slots,
            prefill.up_slots, prefill.expert_offsets, prefill.tile_expert,
            prefill.tile_slot0, prefill.tile_count, tile_bound,
            config.expert_count, expert_ff, hidden, slab_gate, stream);
        LaunchBatchedSwiGLUActivation(prefill.act_slots, prefill.up_slots,
                                      prefill.act_slots, nullptr,
                                      prefill.slot_capacity * expert_ff,
                                      stream);
        LaunchQuantizeActivationQ8_1FromFp32(prefill.act_slots,
                                             prefill.act_q8,
                                             prefill.slot_capacity, expert_ff,
                                             stream);
        LaunchFlintMoeGroupedDownWmma(
            layer.ffn_down_exps.data, layer.ffn_down_exps.type,
            prefill.act_q8, prefill.down_slots,
            prefill.expert_offsets, prefill.slot_weights, prefill.tile_expert,
            prefill.tile_slot0, prefill.tile_count, tile_bound,
            config.expert_count, expert_ff, hidden, slab_down, stream);
      } else {
        LaunchFlintMoeBatchSwiGLU(
            layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
            layer.ffn_gate_exps.type, layer.ffn_up_exps.type, prefill.normed,
            prefill.act_slots,
            prefill.expert_offsets, prefill.slot_tokens, config.expert_count,
            expert_ff, hidden, slab_gate, slab_up, stream);
        LaunchFlintMoeBatchDown(
            layer.ffn_down_exps.data, layer.ffn_down_exps.type,
            prefill.act_slots, prefill.down_slots, prefill.expert_offsets,
            prefill.slot_weights, config.expert_count, expert_ff, hidden,
            slab_down, stream);
      }

      LaunchFlintMoeCombine(prefill.shared_out, prefill.down_slots,
                            prefill.token_slots, prefill.mlp_out, n, used,
                            hidden, stream);
      mark(kRouted);
    }

    LaunchBatchedResidualAdd(prefill.hidden, prefill.mlp_out, prefill.hidden, n,
                             hidden, stream);
  }

  if (compute_logits) {
    LaunchBatchedRMSNorm(prefill.hidden + (n - 1) * hidden,
                         static_cast<const float*>(weights.output_norm.data),
                         buffers.normed, nullptr, 1, hidden, kFlintRmsEps,
                         stream);
    LaunchGEMV(weights.output.data, weights.output.type, buffers.normed,
               buffers.logits, config.vocab_size, hidden, stream);
    LaunchGPUArgmax(buffers.logits, buffers.d_out_token, config.vocab_size,
                    stream);
  }
  mark(kTail);
  collect();
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
