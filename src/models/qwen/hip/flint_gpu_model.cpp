#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/flint_gpu_model.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/weight_regions.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"

namespace gufo::hip {

FlintGpuModel::FlintGpuModel(std::shared_ptr<const core::GgufReader> reader,
                             models::FlintModelWeights weights,
                             std::vector<QwenGpuWeightRegion> weight_regions)
    : reader_(std::move(reader)),
      weights_(std::move(weights)),
      weight_regions_(std::move(weight_regions)) {}

FlintGpuModel::~FlintGpuModel() {
  detail::ReleaseWeightRegions(weight_regions_);
}

std::size_t FlintGpuModel::GetResidentBytes() const noexcept {
  std::size_t total = 0;
  for (const auto& region : weight_regions_) {
    if (region.size > std::numeric_limits<std::size_t>::max() - total) {
      return std::numeric_limits<std::size_t>::max();
    }
    total += region.size;
  }
  return total;
}

std::shared_ptr<const FlintGpuModel> FlintGpuModel::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg) {
  if (reader == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader must not be null";
    }
    return nullptr;
  }

  auto weights_opt = models::FlintModelWeights::LoadFromGguf(*reader, error_msg);
  if (!weights_opt.has_value()) {
    return nullptr;
  }

  std::vector<QwenGpuWeightRegion> weight_regions;
  if (!detail::CreateWeightRegions(*reader, weight_regions, error_msg)) {
    return nullptr;
  }

  const auto remap = [&](models::QwenTensorRef& tensor) {
    return detail::RemapTensor(tensor, weight_regions);
  };
  bool remapped = remap(weights_opt->token_embd) &&
                  remap(weights_opt->output_norm) && remap(weights_opt->output);
  for (auto& layer : weights_opt->layers) {
    remapped = remapped && remap(layer.attn_norm) && remap(layer.attn_q) &&
               remap(layer.attn_k) && remap(layer.attn_v) &&
               remap(layer.attn_output) && remap(layer.attn_q_norm) &&
               remap(layer.attn_k_norm) && remap(layer.attn_sinks) &&
               remap(layer.ffn_norm) && remap(layer.ffn_gate) &&
               remap(layer.ffn_up) && remap(layer.ffn_down) &&
               remap(layer.ffn_gate_inp) && remap(layer.ffn_gate_exps) &&
               remap(layer.ffn_up_exps) && remap(layer.ffn_down_exps) &&
               remap(layer.ffn_gate_shexp) && remap(layer.ffn_up_shexp) &&
               remap(layer.ffn_down_shexp);
  }
  if (!remapped) {
    detail::ReleaseWeightRegions(weight_regions);
    if (error_msg != nullptr) {
      *error_msg = "A flint tensor does not belong to any mapped GGUF shard";
    }
    return nullptr;
  }

  return std::make_shared<const FlintGpuModel>(
      std::move(reader), std::move(*weights_opt), std::move(weight_regions));
}

namespace {

[[nodiscard]] std::size_t FlintArenaPlaneElements(
    const core::ModelConfig& config, std::uint32_t max_context) noexcept {
  return static_cast<std::size_t>(config.num_layers) *
         config.num_key_value_heads * max_context * config.head_dim;
}

/// Tiled Q8_1 activation bytes (prefill_quant_gemm's layout: 16-token tiles of
/// 576-byte K blocks with the fp32 scales inline) PLUS the fp32 activation-sum
/// sidecar the quantizer writes behind the tiled payload. Mirrored here
/// because that header carries device code and cannot be included from a host
/// unit. (The sidecar is easy to miss: only the Q4_K/Q5_K GEMMs read it, but
/// the quantizer always writes it.)
[[nodiscard]] std::size_t FlintQ8ActBytes(std::size_t batch,
                                          std::size_t k) noexcept {
  constexpr std::size_t kTileTokens = 16;
  constexpr std::size_t kTileBytes = 576;
  constexpr std::size_t kSumTileBytes = kTileTokens * sizeof(float);
  const std::size_t tiles = (batch + kTileTokens - 1) / kTileTokens;
  return tiles * (k / 32) * (kTileBytes + kSumTileBytes);
}

}  // namespace

FlintGpuArena::FlintGpuArena(const core::ModelConfig& config,
                             std::uint32_t max_context)
    : config_(config), max_context_(max_context) {
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t attention_size =
      static_cast<std::size_t>(config.num_attention_heads) * config.head_dim;
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t max_ff = std::max<std::size_t>(
      config.intermediate_size,
      std::max<std::size_t>(config.expert_feed_forward_length,
                            config.expert_shared_feed_forward_length));
  const std::size_t plane =
      FlintArenaPlaneElements(config, max_context);

  const std::size_t scratch_floats =
      hidden_size * 5 +  // hidden, normed, attn_proj, expert_out, mlp_out
      attention_size * 2 +  // q, attn_out
      kv_size * 2 +  // k, v
      max_ff * 2 +  // act, act_gate
      config.expert_count +  // router logits
      8 +  // expert weights (legacy readback)
      config.expert_count + 1 +  // expert weights full (F-J1 fused)
      (static_cast<std::size_t>(config.expert_count) *
           config.expert_feed_forward_length +
       config.expert_shared_feed_forward_length) +  // act_moe (F-J1 fused)
      config.vocab_size +  // logits
      plane * 2;  // k/v cache
  scratch_bytes_ = scratch_floats * sizeof(float) +
                   8 * sizeof(std::uint32_t);  // selected experts

  // One device block for the float scratch; the two small integer buffers
  // and the stream are allocated separately.
  float* block = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&block),
                      scratch_floats * sizeof(float)));
  std::size_t offset = 0;
  auto alloc = [&](std::size_t floats) -> float* {
    float* out = block + offset;
    offset += floats;
    return out;
  };
  buffers_.hidden = alloc(hidden_size);
  buffers_.normed = alloc(hidden_size);
  buffers_.q = alloc(attention_size);
  buffers_.k = alloc(kv_size);
  buffers_.v = alloc(kv_size);
  buffers_.attn_out = alloc(attention_size);
  buffers_.attn_proj = alloc(hidden_size);
  buffers_.act = alloc(max_ff);
  buffers_.act_gate = alloc(max_ff);
  buffers_.expert_out = alloc(hidden_size);
  buffers_.mlp_out = alloc(hidden_size);
  buffers_.router_logits = alloc(config.expert_count);
  buffers_.expert_weights_full = alloc(config.expert_count + 1);
  buffers_.act_moe = alloc(
      static_cast<std::size_t>(config.expert_count) *
          config.expert_feed_forward_length +
      config.expert_shared_feed_forward_length);
  buffers_.logits = alloc(config.vocab_size);
  buffers_.k_cache = alloc(plane);
  buffers_.v_cache = alloc(plane);
  buffers_.expert_weights = alloc(8);
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&buffers_.selected),
                      8 * sizeof(std::uint32_t)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&buffers_.d_out_token),
                      sizeof(std::uint32_t)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&buffers_.expert_job_list),
                      gufo::hip::kFlintMoeMaxJobs * sizeof(int)));
  buffers_.max_context = max_context;
  HIP_CHECK(hipStreamCreate(&buffers_.stream));

  // --- chunked-prefill scratch (F-L4) -------------------------------------
  // One block for every f32 row buffer, one block for the integer slot
  // tables, and a tiled Q8_1 staging buffer sized for the widest projection
  // input (the 6144-wide dense FFN rows). The chunk path owns these; the
  // batch-1 decode buffers above stay untouched so a decode step can follow a
  // chunk immediately.
  const std::size_t cap = kFlintPrefillBatchCapacity;
  const std::size_t used = config.expert_used_count;
  const std::size_t slots =
      kFlintPrefillSlotCapacity(cap, used, config.expert_count);
  const std::size_t tiles = slots / 16;  // host bound on the tile table
  const std::size_t ffn_act = std::max<std::size_t>(
      config.intermediate_size, config.expert_shared_feed_forward_length);
  const std::size_t prefill_floats =
      cap * (hidden_size * 5 + attention_size * 2 + kv_size * 2 +
             ffn_act * 2 + config.expert_count + used) +
      slots * (1 + config.expert_feed_forward_length * 2 + hidden_size);
  const std::size_t prefill_ints =
      cap +  // d_tokens staging
      cap * used +  // selected
      config.expert_count * 3 + 1 +  // counts, cursors, offsets
      slots * 2 +                    // slot_tokens, token_slots
      tiles * 2 + 1;                 // tile_expert, tile_slot0, tile_count

  float* prefill_block = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&prefill_block),
                      prefill_floats * sizeof(float)));
  std::size_t prefill_offset = 0;
  auto prefill_alloc = [&](std::size_t floats) -> float* {
    float* out = prefill_block + prefill_offset;
    prefill_offset += floats;
    return out;
  };
  prefill_.batch_capacity = cap;
  prefill_.slot_capacity = slots;
  prefill_.hidden = prefill_alloc(cap * hidden_size);
  prefill_.normed = prefill_alloc(cap * hidden_size);
  prefill_.attn_proj = prefill_alloc(cap * hidden_size);
  prefill_.mlp_out = prefill_alloc(cap * hidden_size);
  prefill_.shared_out = prefill_alloc(cap * hidden_size);
  prefill_.q = prefill_alloc(cap * attention_size);
  prefill_.attn_out = prefill_alloc(cap * attention_size);
  prefill_.k = prefill_alloc(cap * kv_size);
  prefill_.v = prefill_alloc(cap * kv_size);
  prefill_.ffn_gate = prefill_alloc(cap * ffn_act);
  prefill_.ffn_up = prefill_alloc(cap * ffn_act);
  prefill_.router_logits = prefill_alloc(cap * config.expert_count);
  prefill_.sel_weights = prefill_alloc(cap * used);
  prefill_.slot_weights = prefill_alloc(slots);
  prefill_.act_slots = prefill_alloc(slots * config.expert_feed_forward_length);
  prefill_.up_slots = prefill_alloc(slots * config.expert_feed_forward_length);
  prefill_.down_slots = prefill_alloc(slots * hidden_size);

  int* prefill_int_block = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&prefill_int_block),
                      prefill_ints * sizeof(int)));
  int* int_cursor = prefill_int_block;
  prefill_.d_tokens = reinterpret_cast<std::uint32_t*>(int_cursor);
  int_cursor += cap;
  prefill_.selected = reinterpret_cast<std::uint32_t*>(int_cursor);
  int_cursor += cap * used;
  prefill_.expert_counts = int_cursor;
  int_cursor += config.expert_count;
  prefill_.expert_cursors = int_cursor;
  int_cursor += config.expert_count;
  prefill_.expert_offsets = int_cursor;
  int_cursor += config.expert_count + 1;
  prefill_.slot_tokens = int_cursor;
  int_cursor += slots;
  prefill_.token_slots = int_cursor;
  int_cursor += slots;
  prefill_.tile_expert = int_cursor;
  int_cursor += tiles;
  prefill_.tile_slot0 = int_cursor;
  int_cursor += tiles;
  prefill_.tile_count = int_cursor;

  const std::size_t q8_k = std::max<std::size_t>(
      std::max<std::size_t>(hidden_size, attention_size), ffn_act);
  const std::size_t expert_ff = config.expert_feed_forward_length;
  HIP_CHECK(hipMalloc(&prefill_.q8_act, FlintQ8ActBytes(cap, q8_k)));
  HIP_CHECK(hipMalloc(&prefill_.q8_normed, FlintQ8ActBytes(cap, hidden_size)));
  HIP_CHECK(hipMalloc(&prefill_.slot_q8, FlintQ8ActBytes(slots, hidden_size)));
  HIP_CHECK(hipMalloc(&prefill_.act_q8, FlintQ8ActBytes(slots, expert_ff)));

  scratch_bytes_ += prefill_floats * sizeof(float) +
                    prefill_ints * sizeof(int) +
                    FlintQ8ActBytes(cap, q8_k) +
                    FlintQ8ActBytes(cap, hidden_size) +
                    FlintQ8ActBytes(slots, hidden_size) +
                    FlintQ8ActBytes(slots, expert_ff);
}

FlintGpuArena::~FlintGpuArena() { FreeAll(); }

FlintGpuArena::FlintGpuArena(FlintGpuArena&& other) noexcept
    : config_(other.config_),
      max_context_(other.max_context_),
      scratch_bytes_(other.scratch_bytes_),
      buffers_(other.buffers_),
      prefill_(other.prefill_) {
  other.scratch_bytes_ = 0;
  other.buffers_ = {};
  other.prefill_ = {};
}

FlintGpuArena& FlintGpuArena::operator=(FlintGpuArena&& other) noexcept {
  if (this != &other) {
    FreeAll();
    config_ = other.config_;
    max_context_ = other.max_context_;
    scratch_bytes_ = other.scratch_bytes_;
    buffers_ = other.buffers_;
    prefill_ = other.prefill_;
    other.scratch_bytes_ = 0;
    other.buffers_ = {};
    other.prefill_ = {};
  }
  return *this;
}

void FlintGpuArena::FreeAll() noexcept {
  if (buffers_.hidden != nullptr) {
    (void)hipFree(buffers_.hidden);
  }
  if (buffers_.selected != nullptr) {
    (void)hipFree(buffers_.selected);
  }
  if (buffers_.d_out_token != nullptr) {
    (void)hipFree(buffers_.d_out_token);
  }
  if (buffers_.expert_job_list != nullptr) {
    (void)hipFree(buffers_.expert_job_list);
  }
  if (prefill_.hidden != nullptr) {
    (void)hipFree(prefill_.hidden);
  }
  if (prefill_.expert_counts != nullptr) {
    (void)hipFree(prefill_.expert_counts);
  }
  if (prefill_.q8_act != nullptr) {
    (void)hipFree(prefill_.q8_act);
  }
  if (prefill_.q8_normed != nullptr) {
    (void)hipFree(prefill_.q8_normed);
  }
  if (prefill_.slot_q8 != nullptr) {
    (void)hipFree(prefill_.slot_q8);
  }
  if (prefill_.act_q8 != nullptr) {
    (void)hipFree(prefill_.act_q8);
  }
  if (buffers_.stream != nullptr) {
    (void)hipStreamDestroy(buffers_.stream);
  }
  buffers_ = {};
  prefill_ = {};
  scratch_bytes_ = 0;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
