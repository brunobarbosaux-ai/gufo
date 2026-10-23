#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/flint_gpu_model.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/weight_regions.hpp"

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
      8 +  // expert weights
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
  buffers_.logits = alloc(config.vocab_size);
  buffers_.k_cache = alloc(plane);
  buffers_.v_cache = alloc(plane);
  buffers_.expert_weights = alloc(8);
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&buffers_.selected),
                      8 * sizeof(std::uint32_t)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&buffers_.d_out_token),
                      sizeof(std::uint32_t)));
  buffers_.max_context = max_context;
  HIP_CHECK(hipStreamCreate(&buffers_.stream));
}

FlintGpuArena::~FlintGpuArena() { FreeAll(); }

FlintGpuArena::FlintGpuArena(FlintGpuArena&& other) noexcept
    : config_(other.config_),
      max_context_(other.max_context_),
      scratch_bytes_(other.scratch_bytes_),
      buffers_(other.buffers_) {
  other.scratch_bytes_ = 0;
  other.buffers_ = {};
}

FlintGpuArena& FlintGpuArena::operator=(FlintGpuArena&& other) noexcept {
  if (this != &other) {
    FreeAll();
    config_ = other.config_;
    max_context_ = other.max_context_;
    scratch_bytes_ = other.scratch_bytes_;
    buffers_ = other.buffers_;
    other.scratch_bytes_ = 0;
    other.buffers_ = {};
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
  if (buffers_.stream != nullptr) {
    (void)hipStreamDestroy(buffers_.stream);
  }
  buffers_ = {};
  scratch_bytes_ = 0;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
