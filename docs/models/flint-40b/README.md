# Flint 40B (RecursiveSI)

Attention-only sparse MoE: 49 layers, 256 experts top-6 (+1 shared), hidden
2048, 32 query heads / 4 KV heads at head_dim 128, sliding window 513 on 35
rope layers, 14 global NoPE layers. Text in/out. Served by the flint runner
(dedicated HIP decode/prefill path — component-cache-free, one arena per
request state, captured decode graphs).

## Status

Experimental, in active optimization. Serve-path correctness gates (G3/G4/
G5/G6/G8) green; Phase 3 (decode performance) complete. Outstanding before
production claim: tool calls (F-L5), reasoning streaming (F-L6), window-aware
ring cache and snapshots (F-I*), deployment packaging (F-K*/F-M*). Plan and
progress log: `Kazoku-system-prompts/infra/strix-halo/flint-optimization-plan.md`.

## Artifacts

| Quant | File | Note |
| --- | --- | --- |
| IQ4_XS | `RecursiveSI_flint-IQ4_XS.gguf` (bartowski) | **served default** — selected by the GGUF header audit (ceilings doc §7) |
| UD-Q6_K_XL | `flint-UD-Q6_K_XL.gguf` | gate-calibration artifact; mixed stack (gate/up Q6_K, down Q8_0, shared Q8_0) |

Weights are not redistributable from this repo; download separately.

## Usage

```sh
gufo serve llm -m <flint>.gguf -p 8006 -c 32768 --served-model-name t-flint-40b
```

- Chat template: `FlintChatTemplate` (flint markup; tool-call rendering is
  Phase-4 work — the template renders but F-L5 parsing is not landed).
- Sampling defaults: Flint preset pins T 1.0 / p 0.95 / k 20
  (`src/core/text_sampling_defaults.hpp`); explicit request overrides win.
- Context: up to 32768 enforced (`-c`); KV is fp32 by default
  (`GUFO_FLINT_KV_F16=1` opts into the f16 twin planes — decode-negative,
  see BENCHMARKS.md).
- Snapshots/persistence are refused at this stage (Phase 5).

## Test targets (F-K1 mapping)

The F-K1 requirement names seven test targets that do not exist under those
names. Each maps to the real ctest target(s) below. Run `ctest -R flint` in
`build/gpu-test` (GPU, with `GUFO_FLINT_GGUF` set for real-artifact checks)
and `build/cpu-test`.

| F-K1 name | Real target(s) | Coverage |
| --- | --- | --- |
| `flint_config_test` | `qwen_flint_weights_test` | Inventory, router/expert type acceptance (artifact-keyed), NoPE/window, expert weight-scale |
| `flint_moe_ops_test` | `qwen_flint_moe_wmma_ops_test` + `qwen_flint_decode_ops_test` | Grouped WMMA route incl. rows 1/7/100/512; decode MoE ops |
| `flint_attention_test` | `qwen_flint_attention_test` | CPU sink+window attention |
| `flint_layer_probe` | `flint_layer_probe` | Per-layer real-artifact invariants (see below) |
| `flint_target_test` | `qwen_flint_real_smoke_test` | GPU decode vs CPU reference on the real artifact |
| `flint_tokenizer_test` | `qwen_tokenizer_test` | Qwen tokenizer (shared with the flint tokenizer path) |
| `flint_template_test` | `qwen_chat_template_test` | Chat template |

Also registered: `flint_runner_snapshot_test`, `flint_prefix_reuse_test`,
`qwen_flint_decode_step_test`.

**`flint_layer_probe` decision: registered.** `tools/flint/flint_debug_layer.cpp`
is built as `flint_layer_probe`. With no arguments it runs a fixed token pair
through the real artifact on the CPU and asserts that every layer's norms are
finite, every MoE branch is nonzero, and the window/rope attributes match the
config. It skips (77) without `GUFO_FLINT_GGUF`. Run with arguments it stays a
per-layer printer for the oracle-divergence investigation.

Measurements and gates: [BENCHMARKS.md](BENCHMARKS.md).
