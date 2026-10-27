# Flint 40B benchmarks

AMD Strix Halo `gfx1151` (124 GB unified memory), ROCm 7.2.1. Flint 40B is the
256-expert top-6 attention-only model (49 layers, hidden 2048, 32 query heads
over 4 KV heads at head_dim 128, sliding window 513, 14 global/NoPE layers).
Status: **serve path complete through Phase 3** — graph-captured decode,
deterministic fused-MoE, online-tile attention routing, epilogue-fused
residuals. Optimization plan and progress log:
`Kazoku-system-prompts/infra/strix-halo/flint-optimization-plan.md` (the
companion requirements and ceilings docs live beside it).

All serve tables: greedy (T=0), HTTP chat completions, `-c 32768`, firmware
Performance mode, `fj4_bench.py` protocol (edge/Tctl/power sampled per run;
prefill = prompt_tokens/TTFT, decode = (completion−1)/(T_last−T_first)).
Positive gain favors Gufo.

## F-J4 serve cells — IQ4_XS (`RecursiveSI_flint-IQ4_XS.gguf`, bartowski)

2026-10-25, `feat/flint-ar` `65fc4f08`, means of 3 after a 1×long warm-up
(sustained: Tctl 87–88 °C, edge 69–72 °C, 91–105 W, sclk 2900 MHz).

| Cell | Prompt tok | Prefill t/s | Decode t/s | llama.cpp b11804 (UD-Q6_K_XL) |
| --- | ---: | ---: | ---: | ---: |
| msg | 184 | 695.9 | 38.8 | 526.4 / 48.2 (Q6_K artifact) |
| doc | 6652 | 1439.7 | 20.6 | 1444.4 / 44.5 (Q6_K artifact) |
| long | 23586 | 1216.2 | 10.9 | 1131.3 / 38.8 (Q6_K artifact) |

Marginal prefill (long vs msg): ~817.4 t/s. Doc prefill is at llama.cpp
parity on the IQ4_XS artifact (1439.7 vs 1444.4 measured on Q6_K); long
prefill clears the ≥1200 floor.

## F-J4 re-record after the ring fix — IQ4_XS

2026-10-27, `feat/flint-ar` `4e7dfc25`, means of 3 after a warm-up, iGPU
DPM locked to `high` (EC power mode `performance`). Same artifact, `-c 32768`.

| Cell | Prompt tok | Prefill t/s | Decode t/s |
| --- | ---: | ---: | ---: |
| msg | 184 | 750.4 | **38.7** |
| doc | 6652 | 1717.0 | **20.6** |
| long | 23586 | 1394.0 | **10.9** |

Clears the revised §9 floor (38/20/10) and returns to the Phase-3 record
(38.8/20.6/10.9). Two independent gaps had to close:

- **Environmental (~1.0 t/s).** The pre-ring layout measured 37.8/19.4/10.1
  with the iGPU DPM level stuck at `auto`. Locking
  `power_dpm_force_performance_level=high` (EC power mode `performance`) took
  the same binary to 38.3/20.0/10.4. The Phase-3 record was measured with
  this setting active; it is now part of the reproduce recipe.
- **Ring (~0.7 t/s).** F-I2 allocated windowed layers
  `min(max_context, 513 + 1024) = 1537` rows, so every key access in the 35
  windowed-layer attention loops paid a runtime modulo. Rounding the ring up
  to a power of two (2048) and wrapping with a mask restored the loss;
  prefill also improved (doc 1653→1717, long 1345→1394). See the kernel
  history row below.

## F-J4 serve cells — UD-Q6_K_XL (`flint-UD-Q6_K_XL.gguf`)

2026-10-25, same binary, means of 2 (apples-to-apples against the llama.cpp
anchors, which were measured on this artifact).

| Cell | Prompt tok | Prefill t/s | Decode t/s | llama.cpp b11804 |
| --- | ---: | ---: | ---: | ---: |
| msg | 184 | 533.7 | 36.7 | 526.4 / 48.2 |
| doc | 6652 | 1165.4 | 19.7 | 1444.4 / 44.5 |
| long | 23586 | 1038.8 | 10.6 | 1131.3 / 38.8 |

Marginal prefill: ~955.5 t/s. **msg prefill beats the llama.cpp anchor**
(+1.4%). Decode is nearly quant-invariant (IQ4_XS → UD-Q6_K_XL is +60%
routed-FFN bytes yet decode moves −3/−5%), while prefill scales with bytes
(−15/−23%) — the decode bottleneck is structural, not weight-bandwidth.

## Prefill chunk sweep (F-L4)

| Config | Chunk | t/s | When |
| --- | --- | ---: | --- |
| UD-Q6_K_XL, pre-flash | 512 | 921.4 | 2026-10-24 W2 sweep |
| IQ4_XS, pre-flash | 512 | 1067.7 | 2026-10-24 (lever 1) |
| IQ4_XS, flash attention (2.6) | 512 | 1430.3 | 2026-10-24 |
| IQ4_XS, flash attention (2.6) | 1024 | **1603.1** | 2026-10-24 — the committed serve config |

Flash attention made attention cost linear (143→39 ms @512, 450→85 ms
@1024); router GEMM 82.2→18.6 ms/chunk (4.4×); grouped-WMMA MoE 4.06×.

## Kernel-change history (decode)

| Change | Commit | Effect |
| --- | --- | --- |
| HIP graph capture (3.1) | `605fe52e` | serve decode +67% (30.8→51.5 t/s @msg); one control write + one `hipGraphLaunch` per step |
| Deeper MoE decode kernels (3.2) | — | step 18.9→17.0 ms; FFN ~310 GB/s |
| **Deterministic fused-MoE accumulation (3.6)** | `9215b85b` | replaced per-row `atomicAdd` (wave-order float adds = run-to-run noise that flipped fragile argmax ties) with disjoint per-job partials + fixed-order reduce; serve wander/G8 flakiness eliminated in both KV modes |
| Global-layer routing (3.7) | `b3d39a38` | doc decode at `-c 8192`: 8.2→**20.7 t/s** (2.5×); spans >4096 always take the online-tile kernel |
| Residual-epilogue fusion (3.3) | `65fc4f08` | o-proj/dense-down/MoE-reduce store `residual + A·x` directly (~94 launches/step fewer); numerically transparent, perf neutral (recorded) |
| Ring rows rounded to a power of two (F-I2/F-J4) | `4e7dfc25` | windowed ring 1537→2048 rows and `position & (rows-1)` instead of `position % rows`: decode 37.3/19.1/9.8 → **38.7/20.6/10.9**, prefill doc +3.9% / long +3.6%; +73 MiB fp32 windowed planes at 32k |
| f16 KV cache (opt-in, `GUFO_FLINT_KV_F16=1`) | `4c17326d` | machinery + gates green, but A/B **negative for decode** (doc −10%, long −15%): decode is not KV-byte-bound. Stays opt-in; valid for per-session KV-arena halving |
| Flash-decode variants | — | all five rejected (recorded); fixed-LDS ladder remains for spans ≤4096 |

## Correctness gates

All gates run against the same artifact (oracle = llama.cpp b11804 on the
box, or the CPU reference for G3-format).

- **G3-format smoke** (`qwen_flint_real_smoke_test`): IQ4_XS fp32 cos=1,
  dlogp ≤ 3.3e-4; UD-Q6_K_XL cos=1, dlogp ≤ 2.8e-4; f16-KV opt-in within its
  documented envelope (cos ≥ 0.995, token sequence holds).
- **G4 near-tie fork structure** (serve vs oracle, IQ4_XS envelope Δ≤1.0,
  rank ≤3): 7/8 — the lone failure is the documented finding-20 cjk fork
  (oracle-side instability, identical at every depth; clean verdict parked on
  the Phase-4 `logprobs` gate).
- **G5 distribution** (200-row word-category): PASS, TV=0.19.
- **G6 split invariance**: multi-split bit-exact (9 points, both artifacts;
  chunk split(512+512) and split@3way bit-exact).
- **G8 concurrency identity**: byte-identical across 1–8 concurrent sessions
  (was intermittently 2-distinct before the 3.6 atomic fix).
- **Serve determinism**: identical greedy outputs across repeated identical
  requests in both KV modes — a property of the committed config since
  `9215b85b`.

## Reproduce

```sh
# serve (gufo)
gufo serve llm -m RecursiveSI_flint-IQ4_XS.gguf -p 8006 -c 32768 \
  --served-model-name t-flint-40b
# F-J4 cells (pin -c and firmware mode)
python3 tools/flint/fj4_bench.py --runs 3 --max-tokens 256 \
  --prompts msg,doc,long --firmware-mode performance
# G4/G5/G8 gates (oracle on :8098)
python3 tools/flint/serve_gates.py g8 --engine serve --sessions 8
# G3-format smoke + G6 splits (real artifact)
GUFO_FLINT_GGUF=<gguf> ctest -R qwen_flint_real_smoke_test
GUFO_FLINT_GGUF=<gguf> GUFO_FLINT_SPLITS=1 \
  ./tests/models/qwen/qwen_flint_real_smoke_test chunk
```

## TODO cells

- llama.cpp anchor re-measure on IQ4_XS (the current anchors are its
  UD-Q6_K_XL table).
- doc/long decode vs llama.cpp: the structural gap (−56%/−73% on Q6_K) —
  next investigation; §9 decode floors (60/55/42) not met, acceptance
  question (LUT stretch vs floor revision) open.
- IOMMU-off host lever (×1.13–1.16 prefill, measured on same silicon) not
  yet taken.
