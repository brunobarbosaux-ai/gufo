# Flint 40B experiments

Rejected or negative alternatives, with the numbers that decided them. Retained
changes are in [BENCHMARKS.md](BENCHMARKS.md) (kernel-change history).

| Experiment | Decision | Numbers |
| --- | --- | --- |
| f16 KV cache (3.6, `4c17326d`) | Rejected as default; stays opt-in (`GUFO_FLINT_KV_F16=1`) | A/B decode: doc 20.7 → 18.6 (−10%), long 11.0 → 9.4 (−15%), msg −2%. Global-layer KV is <2% of a doc step (381→190 MB of ~48 ms), so halving KV bytes cannot pay. Kept for per-session arena halving (5.2/5.3 memory math). |
| Flash-decode variants (3.x) | Rejected, all five | Recorded as negative; fixed-LDS ladder remains for spans ≤4096. |
| Graph vs legacy decode (3.1, `605fe52e`) | Graph kept; legacy available as `GUFO_FLINT_DECODE_GRAPH=0` | Graph: 30.8 → 51.5 t/s at msg (+67%), step 32.5 → 19.4 ms. Legacy runs ~60% of graph throughput. Legacy is the zero-code-change fallback for the G6 hang. |
| Unsynced graph launches (G6 blocker) | Rejected; serve must sync periodically | Hang onset ~42 unsynced steps (ring-size bound); sync=0 HANG, sync=16 PASS (bit-identical logits, `max_abs=0`), sync=32 and sync=64 HANG. |
| Fused-MoE per-row `atomicAdd` (3.6, `9215b85b`) | Rejected; replaced with disjoint per-job partials and fixed-order reduce | Float atomic order caused run-to-run noise that flipped fragile argmax ties (serve wander, G8 2-distinct). After fix: wander 6/6 identical, G8 PASS ×4. Cost: +1 tiny launch, doc 20.2 → 20.7 t/s. |
| Global-layer `<8192>` fixed-LDS tier (3.7, `b3d39a38`) | Removed; spans >4096 use the online-tile kernel | Doc decode at `-c 8192`: 8.2 → 20.7 t/s (2.5×). Regime knob removed. |
| Residual-epilogue fusion (3.3, `65fc4f08`) | Retained as simplification; perf-neutral | msg 38.6 / doc 20.5 vs 39.2 / 20.7 pre-fusion (within noise). Each ~50-launch cut ≈ 1% at msg. |
| Chunk size 512 vs 1024 (2.5 → 2.6) | 1024 adopted after flash attention | Pre-flash: 181 = 801.1, 256 = 945.6, 512 = 1067.7, 1024 = 990.1 (lost to attention superlinearity). Post-flash: 512 = 1430.3, 1024 = 1603.1. |
| Pre-flash attention (2.6) | Replaced by `FlintWmmaCausalAttention` | Attention 143.4 → 38.7 ms @512; 450 → 84.6 ms @1024 (linear). Pre-flash 1024 chunk fell to 990 t/s. |
| Scalar vs WMMA MoE buckets (2.2) | WMMA always on at serve chunk sizes | WMMA 829.7 t/s vs scalar 337.8 t/s at 512 (2.46×). `GUFO_FLINT_MOE_SCALAR=1` is the A/B escape hatch. |
| Warp-per-expert scalar router (2.7) | Replaced by tiled GEMM | Router 82.2 → 18.6 ms/chunk (4.4×). Quantizing the router to q8_0 was rejected: ~0.3% logit error flips top-6 selections against decode. Router stays F16. |
| Grouped MoE: scalar vs WMMA tiles (2.1) | WMMA retained | Routed MoE 1195 → 294.5 ms per 512-chunk (4.06×). |
| UD-Q6_K_XL vs IQ4_XS artifact (ceilings §7, F-J4) | IQ4_XS served default; Q6_K_XL kept as gate-calibration artifact | Prefill scales with bytes: msg 533.7 vs 695.9, doc 1165.4 vs 1439.7, long 1038.8 vs 1216.2 t/s (Q6 vs IQ4). Decode nearly invariant (36.7 vs 38.8, 19.7 vs 20.6, 10.6 vs 10.9). Decode is structurally bound, not weight-bandwidth bound. |
| Fused vs unfused NoPE (1.8 / F-D3) | Not separately benchmarked | NoPE on the 14 global layers is driven by `rope_pattern`; correctness is covered by `qwen_flint_attention_test` and `qwen_flint_weights_test` (CPU). No speed A/B recorded. |
| Full-KV vs ring cache (5.2/5.3) | Ring for windowed layers; full-KV kept for global layers | fp32 full at 128k ≈ 48.7 GB with IQ4_XS (over the 45 GiB budget). fp32 ring at 262k ≈ 37.5 GB (fits). Ring makes 262k claimable. |
| Ring row modulus: divide vs mask (F-I2/F-J4, `4e7dfc25`) | Adopted: ring rows rounded up to a power of two, wrap with `position & (rows-1)` | The 1537-row ring put a runtime modulo in every key access of the 35 windowed-layer loops: decode 37.3/19.1/9.8 vs 38.3/20.0/10.4 pre-ring. Rounding to 2048 rows and masking restored 38.7/20.6/10.9 (Phase-3 record) and lifted prefill (doc 1653→1717, long 1345→1394 t/s). Costs +73 MiB fp32 windowed planes at 32k. |
| Tile shapes at head_dim 128 (2.3) | Retired; current 16×16×16 tiles kept | Audit not executed because prefill reached the stretch floor (doc 1450 vs 1500 target, llama.cpp parity). Revisit only if prefill regresses. |
| Decode floors 60/55/42 t/s (§9) | Revised to 38/20/10 t/s (user decision, 2026-10-25) | Measured sustained: 38.8 / 20.6 / 10.9. Byte-centric levers were measured out (f16 KV negative, decode quant-invariant). |

## Measurement preconditions

The Phase-3 record and the 2026-10-27 re-record were both taken with the iGPU
DPM level locked to `high` (EC power mode `performance`). With DPM left at
`auto`, the same pre-ring binary measures ~1.0 t/s lower on every cell
(37.8/19.4/10.1 vs 38.3/20.0/10.4 at commit `f674e8fb`) and the floor is not
reachable. Confirm both settings before quoting or comparing any F-J4 number.

## Not recorded

- **`GUFO_FLINT_DECODE_SYNC` in serve.** The plan's G6 section names this knob
  as the sync=16 production route. The knob appears only in the smoke harness
  (`tests/models/qwen/hip/flint_real_smoke_test.cpp`); the serve decode step
  in `src/models/qwen/hip/flint_decode_step.cpp` was not confirmed to sync
  every 16 steps. Verify before relying on the production route.
- **Tile-shape table (2.3).** Not measured; see above.
- **Standalone NoPE A/B.** Not measured.
