# Flint 40B quality

Serve-path correctness gates for Flint 40B on `gfx1151`, ROCm 7.2.1. Oracle is
llama.cpp b11804 on the same box (same artifact), or the CPU reference for the
G3 format check. These are execution-consistency and regression checks. They
do not establish upstream or original-checkpoint parity.

## Gate summary

| Gate | Check | Result | Evidence |
| --- | --- | --- | --- |
| G3 format smoke | Real-artifact GPU decode vs CPU reference | IQ4_XS cos=1, max dlogp ≤ 3.6e-4; UD-Q6_K_XL cos=1, dlogp ≤ 2.8e-4 | `qwen_flint_real_smoke_test` |
| G3 oracle delta | Same-artifact serve vs oracle, 4 depths (64/512/2048/6144) | 22/23 rows PASS; sole failure is the finding-20 cjk fork (rank #3, Δ 5.52, identical at every depth) | `serve_gates.py`, 2026-10-25 |
| G4 near-tie forks | 512 tokens × 8 prompts, greedy | 7/8 pass; all forks rank #2 in oracle top-20, Δ 0.0010–0.3668, synonym flips with re-convergence; lone failure is the same cjk fork | `serve_gates.py g4` / `g4-compare` |
| G5 distribution | T 1.0 / p 0.95 / k 20, word categories | PASS, TV = 0.19 (cold-word 0.155, ocean 0.050) | `serve_gates.py g5` |
| G6 split invariance | Chunk split points vs unsplit chunk | Bit-exact at 9 two-way split points and a 3-way split, both artifacts | `GUFO_FLINT_SPLITS=1` chunk mode |
| G7 cache suite | `run.py --suite cache --sampling-preset flint` | 8/8 (text, thinking-tool, legacy-tool cancel; disk-spacing; four `-disk` restart cases) | 5.5, `ae7db7f2` |
| G8 concurrency | 1–8 concurrent sessions, identical greedy output | Byte-identical 1–8 sessions (1491-char stream) | `serve_gates.py g8` |
| G9 long context | 32k / 128k / 262k prefix, prefill + decode | All coherent; same 32k prompt twice gives identical output | 5.6, `80011962` |
| F-I4 scratch isolation | Target prompt alone vs under N=8 concurrent different-domain peers | Byte-identical (314 chars) at N=8 | `serve_gates.py fi4`, 5.7, `5e69ba99` |
| Prefix reuse | Reused prefix vs cold prefill, sub-window / over-window / ring-wrap | Bit-exact (512/256, 1024/600, 2048/1600) | `flint_runner_snapshot_test`, 5.4, `c8d903ec` |
| Tool calls | 13-check live agent loop | 13/13 | 4.2, `83a9982c` |
| Reasoning stream | Think markers split across SSE chunks | No reasoning leakage; accumulated channels match | `TestFlintReasoningStreaming`, 4.3 |

## Known limits

- **Thinking-always.** Flint ignores `enable_thinking: false`. Cache and
  continuation cases use reasoning-aware token budgets and keep
  `reasoning_content` in history.
- **Non-conformant IQ4_XS.** The IQ4_XS artifact omits
  `tokenizer.ggml.eog_token_ids`. The tokenizer derives the four terminators
  (`150001`, `150003`, `150005`, `150018`) from the special-token table when the
  metadata is absent (4.2).
- **Finding-20 cjk fork.** An oracle-side near-tie instability, not a serve
  defect. The clean verdict is parked on the Phase-4 logprobs gate.
- **G6 hang workaround.** Unsynced `hipGraphLaunch` calls beyond about 2 AQL
  rings (~42 steps) hang on ROCm 7.2.1 (upstream #13234). The production serve
  decode path is the graph route; the bisect and sync-interval measurements live
  in the plan's G6 blocker note. The `GUFO_FLINT_DECODE_SYNC` knob exists in the
  smoke harness only (`tests/models/qwen/hip/flint_real_smoke_test.cpp`). Confirm
  the serve-path sync cadence in `src/models/qwen/hip/flint_decode_step.cpp`
  before relying on it.
- **Pre-existing chunk assertion.** The smoke `chunk` mode fails its
  chunk-vs-token cosine assertion at n=181 (cos 0.996) on the untouched
  baseline. It is not caused by later work and is not a gate failure.
- **f16 KV** is opt-in. Its gates hold within the documented envelope
  (cos ≥ 0.995, token sequence holds), but it is not the default.
- **Weights not redistributable.** Gates are reproducible only with the
  artifacts in hand; the IQ4_XS file is not present on the current box, so its
  sha256 is recorded from the source docs (see reproducibility below).

## Reproducibility

Per headline number. Build rev is the commit at which the number was measured.

| Item | Value |
| --- | --- |
| Hardware | AMD Strix Halo `gfx1151`, 124 GB unified memory |
| ROCm | 7.2.1 |
| Kernel | 7.0.0-34-generic |
| Gufo build rev (F-J4 headline cells) | `65fc4f08` (`feat/flint-ar`) |
| Gufo build rev (gate landings) | G7 `ae7db7f2`, G9 `80011962`, F-I4 `5e69ba99`, F-L5/F-G3 `83a9982c` (`feat/flint-ar`); docs-branch head `cd433365` |
| Oracle | llama.cpp b11804 (ROCm, `-ngl 99`, `-c 32768`) |
| IQ4_XS artifact | `RecursiveSI_flint-IQ4_XS.gguf` (bartowski), sha256 `4f1fd7e09c225a4eecd257df99d0ba5ce1657add86ec608c8bc554151660fb2b` (from source docs; file not present on the current box) |
| UD-Q6_K_XL artifact | `flint-UD-Q6_K_XL.gguf`, sha256 `086231093581c67f8bf2964f8b49198889c4f0775b39857ba63b4c37041279c2` (verified locally) |
| Firmware | Performance mode, pre-bench soak; `-c` pinned to 32768 |

Server command line:

```sh
gufo serve llm -m RecursiveSI_flint-IQ4_XS.gguf -p 8006 -c 32768 \
  --served-model-name t-flint-40b
```

Gate commands are listed in [BENCHMARKS.md](BENCHMARKS.md#reproduce). Headline
cells (IQ4_XS, 2026-10-25, means of 3, `build 65fc4f08`):

| Cell | Prefill t/s | Decode t/s |
| --- | ---: | ---: |
| msg (184 tok) | 695.9 | 38.8 |
| doc (6652 tok) | 1439.7 | 20.6 |
| long (23586 tok) | 1216.2 | 10.9 |

Decode floors were revised to 38 / 20 / 10 t/s on 2026-10-25 (see the plan's
§9 note). The G10 deployment floor check (6.7) and G11 end-to-end acceptance
(6.8) are not yet run.

## Scope

Not yet qualified: the G10 floor check and G11 end-to-end coding acceptance
(deployment gates 6.7/6.8), the clean logprobs verdict for the cjk fork, and
the llama-swap deployment profile (6.5). Matched llama.cpp anchors on IQ4_XS are
pending; the current anchors were measured on UD-Q6_K_XL.
