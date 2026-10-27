# Flint 40B — upstream PR readiness

Status of `feat/flint-ar` relative to a pull request against the official Gufo
repository. This is a handoff note, not a claim that the branch is merge-ready.
The technical facts it cites live in [QUALITY.md](QUALITY.md),
[BENCHMARKS.md](BENCHMARKS.md) and [EXPERIMENTS.md](EXPERIMENTS.md); this file
only records the delta between "works on the reference box" and "a maintainer
can build, test and trust it from the official tree".

## Verdict

- **Draft PR / RFC: yes.** The branch builds, the flint suite is green, and the
  serve path meets its performance and acceptance gates on the reference box.
- **Merge-ready PR: no.** The remaining gap is portability and CI, plus one
  unresolved stability question (serve decode sync).

## What is ready

- `feat/flint-ar` is clean and pushed to the fork.
- `ctest -R flint` green (9 targets): `qwen_flint_weights_test`,
  `qwen_flint_attention_test`, `flint_layer_probe`,
  `qwen_flint_decode_ops_test`, `qwen_flint_decode_step_test`,
  `qwen_flint_moe_wmma_ops_test`, `qwen_flint_real_smoke_test`,
  `flint_runner_snapshot_test`, `flint_prefix_reuse_test`. Four of them need
  `GUFO_FLINT_GGUF` and the real artifact; `flint_layer_probe` skips (77)
  without it.
- Real-artifact gates: G3 format, G4 near-tie (7/8 — finding-20 carve-out), G5
  distribution, G6 split invariance, G8 concurrency identity. See
  [BENCHMARKS.md](BENCHMARKS.md#correctness-gates).
- F-J4 serve cells (IQ4_XS, `-c 32768`, sustained, iGPU DPM=high):
  **38.7 / 20.6 / 10.9 t/s** (msg/doc/long), clearing the revised 38/20/10
  floor.
- G11 end-to-end coding acceptance: 25.2k context, 2 sequential tool
  round-trips, streaming, greedy + sampled, both answers correct.
- Reproducibility and rejected alternatives are recorded
  ([QUALITY.md](QUALITY.md#reproducibility),
  [EXPERIMENTS.md](EXPERIMENTS.md)).

## What is still needed

### Correctness / stability

- **Serve decode sync — blocking.** Unsynced `hipGraphLaunch` calls past ~42
  steps hang on ROCm 7.2.1 (upstream #13234). The `GUFO_FLINT_DECODE_SYNC` knob
  is wired in the smoke harness only; the cadence in the production serve path
  (`src/models/qwen/hip/flint_decode_step.cpp`) is unconfirmed. Confirm before
  making any reliability claim.
- **Finding-20 cjk fork.** G4 is 7/8; the carve-out is an oracle-side near-tie.
  A clean verdict needs the `logprobs` gate.
- **Full suite.** Only `ctest -R flint` has been run. A PR gate is the full
  `ctest` on `build/gpu-test` (all models) plus the `cpu-sanitizer` preset.

### CI / testability

- The real gates are env-gated on a ~22 GB GGUF (`GUFO_FLINT_GGUF`); official CI
  will not have it. Need synthetic coverage that runs without weights, or a
  documented nightly/manual lane.
- Confirm the build hooks: presets, test registration, and the artifact-keyed
  inventory (`tools/ci/kernel-resources.json`).

### Portability / hygiene

- `tools/flint/make_test_image.sh` builds from a private base image
  (`gufo/test:flint-f-l4`) — not portable. Drop it from the PR, make it
  generic, or document it as reference-only.
- The model docs link to an internal plan/progress document. A public PR must
  not depend on it: these four docs should stand alone.
- Tidy review noise: the unused `kFKvWidth`, the `kUniformKvBase` compatibility
  path, and the debug tooling under `tools/flint/`.
- The ring KV layout rounds windowed planes up to a power of two (+73 MiB fp32
  at 32k) in exchange for a mask instead of a divide. Deliberate trade, measured
  — see [EXPERIMENTS.md](EXPERIMENTS.md).

### Process

- The official repository's policy on new model families is unknown; an
  issue/RFC stating scope and the above is the low-risk first step.
- PR shape: this is the full Phase 0–6 stack. Either one PR with a
  phase-to-commit map in the description, or split into logical PRs
  (infra/loading → kernels → serve integration → tests/docs).

## Pre-PR checklist

- [ ] Confirm the serve decode-sync cadence (blocking).
- [ ] Full `ctest` green on `gpu-test` and `cpu-sanitizer`.
- [ ] A CI lane that does not need the 22 GB artifact.
- [ ] `make_test_image.sh` made portable or removed from the PR.
- [ ] Model docs decoupled from the internal plan document.
- [ ] Decide single PR vs split; write the phase-to-commit map.
- [ ] Open the upstream RFC/discussion first.

## Evidence

- [QUALITY.md](QUALITY.md) — gates, known limits, reproducibility.
- [BENCHMARKS.md](BENCHMARKS.md) — measured cells, kernel-change history.
- [EXPERIMENTS.md](EXPERIMENTS.md) — rejected alternatives, measurement
  preconditions.
