# Serve-path quality gates

HTTP-only gates for validating a served model against a pinned llama.cpp
oracle. They need no engine code: point them at an OpenAI-style chat
endpoint (or llama.cpp's native `/completion` for the oracle) and at the
same GGUF loaded in both engines. The gates were built and validated on
the Flint 40B port and are model-agnostic by construction; they are the
acceptance path for new models and, importantly, for non-blessed quants
of known models.

| Gate | What it checks |
| --- | --- |
| G4 | Greedy divergence vs the oracle: streams identical up to each fork, every fork token a near-tie in the oracle's captured top-20 |
| G5 | Sampling distribution (T1.0 / p0.95 / k20): same support, same ordering, TV within bound |
| G8 | Concurrency: N identical concurrent requests produce byte-identical outputs |

## The oracle

"The oracle" is the pinned llama.cpp build used as the reference engine:
an independent implementation of GGUF inference with a different kernel
stack, sharing only the file format with Gufo. Comparing against it is
what makes the gates evidence rather than self-checks: if a served model
matches an implementation that shares no code with it, the kernels are
probably right, and the comparison also demonstrates the "faster than
llama.cpp without breaking correctness" claim directly. The pin matters
as much as the engine: an oracle that changes under the gates makes
runs from different dates incomparable, so every evidence file records
which build produced it.

The oracle is not perfectly self-consistent, and that is deliberate
input to the design, not a flaw to hide. llama.cpp evaluates prompt
positions with a batched prefill path and generated positions with a
cached one-token-at-a-time decode path; the two do the same arithmetic
in different orders, so near-tie tokens can flip between them even
though nothing is wrong. The calibration below measures exactly that:
how much disagreement is normal for this model and quant. A later
Gufo-vs-oracle comparison is then judged against that measured noise
floor, so real signal (an early fork, a token far outside the top
ranks) separates from harmless rounding.

## The G4 near-tie envelope

Bit-exact identity between two quantized-kernel stacks is unattainable:
the oracle itself flips near-ties between its batched-prefill and
cached-decode paths, and the two engines round differently throughout. A
structural bug, by contrast, forks early and lands far outside the top
ranks. So G4 accepts a fork token when the oracle itself considered it
essentially co-optimal: rank <= `G4_MAX_RANK` in the captured top-k and
within `G4_NEAR_TIE_DELTA` logprobs of top-1.

Those constants are **per model/quant**, calibrated against the oracle's
own self-flip distribution (prefill-vs-decode on the same file, one
generation per prompt, freshly started server). The first calibration,
Q4_K_XL on a 40B MoE (2026-11-08):

- 397 near-tie positions measured
- the oracle flips 15.9% of them (63/397) against itself
- P95 self-flip delta 0.709, max 1.335, all at decode gap < 0.478
- zero structural disagreements (no flip beyond rank 4)

The envelope delta <= 0.8 / rank <= 4 covers P97 of the oracle's own
flips and the full observed rank range. The previous envelope
(delta <= 0.5, rank <= 3) was stricter than the oracle's own
self-consistency on Q4 and produced false failures.

Raw data: `docs/quality/g4-envelope-calibration.json` (fields: `sanity`
capture alignment, `flips` per-position records, `divergence` first
fork per class, `summary`).

Coarser quants need looser envelopes: early IQ-format data measured
roughly 2x the Q4 envelope. Every new model/quant pair gets its own
calibration run before its gate verdict is published; the per-pair
envelopes are recorded next to the run's evidence.

## Usage

Start both engines outside the script (on a single-iGPU machine they
cannot run at the same time; every gate is phase-split — capture one arm
to a file, then the other, then compare).

```sh
# G4: capture oracle arm on a FRESHLY STARTED server (one generation per
# prompt; repeats hit the cached-decode path and flip at near-ties)
python3 tools/gates/serve_gates.py g4 --engine oracle \
  --base-url http://127.0.0.1:8098 --model ORACLE-MODEL \
  --out /tmp/g4-oracle.json --prompts 8
python3 tools/gates/serve_gates.py g4 --engine serve \
  --base-url http://127.0.0.1:8006/v1 --model SERVE-MODEL \
  --out /tmp/g4-serve.json
python3 tools/gates/serve_gates.py g4-compare /tmp/g4-oracle.json /tmp/g4-serve.json \
  --calibration tools/gates/calibration/<model>-<quant>.json

# G4 envelope calibration from a captured oracle arm (see below)
python3 tools/gates/serve_gates.py g4-calibrate \
  --capture /tmp/g4-oracle.json --out tools/gates/calibration/<model>-<quant>.json

# G5: distribution arms, then the verdict
python3 tools/gates/serve_gates.py g5 --engine oracle \
  --base-url http://127.0.0.1:8098 --out /tmp/g5-oracle.json --rows 200
python3 tools/gates/serve_gates.py g5 --engine serve \
  --base-url http://127.0.0.1:8006/v1 --out /tmp/g5-serve.json --rows 200
python3 tools/gates/serve_gates.py g5-verdict /tmp/g5-serve.json /tmp/g5-oracle.json

# G8: concurrent identical requests, 1..N sessions
python3 tools/gates/serve_gates.py g8 --engine serve \
  --base-url http://127.0.0.1:8006/v1 --sessions 8 --gate
```

G4 prompt construction samples `tools/gates/corpus.py` (deterministic,
fixed seed, nine prompt classes — `--prompts` is capped at one prompt per
class) and trims prompts to depth with the oracle's `/tokenize`. Reasoning
handling is normalised in the harness: an engine that separates reasoning
into `reasoning_content` and one that inlines inline think blocks produce
the same comparable streams, so the gates do not depend on the oracle's
`--reasoning-format`.

## Calibrating a new model/quant pair

The envelope comes from the oracle's own prefill-vs-decode disagreement,
measured with `g4-calibrate`:

1. Load the GGUF in a freshly started pinned llama.cpp server.
2. Capture an oracle arm: `g4 --engine oracle --out /tmp/arm.json`
   (the arm stores the prompts and per-position top-20 logprobs).
3. Re-score the arm through the prefill path:
   `g4-calibrate --capture /tmp/arm.json --out <calibration>.json`.
   Every position whose captured top1-top2 gap is within `--max-gap`
   (default 0.5, the near-tie region) is re-sent as an assistant-message
   prefill, so the scored position is computed by the batched-prefill
   path instead of the decode path that produced the capture. Any top-1
   disagreement is the oracle flipping against itself.
4. Choose the envelope at the desired coverage (P97 of flip deltas and
   the full observed rank range was chosen for both calibrations so far)
   and record it in the calibration file's `envelope` field.

Calibration files live in `tools/gates/calibration/<model>-<quant>.json`
and carry their `envelope`; `g4-compare` takes the envelope from
`--calibration <file>` (or explicit `--max-delta/--max-rank`) and refuses
to run without one — there is no cross-model default. The first
calibration, Q4_K_XL on a 40B MoE (2026-11-08):

- 397 near-tie positions measured
- the oracle flips 15.9% of them (63/397) against itself
- P95 self-flip delta 0.709, max 1.335, all at decode gap < 0.478
- zero structural disagreements (no flip beyond rank 4)

The envelope delta <= 0.8 / rank <= 4 covers P97 of the oracle's own
flips and the full observed rank range. The previous envelope
(delta <= 0.5, rank <= 3) was stricter than the oracle's own
self-consistency on Q4 and produced false failures.

Second calibration, UD-Q4_K_XL on Qwen3.8-Flash-Next (2026-11-13): 591
near-tie positions, 28.1% self-flip rate, P97 delta 0.513, max 0.825,
flips out to rank 6, zero structural disagreements — envelope delta <=
0.55 / rank <= 6. The Flint envelope does not transfer to this model:
4 of 166 flips fall outside it, which is why the compare refuses to run
without an explicit calibration.

Raw data: the `flips` field of each calibration file (per-position
records: capture vs rescored top-1, delta, rank, decode gap), plus the
`summary` and `sanity` blocks.

Coarser quants need looser envelopes: early IQ-format data measured
roughly 2x the Q4 envelope. Every new model/quant pair gets its own
calibration run before its gate verdict is published; the per-pair
envelopes are recorded next to the run's evidence.
