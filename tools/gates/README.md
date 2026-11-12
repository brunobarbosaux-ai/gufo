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
python3 tools/gates/serve_gates.py g4-compare /tmp/g4-oracle.json /tmp/g4-serve.json

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
fixed seed, nine prompt classes) and trims prompts to depth with the
oracle's `/tokenize`.

## Calibrating a new model/quant pair

1. Load the GGUF in a freshly started pinned llama.cpp server.
2. Run `g4 --engine oracle` twice with different prompt subsets and
   compare the arms with `g4-compare --max-delta 0 --max-rank 1`: any
   disagreement is the oracle flipping against itself. The per-position
   flip records from a prefill-vs-decode comparison are the calibration
   sample (see the `flips` field in the evidence JSON for the schema).
3. Choose the envelope at the desired coverage (P97 was chosen for Q4)
   and record it with the run next to the model's other evidence.
