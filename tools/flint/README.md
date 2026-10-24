# Flint development tools

Oracle harness for the Flint 40B AR port. The oracle is llama.cpp
**b11804** (`0.7.0-dev, commit c7fc50941`), which carries `flint` support
in `libllama.so`, against the pinned artifact
`/home/bruno/AI/GGUF/flint-UD-Q6_K_XL.gguf`
(sha256 `08623109…41279c2`, 34 GiB). Full context:
`infra/strix-halo/gufo-flint-ar-requirements.md` (M0, G1–G5).

| Tool | Purpose |
| --- | --- |
| `serve_gates.py` | F-L7 serve-path gates (G4, G5, G8) against running endpoints. `g4` captures arms (oracle with top-20 logprobs, one generation per prompt — **restart the oracle first**: repeats hit its cached-decode path and flip near-ties); `g4-compare` verifies every greedy fork against the oracle's own top-k (rank ≤ 3, logprob delta ≤ 0.5); `g5` captures T1.0/p0.95/k20 distribution arms (first word of the full stream — the answer is empty at small max_tokens for a reasoning model); `g5-verdict` applies the re-scoped criterion (support + ordering + TV ≤ 0.25); `g8` concurrency identity. See the requirements doc §G4/§G5 for the calibration |
| `oracle_server.sh [cpu\|gpu]` | Pinned oracle server, port 8098. `cpu` uses `--device none` (no GPU) and is enough for tokenize/template goldens; `gpu` full-offloads for logit dumps and G4/G5 arms. Note: this llama.cpp build wants `--load-mode mmap` (not `--mmap`) |
| `corpus.py OUTDIR [--probe]` | Deterministic F-H2 corpus: 2 700 cases in nine classes, plus the G3 depth probes (64 / 600 / 4096 / 16384 nominal tokens; 600 sits just past the 513 window) |
| `golden.py tokenize OUTDIR` | M0.6: `/tokenize` goldens for the whole corpus, `add_special` true and false |
| `golden.py template OUTDIR` | M0.7: `/apply-template` goldens for the G2 fixtures (roles, reasoning, tools, malformed calls, escaping) |
| `golden.py compare A.json B.json` | Token-id diff of two goldens; reports the first divergence with context |
| `logit_dump.py OUTDIR [--gpu]` | M0.5: `llama-perplexity --save-all-logits` per depth; the `_logits_` u16 log-prob format is decoded and parsed (`parse_dump`/`dequant_row`); tokens cross-checked against `/tokenize` |
| `compare_dumps.py A.f32 B.f32` | G3 comparator skeleton: dequantises two dumps and reports max \|Δ log p\| (head), cosine min/mean, argmax agreement, floor disagreements |
| `g1_cases.py GOLDEN.json OUT.bin` | Joins the regenerated corpus with the golden ids into the packed G1 case file (refuses to write if any case id drifts) |
| `tokenizer_probe.cpp` | G1 probe: loads the GGUF tokenizer and compares to the golden (`MODEL.gguf CASES.bin`), or encodes one string (`MODEL.gguf --text TEXT`); exits 1 on any mismatch |
| `template_probe.cpp` + `g2_cases.py` | G2 probe: renders the 12 template fixtures through `FlintChatTemplate` and compares bytes + token ids to the golden |
| `flint_cpu_logit.cpp` | D4 CPU reference dump: runs the flint CPU forward over a probe text and writes vocab-wide f32 log-prob rows (`MODEL.gguf PROBE.txt OUT.raw [--max N]`) |
| `compare_cpu_reference.py ORACLE.f32 GUFO.raw` | D4 validation: oracle chunk-0 rows (full prefix) vs the CPU reference rows; argmax agreement + cosine + max \|Δ log p\| at cross-implementation tolerance (§17.9) |

Goldens are artifacts and stay outside Git; each run writes a sha256 into
`OUTDIR/manifest.json` alongside the oracle build and model identity.
Default artifact directory: `/home/bruno/AI/Gufo-testing/flint-oracle/`.

```sh
OUT=/home/bruno/AI/Gufo-testing/flint-oracle
python3 tools/flint/corpus.py "$OUT" --probe
python3 tools/flint/golden.py tokenize "$OUT"            # spawns cpu oracle
python3 tools/flint/golden.py template "$OUT"
python3 tools/flint/logit_dump.py "$OUT" --gpu           # needs the GPU window
```

Gate mapping: M0.4 launcher · M0.5 `logit_dump.py` (G3) · M0.6
`golden.py tokenize` (G1) · M0.7 `golden.py template` (G2). G4/G5 reuse
the same pinned server with the sampling paths.
