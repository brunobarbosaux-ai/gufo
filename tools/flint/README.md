# Flint development tools

Oracle harness for the Flint 40B AR port. The oracle is llama.cpp
**b11804** (`0.7.0-dev, commit c7fc50941`), which carries `flint` support
in `libllama.so`, against the pinned artifact
`/home/bruno/AI/GGUF/flint-UD-Q6_K_XL.gguf`
(sha256 `08623109…41279c2`, 34 GiB). Full context:
`infra/strix-halo/gufo-flint-ar-requirements.md` (M0, G1–G5).

| Tool | Purpose |
| --- | --- |
| `oracle_server.sh [cpu\|gpu]` | Pinned oracle server, port 8098. `cpu` uses `--device none` (no GPU) and is enough for tokenize/template goldens; `gpu` full-offloads for logit dumps and G4 identity |
| `corpus.py OUTDIR [--probe]` | Deterministic F-H2 corpus: 2 700 cases in nine classes, plus the G3 depth probes (32 / 600 / 4096 / 16384 tokens; 600 sits just past the 513 window) |
| `golden.py tokenize OUTDIR` | M0.6: `/tokenize` goldens for the whole corpus, `add_special` true and false |
| `golden.py template OUTDIR` | M0.7: `/apply-template` goldens for the G2 fixtures (roles, reasoning, tools, malformed calls, escaping) |
| `golden.py compare A.json B.json` | Token-id diff of two goldens; reports the first divergence with context |
| `logit_dump.py OUTDIR --gpu` | M0.5: `llama-perplexity --save-all-logits` per depth, paired with `/tokenize` ids; Gufo's `flint_layer_probe` must emit the same layout |

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
