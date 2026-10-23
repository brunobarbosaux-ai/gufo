#!/usr/bin/env python3
"""Depth-probe logit oracle (G3 / M0.5).

Runs llama.cpp b11804 `llama-perplexity --save-all-logits` on the four
probe texts and stores the oracle's own dump per depth, plus metadata.

Dump format (verified against perplexity.cpp at the pinned commit
c7fc50941): magic "_logits_", then int32 n_ctx / n_vocab / n_chunk, then
n_chunk*n_ctx int32 token ids, then per chunk (n_ctx - 1 - n_ctx/2)
rows of nv uint16, nv = 2*((n_vocab+1)/2) + 4. Row layout: f32 scale,
f32 min_log_prob, then n_vocab uint16 quantised log-probs with
log_p[i] = scale * q[i] + min_log_prob (quantisation floor: a 16-nat
window in 65535 steps, ~1.2e-4 nats). Rows cover only the conditioned
half of each chunk: chunk c row j sits at context position
c*n_ctx + n_ctx/2 + j and predicts the token after it. G3 therefore
compares dequantised log-probabilities (quant-aware threshold), not raw
logits - the raw-logit 0.05 threshold from the draft is superseded by
max |delta log p| (see the requirements doc, oracle findings).

Perplexity refuses inputs shorter than 2x context, so the probe text is
tokenized first (oracle /tokenize) and the context is sized to n/2.

Usage:
  python3 tools/flint/logit_dump.py OUTDIR --gpu            # the real thing
  python3 tools/flint/logit_dump.py OUTDIR                  # CPU works too
  python3 tools/flint/logit_dump.py OUTDIR --depths 64      # one depth
"""

import argparse
import hashlib
import json
import pathlib
import struct
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
LLAMA_DIR_DEFAULT = "/home/bruno/AI/llama/llama-b11804-rocm"
MODEL_DEFAULT = "/home/bruno/AI/GGUF/flint-UD-Q6_K_XL.gguf"
THREADS = 32
MAGIC = b"_logits_"


def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_dump(path):
    """Parse a --save-all-logits file. Returns (meta, tokens, row_index)."""
    data = path.read_bytes()
    if data[:8] != MAGIC:
        sys.exit("{}: bad magic - not a --save-all-logits dump".format(path))
    n_ctx, n_vocab, n_chunk = struct.unpack_from("<iii", data, 8)
    off = 20
    n_stored = n_chunk * n_ctx
    tokens = list(struct.unpack_from("<{}i".format(n_stored), data, off))
    off += 4 * n_stored
    nv = 2 * ((n_vocab + 1) // 2) + 4
    rows_per_chunk = n_ctx - 1 - n_ctx // 2
    n_rows = n_chunk * rows_per_chunk
    if len(data) != off + n_rows * nv * 2:
        sys.exit("{}: size mismatch (expected {} rows of {} u16)".format(
            path, n_rows, nv))
    # Row (c, j) conditions on positions < p and predicts token index p,
    # p = c*n_ctx + n_ctx//2 + j + 1.
    eval_indices = [
        c * n_ctx + n_ctx // 2 + j + 1
        for c in range(n_chunk) for j in range(rows_per_chunk)
    ]
    meta = {
        "n_ctx": n_ctx, "n_vocab": n_vocab, "n_chunk": n_chunk,
        "rows": n_rows, "nv": nv, "rows_per_chunk": rows_per_chunk,
        "row_u16": nv, "quant": "u16 log-probs, 16-nat window, f32 scale+min",
    }
    return meta, tokens, eval_indices


def row_offset(meta, row):
    return 20 + meta["n_chunk"] * meta["n_ctx"] * 4 + row * meta["nv"] * 2


def dequant_row(data, meta, row):
    """Dequantise one row into n_vocab f32 log-probs."""
    base = row_offset(meta, row)
    scale, min_log_prob = struct.unpack_from("<2f", data, base)
    q = struct.unpack_from("<{}H".format(meta["n_vocab"]), data, base + 8)
    return [scale * v + min_log_prob for v in q], scale, min_log_prob


def run(args):
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    llama_dir = pathlib.Path(args.llama_dir)
    pp = llama_dir / "llama-perplexity"
    if not pp.exists():
        sys.exit("missing {}".format(pp))

    from golden import Oracle  # same directory

    with Oracle(args.base_url, gpu=args.gpu) as oracle:
        print("oracle: model_path={}".format(oracle.props.get("model_path", "unknown")))

        for depth in args.depths:
            probe = outdir / "probe-{}.txt".format(depth)
            if not probe.exists():
                sys.exit("missing {} — run: python3 tools/flint/corpus.py {} --probe".format(
                    probe, outdir))
            text = probe.read_text(encoding="utf-8")

            ref_tokens = oracle.tokenize(text, False)
            n = len(ref_tokens)
            if n < 4:
                sys.exit("depth {}: probe tokenizes to {} tokens - too small".format(depth, n))
            ctx = n // 2  # perplexity requires >= 2x ctx tokens
            batch = min(2048, ctx)
            print("depth {}: {} tokens, ctx {}".format(depth, n, ctx))

            dump = outdir / "logits-{}.f32".format(depth)
            cmd = [str(pp), "--model", args.model, "-f", str(probe),
                   "--save-all-logits", str(dump),
                   "-c", str(ctx), "-b", str(batch), "-ub", str(batch), "-fa", "off"]
            if args.gpu:
                cmd += ["-ngl", "99"]
            else:
                cmd += ["-ngl", "0", "--device", "none",
                        "-t", str(THREADS), "-tb", str(THREADS)]
            print("  {}".format(" ".join(cmd)))
            subprocess.run(cmd, check=True, stdout=sys.stderr, stderr=sys.stderr)

            meta, file_tokens, eval_indices = parse_dump(dump)
            # Cross-check: the dump's own token array must equal the server
            # tokenisation over the stored span (add_bos_token=false, so no
            # special tokens on either side).
            span = len(file_tokens)
            token_check = file_tokens == ref_tokens[:span]
            if not token_check:
                print("  WARNING: dump tokens != /tokenize ids over {} tokens".format(span))

            meta.update({
                "depth": depth,
                "gpu": bool(args.gpu),
                "backend": "rocm" if args.gpu else "cpu",
                "n_tokens_server": n,
                "token_check": token_check,
                "eval_token_indices": eval_indices,
                "sha256": _sha(dump),
            })
            (outdir / "logits-{}.json".format(depth)).write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print("  ok: {} rows, {} eval positions, token_check={}, sha {}…".format(
                meta["rows"], len(eval_indices), token_check, meta["sha256"][:16]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir", type=pathlib.Path)
    ap.add_argument("--gpu", action="store_true", help="full offload (CPU also works, slower)")
    ap.add_argument("--depths", type=int, nargs="+", default=[64, 600, 4096, 16384])
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--llama-dir", default=LLAMA_DIR_DEFAULT)
    ap.add_argument("--base-url", default=None, help="running oracle for /tokenize cross-check")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
