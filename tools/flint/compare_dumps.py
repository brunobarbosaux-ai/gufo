#!/usr/bin/env python3
"""Compare two --save-all-logits dumps (G3 comparator skeleton, M0.5 calibration).

Reads two dumps in the format parsed by logit_dump.parse_dump, aligns rows by
eval position, dequantises the u16 log-prob rows, and reports the divergence:

- max |delta log p| over values both sides encode inside their quant window
  (floored tail values — clamped to the 16-nat window edge — are excluded from
  the max, and floor disagreements are counted separately; a 1-nat tail clamp
  difference is a format artefact, not a kernel difference);
- cosine of the probability vectors per row (min and mean);
- floor-mismatch counts.

This is exactly the metric pair G3 gates on (cosine >= 0.999, max |d log p|
<= 5e-3) — run today CPU-vs-ROCm to calibrate what "same weights, different
kernels" costs; in M2 the same tool compares Gufo's probe output against the
oracle dump.

Usage:
  python3 tools/flint/compare_dumps.py A.f32 B.f32 [--stride N] [--json OUT]
"""

import argparse
import json
import math
import pathlib
import sys

from logit_dump import parse_dump, MAGIC

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _rows_array(path):
    """All row bytes as a (rows, nv) uint16 numpy view, header stripped."""
    meta, tokens, evals = parse_dump(path)
    data = path.read_bytes()
    off = 20 + meta["n_chunk"] * meta["n_ctx"] * 4
    body = np.frombuffer(data, dtype=np.uint8, offset=off)
    rows = body.reshape(meta["rows"], meta["nv"] * 2)
    return meta, tokens, evals, rows


def _dequant(rows_np, n_vocab):
    """Dequantise a (rows, nv*2) u16-byte matrix to f32 log-probs (rows, n_vocab)."""
    scale = rows_np[:, 0:4].copy().view(np.float32).ravel()      # f32 scale per row
    mn = rows_np[:, 4:8].copy().view(np.float32).ravel()         # f32 min per row
    q = rows_np[:, 8:8 + n_vocab * 2].copy().view(np.uint16)     # (rows, n_vocab)
    return q.astype(np.float32) * scale[:, None] + mn[:, None], mn


def compare(path_a, path_b, stride):
    meta_a, tok_a, ev_a, raw_a = _rows_array(path_a)
    meta_b, tok_b, ev_b, raw_b = _rows_array(path_b)
    if meta_a["n_vocab"] != meta_b["n_vocab"]:
        sys.exit("vocab mismatch: {} vs {}".format(meta_a["n_vocab"], meta_b["n_vocab"]))
    if ev_a != ev_b:
        sys.exit("eval positions differ ({} rows vs {} rows, misaligned)".format(
            len(ev_a), len(ev_b)))
    tok_match = tok_a == tok_b

    n_vocab = meta_a["n_vocab"]
    lp_a, mn_a = _dequant(raw_a, n_vocab)
    lp_b, mn_b = _dequant(raw_b, n_vocab)

    if stride > 1:
        lp_a, lp_b, mn_a, mn_b = (lp_a[::stride], lp_b[::stride], mn_a[::stride], mn_b[::stride])

    eps = 1e-6
    floor_a = mn_a[:, None] + eps
    floor_b = mn_b[:, None] + eps
    in_head = (lp_a > floor_a) & (lp_b > floor_b)   # both encoded inside the window
    a_floored_only = (lp_a <= floor_a) & (lp_b > floor_b)
    b_floored_only = (lp_b <= floor_b) & (lp_a > floor_a)

    delta = np.abs(lp_a - lp_b)
    head_max = float(delta[in_head].max()) if in_head.any() else 0.0
    head_mean = float(delta[in_head].mean()) if in_head.any() else 0.0

    argmax_agree = float((lp_a.argmax(axis=1) == lp_b.argmax(axis=1)).mean())

    # Cosine on probability vectors, per row.
    p_a = np.exp(lp_a - lp_a.max(axis=1, keepdims=True))
    p_b = np.exp(lp_b - lp_b.max(axis=1, keepdims=True))
    num = (p_a * p_b).sum(axis=1)
    den = np.linalg.norm(p_a, axis=1) * np.linalg.norm(p_b, axis=1)
    cos = num / np.maximum(den, 1e-300)

    return {
        "rows_compared": int(lp_a.shape[0]),
        "stride": stride,
        "tokens_match": bool(tok_match),
        "max_abs_delta_logp_head": head_max,
        "mean_abs_delta_logp_head": head_mean,
        "argmax_agreement": argmax_agree,
        "cosine_min": float(cos.min()),
        "cosine_mean": float(cos.mean()),
        "a_floored_b_not": int(a_floored_only.sum()),
        "b_floored_a_not": int(b_floored_only.sum()),
        "n_vocab": n_vocab,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump_a", type=pathlib.Path)
    ap.add_argument("dump_b", type=pathlib.Path)
    ap.add_argument("--stride", type=int, default=1, help="compare every Nth row")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    if args.dump_a.read_bytes()[:8] != MAGIC or args.dump_b.read_bytes()[:8] != MAGIC:
        sys.exit("bad magic on input")
    r = compare(args.dump_a, args.dump_b, args.stride)
    label = "{} vs {}".format(args.dump_a.parent.name, args.dump_b.parent.name)
    print("{}: rows={} stride={} tokens_match={}".format(
        label, r["rows_compared"], r["stride"], r["tokens_match"]))
    print("  max |d log p| (head): {:.3e}   mean: {:.3e}   argmax agree: {:.1%}".format(
        r["max_abs_delta_logp_head"], r["mean_abs_delta_logp_head"], r["argmax_agreement"]))
    print("  cosine: min {:.9f}  mean {:.9f}".format(r["cosine_min"], r["cosine_mean"]))
    print("  floor disagreements: a-only {}  b-only {}".format(
        r["a_floored_b_not"], r["b_floored_a_not"]))
    if args.json:
        args.json.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
