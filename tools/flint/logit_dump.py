#!/usr/bin/env python3
"""Depth-probe logit oracle (G3 / M0.5).

Runs llama.cpp b11804 `llama-perplexity --save-all-logits` on the four
probe texts (32 / 600 / 4096 / 16384 tokens) and stores, per depth, the
teacher-forced token ids and full-vocab logits from the pinned oracle.
Gufo's flint_layer_probe (M2) must emit the same layout so the comparison
is a plain array diff.

The perplexity dump carries logits only, so token ids come from the
server /tokenize endpoint of the same build with add_special=false.
No BOS is expected either way: the artifact sets add_bos_token=false.

Usage:
  python3 tools/flint/logit_dump.py OUTDIR --gpu            # the real thing
  python3 tools/flint/logit_dump.py OUTDIR --depths 32      # CPU smoke
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
N_VOCAB = 150144
THREADS = 32


def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tokenize_probe(base_url, text):
    import urllib.request
    req = urllib.request.Request(
        base_url + "/tokenize",
        data=json.dumps({"content": text, "add_special": False}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())["tokens"]


def _read_logits(path, n_rows_expected):
    """Raw f32 rows written by --save-all-logits. Returns list of rows."""
    size = path.stat().st_size
    if size % (4 * N_VOCAB) != 0:
        sys.exit("{}: size {} not a multiple of {} f32 - unknown format".format(
            path, size, N_VOCAB))
    n_rows = size // (4 * N_VOCAB)
    if n_rows != n_rows_expected:
        sys.exit("{}: {} rows, expected {} (tokenisation mismatch?)".format(
            path, n_rows, n_rows_expected))
    rows = []
    with path.open("rb") as fh:
        for _ in range(n_rows):
            rows.append(struct.unpack("<{}f".format(N_VOCAB), fh.read(4 * N_VOCAB)))
    return rows


def run(args):
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    llama_dir = pathlib.Path(args.llama_dir)
    pp = llama_dir / "llama-perplexity"
    server = llama_dir / "llama-server"

    if not pp.exists():
        sys.exit("missing {}".format(pp))

    for depth in args.depths:
        probe = outdir / "probe-{}.txt".format(depth)
        if not probe.exists():
            sys.exit("missing {} — run: python3 tools/flint/corpus.py {} --probe".format(probe, outdir))
        text = probe.read_text(encoding="utf-8")

        dump = outdir / "logits-{}.f32".format(depth)
        ctx = depth + 64
        cmd = [str(pp), "--model", args.model, "-f", str(probe),
               "--save-all-logits", str(dump),
               "-c", str(ctx), "-b", "2048", "-ub", "2048", "-fa", "off"]
        if args.gpu:
            cmd += ["-ngl", "99"]
        else:
            cmd += ["-ngl", "0", "--device", "none", "-t", str(THREADS), "-tb", str(THREADS)]
        print("depth {}: {}".format(depth, " ".join(cmd)))
        subprocess.run(cmd, check=True, stdout=sys.stderr, stderr=sys.stderr)

        # Token ids from the same build's server, for pairing with the dump.
        # Spun up only for the metadata call; harmless if already running.
        tokens = None
        from golden import Oracle  # noqa: E402  (same directory)
        with Oracle(args.base_url, gpu=args.gpu) as oracle:
            tokens = oracle.tokenize(text, False)
        rows = _read_logits(dump, len(tokens))
        meta = {
            "depth": depth,
            "n_tokens": len(tokens),
            "n_vocab": N_VOCAB,
            "gpu": bool(args.gpu),
            "backend": "rocm" if args.gpu else "cpu",
            "logits_sha256": _sha(dump),
        }
        sidecar = outdir / "logits-{}.json".format(depth)
        sidecar.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        # Rows stay in the raw .f32; tokens go beside them for the diff.
        tok_path = outdir / "logits-{}.tokens.json".format(depth)
        tok_path.write_text(json.dumps(tokens) + "\n", encoding="utf-8")
        print("depth {}: {} tokens, logits sha256 {}…".format(depth, len(tokens), meta["logits_sha256"][:16]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir", type=pathlib.Path)
    ap.add_argument("--gpu", action="store_true", help="full offload (default is a CPU-only run)")
    ap.add_argument("--depths", type=int, nargs="+", default=[32, 600, 4096, 16384])
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--llama-dir", default=LLAMA_DIR_DEFAULT)
    ap.add_argument("--base-url", default=None, help="running oracle for /tokenize pairing")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
