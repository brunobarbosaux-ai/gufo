#!/usr/bin/env python3
"""Compare the Gufo flint CPU reference log-prob dump against the pinned
llama.cpp CPU oracle dump (D4 validation: CPU reference vs CPU oracle).

Cross-implementation tolerance (requirements 17.9): llama.cpp's own CPU and
GPU stacks flip ~8 % of argmaxes on flat distributions, so this gate checks
SEMANTICS (sinks, window, NoPE, router), not bit parity. Pass bar: argmax
agreement at or above the 92 % cross-backend baseline and no systematic
multi-nat divergence.

Usage: compare_cpu_reference.py ORACLE.f32 GUFO.raw [--row0 N] [--rows M]
  ORACLE rows [0..31] are chunk-0 (full prefix); GUFO.raw row r predicts
  token index r+1, so oracle row j aligns to GUFO row (32 + j) by default.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from logit_dump import parse_dump, dequant_row  # noqa: E402

import struct


def main():
    oracle_path, gufo_path = Path(sys.argv[1]), Path(sys.argv[2])
    row0, rows = 32, 32
    for i, a in enumerate(sys.argv):
        if a == "--row0":
            row0 = int(sys.argv[i + 1])
        if a == "--rows":
            rows = int(sys.argv[i + 1])

    meta, tokens, eval_indices = parse_dump(oracle_path)
    vocab = meta["n_vocab"]
    gufo_data = gufo_path.read_bytes()
    gufo_rows = len(gufo_data) // (vocab * 4)
    if gufo_rows < row0 + rows:
        sys.exit(f"gufo dump has {gufo_rows} rows, need {row0 + rows}")

    max_delta, argmax_hits, cosines = 0.0, 0, []
    worst = None
    for j in range(rows):
        oracle_lp, _, _ = dequant_row(oracle_path.read_bytes(), meta, j)
        off = (row0 + j) * vocab * 4
        gufo_lp = list(struct.unpack_from(
            f"<{vocab}f", gufo_data, off))

        o_am = max(range(vocab), key=lambda v: oracle_lp[v])
        g_am = max(range(vocab), key=lambda v: gufo_lp[v])
        if o_am == g_am:
            argmax_hits += 1

        d = max(abs(a - b) for a, b in zip(oracle_lp, gufo_lp)
                if a > -1e30 and b > -1e30)
        if d > max_delta:
            max_delta, worst = d, (j, eval_indices[j], o_am, g_am)

        dot = sum(a * b for a, b in zip(oracle_lp, gufo_lp))
        na = math.sqrt(sum(a * a for a in oracle_lp))
        nb = math.sqrt(sum(b * b for b in gufo_lp))
        cosines.append(dot / (na * nb) if na and nb else 0.0)

    print(f"rows: {rows} (oracle eval indices "
          f"{eval_indices[0]}..{eval_indices[rows - 1]})")
    print(f"argmax agreement: {argmax_hits}/{rows} "
          f"({100.0 * argmax_hits / rows:.1f}%)")
    print(f"cosine: min {min(cosines):.6f} mean "
          f"{sum(cosines) / len(cosines):.6f}")
    print(f"max |delta log p|: {max_delta:.4f} "
          f"(row {worst[0]}, idx {worst[1]}, oracle {worst[2]} vs gufo "
          f"{worst[3]})")


if __name__ == "__main__":
    main()
