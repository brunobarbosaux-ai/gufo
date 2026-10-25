#!/usr/bin/env python3
"""Deterministic F-J4 bench docs (0.1): msg / doc / long prompt files.

The nemotron bench's "same 6.6K doc" was session-generated and never
committed; these files are the reproducible replacement. Fixed seed,
fixed bytes — same prompts on every future run, so Gufo-vs-llama.cpp
comparisons stay prompt-identical. Content is agent-style long-form
prose (short paragraphs, technical vocabulary, numbers) to match the
real traffic the llama.cpp numbers were taken on.

Targets ~181 / ~6.6K / ~23.5K tokens at flint's tokenizer (~3.9 chars
per token for this prose). Actual counts are reported by fj4_bench.py
from the engine's own usage block — quote those, not the targets.

Usage: python3 tools/flint/fj4_docs.py [OUTDIR]
Writes msg.txt, doc.txt, long.txt + prints byte sizes.
"""

import pathlib
import random
import sys

SEED = 40961

# Technical-prose vocabulary: two-part sentences with numbers keep the
# tokenizer in the regime the 6.6K bench doc used (plain English prose).
SUBJECTS = (
    "the scheduler", "the ingest pipeline", "the cache layer",
    "the allocator", "the replica set", "the queue pair", "the shard map",
    "the compaction thread", "the watchdog", "the indexer",
    "the speculative decoder", "the router", "the verifier",
    "the checkpoint writer", "the metrics emitter",
)
VERBS = (
    "walks", "rebases", "flushes", "drains", "retries", "rotates",
    "prefetches", "compacts", "balances", "validates", "truncates",
    "pins", "spills", "merges", "elides",
)
OBJECTS = (
    "the segment table", "every dirty page", "the manifest",
    "the bloom filter", "twelve open descriptors", "the watermark",
    "a stale lease", "the parquet footer", "each cold bucket",
    "the delta log", "the header block", "its private arena",
    "the per-epoch budget", "the write-ahead tail",
)
CLAUSES = (
    "before the barrier releases", "after the lease expires",
    "when the quota is exceeded", "unless the checksum matches",
    "while the follower catches up", "once the tail is fsynced",
    "if the epoch advances", "because the chunk header moved",
    "so the read path stays flat", "though the median regresses",
    "as the window slides forward", "until the referee responds",
)
NUMBERS = (17, 23, 391, 512, 1024, 4096, 150144, 65536, 8192, 2048)


def sentence(rng: random.Random) -> str:
    n = rng.choice(NUMBERS)
    return (f"{rng.choice(SUBJECTS).capitalize()} {rng.choice(VERBS)} "
            f"{rng.choice(OBJECTS)} {rng.choice(CLAUSES)}, at {n} "
            f"{rng.choice(['bytes', 'entries', 'tokens', 'pages', 'rows'])} "
            f"per {rng.choice(['pass', 'epoch', 'window', 'chunk'])}.")


def paragraph(rng: random.Random, size: int) -> str:
    parts = []
    used = 0
    while used < size:
        s = sentence(rng)
        parts.append(s)
        used += len(s) + 1
    return " ".join(parts)


def build(target_chars: int) -> str:
    rng = random.Random(SEED)
    out = []
    size = 0
    while size < target_chars:
        p = paragraph(rng, min(rng.randint(280, 620),
                               target_chars - size))
        out.append(p)
        size += len(p) + 2
    return "\n\n".join(out) + "\n"


def main() -> None:
    outdir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                          else "tools/flint/fixtures/fj4")
    outdir.mkdir(parents=True, exist_ok=True)
    # msg ~181 tok, doc ~6.6K tok, long ~23.5K tok at ~3.9 chars/token.
    for name, chars in (("msg.txt", 850), ("doc.txt", 31_000),
                        ("long.txt", 110_000)):
        text = build(chars)
        path = outdir / name
        path.write_text(text)
        print(f"{path}: {len(text)} bytes")


if __name__ == "__main__":
    main()
