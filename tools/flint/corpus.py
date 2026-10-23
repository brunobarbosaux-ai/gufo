#!/usr/bin/env python3
"""Deterministic Flint parity corpus - requirement F-H2 (7.3).

Nine classes, 2700 prompts, fixed seed. The corpus is the contract for G1
(tokenizer parity) and feeds the G3 probe prompts. Same seed, same bytes,
on any machine.

Classes and counts (F-H2):
  ascii 500 / code 500 / cjk 300 / mixed 200 / numeric 200
  contraction 300 / whitespace 300 / tool_xml 200 / long 200

Probe depths for the logit oracle (G3): 64 / 600 / 4096 / 16384 tokens (nominal -
perplexity requires inputs of at least 2x context, so the actual count comes from
/tokenize at dump time and the context is sized to n/2). 600 sits just past the
513 sliding window; 4096 and 16384 exercise window wrap-around at depth.

Usage:
  python3 tools/flint/corpus.py OUTDIR            # writes corpus.jsonl
  python3 tools/flint/corpus.py OUTDIR --probe    # writes probe-<n>.txt files
"""

import argparse
import json
import pathlib
import random

SEED = 1151

COUNTS = {
    "ascii": 500,
    "code": 500,
    "cjk": 300,
    "mixed": 200,
    "numeric": 200,
    "contraction": 300,
    "whitespace": 300,
    "tool_xml": 200,
    "long": 200,
}

PROBE_DEPTHS = (64, 600, 4096, 16384)

ASCII_WORDS = (
    "alpha bravo charlie delta echo foxtrot hotel india juliet kilo lima"
    " mike november oscar papa quebec romeo sierra tango uniform victor"
    " whiskey xray yankee zulu the quick brown fox jumps over lazy dog"
).split()

CJK_SENTENCES = (
    "東京タワーは高さ三百三十三メートルです。",
    "语言模型需要统一的分词规则。",
    "데이터는 정확해야 합니다.",
    "数字一二三と números mezclados。",
    "多次元の配列を扱う場合はインデックスに注意。",
)


def _ascii(rng):
    return " ".join(rng.choice(ASCII_WORDS) for _ in range(rng.randint(3, 40)))


def _code(rng):
    lines = []
    for _ in range(rng.randint(2, 12)):
        kind = rng.randrange(4)
        if kind == 0:
            lines.append("{}x{} = fn({});".format(" " * rng.randrange(0, 12), rng.randrange(1000), rng.randrange(999)))
        elif kind == 1:
            lines.append("# path: /var/log/app-{}.log  // TODO".format(rng.randrange(100)))
        elif kind == 2:
            lines.append("if (a{}{}b >= {}) return {};".format(
                " " * rng.randrange(1, 3),
                rng.choice(["==", "!=", "<", ">", "<=", ">="]),
                rng.randrange(100), rng.randrange(10)))
        else:
            lines.append("const v{} = {}{}{};".format(
                rng.randrange(100), rng.randrange(1000),
                rng.choice([" + ", " - ", " * ", " / ", " % "]), rng.randrange(1000)))
    return "\n".join(lines)


def _cjk(rng):
    parts = []
    for _ in range(rng.randint(1, 6)):
        parts.append(rng.choice(CJK_SENTENCES))
        if rng.random() < 0.4:
            parts.append(rng.choice(["、", "。", "！", "？", " ", ""]))
    return "".join(parts)


def _mixed(rng):
    emoji = rng.choice(["🚀", "👍🏽", "🇯🇵", "👨‍👩‍👧", "✅"])
    cyr = "".join(chr(rng.randrange(0x410, 0x44F)) for _ in range(rng.randint(2, 8)))
    ara = "".join(chr(rng.randrange(0x600, 0x6FF)) for _ in range(rng.randint(2, 8)))
    lat = rng.choice(ASCII_WORDS)
    order = [emoji, cyr, ara, lat]
    rng.shuffle(order)
    sep = rng.choice([" ", "", " · ", " "])
    return sep.join(order)


def _numeric(rng):
    forms = [
        str(rng.randint(0, 10**9)),
        "{:,}".format(rng.randint(1000, 999999)),
        "{:.5f}".format(rng.random()),
        hex(rng.randrange(0x10000)),
        "{}.{}.{}-rc{}".format(rng.randint(0, 9), rng.randint(0, 20), rng.randint(0, 30), rng.randint(0, 9)),
        "{}-{:02d}-{:02d}".format(2026, rng.randint(1, 12), rng.randint(1, 28)),
        "{} {}".format(rng.randint(1, 999), rng.choice(["kg", "m/s", "°C", "%"])),
        "1234",
        "12,345",
        "3.14159",
        "0x1F",
    ]
    return rng.choice(forms)


def _contraction(rng):
    stems = ["they", "They", "we", "you", "it", "she", "he", "I", "cat", "Bob", "the model"]
    suffixes = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
    parts = []
    for _ in range(rng.randint(1, 10)):
        parts.append(rng.choice(stems) + rng.choice(suffixes))
        parts.append(" ")
    return "".join(parts).strip()


def _whitespace(rng):
    pieces = []
    for _ in range(rng.randint(2, 10)):
        kind = rng.randrange(6)
        if kind == 0:
            pieces.append("\n" * rng.randint(2, 6))
        elif kind == 1:
            pieces.append(" " * rng.randint(2, 20) + "word")
        elif kind == 2:
            pieces.append("\t" * rng.randint(1, 4) + "tabbed")
        elif kind == 3:
            pieces.append("line\r\nbreak\r\n")
        elif kind == 4:
            pieces.append("end   \n")
        else:
            pieces.append(" ")
    return "".join(pieces)


def _tool_xml(rng):
    names = ["get_weather", "search", "create_file", "run_query"]
    name = rng.choice(names)
    forms = [
        '<{} town="Curitiba" unit="c"/>'.format(name),
        '<{} q="{}" limit="{}"/>'.format(name, rng.choice(ASCII_WORDS), rng.randint(1, 50)),
        '<{} a={{"nested": [1, 2, {{"k": "v"}}]}} b=""/>'.format(name),
        '<{} name="名前" emoji="🚀"/>'.format(name),
        "<{} />".format(name),
        "<{}>text &amp; more</{}>".format(name, name),
        "<{} broken=\"unterminated".format(name),
    ]
    return rng.choice(forms)


def _long(rng):
    target = rng.randint(4 * 1024, 40 * 1024)
    chunks = []
    size = 0
    while size < target:
        piece = "{} value={}；{}\n{}\n".format(
            _ascii(rng), rng.randrange(10**6), _cjk(rng), _code(rng))
        chunks.append(piece)
        size += len(piece)
    return "".join(chunks)


_GENERATORS = {
    "ascii": _ascii,
    "code": _code,
    "cjk": _cjk,
    "mixed": _mixed,
    "numeric": _numeric,
    "contraction": _contraction,
    "whitespace": _whitespace,
    "tool_xml": _tool_xml,
    "long": _long,
}

_PROBE_SENTENCES = (
    "The window is five hundred thirteen tokens wide. ",
    "第{}層は注意の範囲を制限する。 ",
    "count = {} / 0x{:X}; ",
    "Už asi {} variantů. ",
)


def probe_text(depth):
    """Filler text intended to reach roughly `depth` tokens.

    English-heavy BPE runs ~5 chars/token on this mix; generate 10% extra
    so the actual count lands at or above the nominal depth. The exact
    count comes from /tokenize at dump time.
    """
    rng = random.Random(SEED + depth)
    n_chars = int(depth * 5.0 * 1.10)
    out = []
    size = 0
    i = 0
    while size < n_chars:
        s = _PROBE_SENTENCES[i % len(_PROBE_SENTENCES)].format(i, i * 7919)
        out.append(s)
        size += len(s)
        i += 1
    return "".join(out)


def build():
    rng = random.Random(SEED)
    cases = []
    for cls in COUNTS:
        for i in range(COUNTS[cls]):
            cases.append({"id": "{}-{:04d}".format(cls, i), "class": cls, "text": _GENERATORS[cls](rng)})
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir", type=pathlib.Path)
    ap.add_argument("--probe", action="store_true", help="also write probe-<depth>.txt files")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cases = build()
    path = args.outdir / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    print("{}: {} cases, {:.1f} MB".format(path, len(cases), path.stat().st_size / 1e6))

    if args.probe:
        for depth in PROBE_DEPTHS:
            p = args.outdir / "probe-{}.txt".format(depth)
            p.write_text(probe_text(depth), encoding="utf-8")
            print("{}: ~{} tokens target ({} bytes)".format(p, depth, p.stat().st_size))


if __name__ == "__main__":
    main()
