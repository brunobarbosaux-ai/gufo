#!/usr/bin/env python3
"""Pack the G1 corpus + llama.cpp golden ids into a binary case file.

The golden stores ids only (texts are regenerated bit-identically by
corpus.py's fixed seed), so this joins them by case id and refuses to write a
file when the two disagree on any case.

Usage:
  python3 tools/flint/g1_cases.py GOLDEN.json OUT.bin
"""

import json
import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import corpus  # noqa: E402


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    golden_path = pathlib.Path(sys.argv[1])
    out_path = pathlib.Path(sys.argv[2])

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    cases = golden["cases"]
    generated = corpus.build()
    if len(generated) != len(cases):
        sys.exit("case count mismatch: corpus {} vs golden {}".format(
            len(generated), len(cases)))

    payload = bytearray()
    payload += struct.pack("<I", len(cases))
    for gen, case in zip(generated, cases):
        if gen["id"] != case["id"]:
            sys.exit("case id mismatch: {} vs {}".format(gen["id"], case["id"]))
        text = gen["text"].encode("utf-8")
        ids = case["special_true"]
        payload += struct.pack("<I", len(text)) + text
        payload += struct.pack("<I", len(ids))
        payload += struct.pack("<{}I".format(len(ids)), *ids)

    out_path.write_bytes(payload)
    print("{} cases, {:.1f} KiB -> {}".format(
        len(cases), len(payload) / 1024.0, out_path))


if __name__ == "__main__":
    main()
