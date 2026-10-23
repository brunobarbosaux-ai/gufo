#!/usr/bin/env python3
"""Pack the G2 template fixtures plus their oracle renderings into a binary.

The fixture definitions live in golden.py (single source of truth); the rendered
text and token ids come from template_golden.json. This joins them and refuses
to write when the two drift apart.

Usage:
  python3 tools/flint/g2_cases.py template_golden.json OUT.bin
"""

import json
import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import golden  # noqa: E402

ROLE = {"system": 0, "developer": 1, "user": 2, "assistant": 3, "tool": 4}


def put_string(buf, text):
    data = text.encode("utf-8")
    buf += struct.pack("<I", len(data))
    buf += data


def pack_tool(buf, tool):
    fn = tool.get("function", tool)
    put_string(buf, fn.get("name", ""))
    put_string(buf, fn.get("description", ""))
    params = fn.get("parameters", {})
    put_string(buf, json.dumps(params, ensure_ascii=False))


def pack_message(buf, msg):
    buf += struct.pack("<B", ROLE[msg["role"]])
    put_string(buf, msg.get("content") or "")
    put_string(buf, msg.get("reasoning_content") or "")
    calls = msg.get("tool_calls") or []
    buf += struct.pack("<I", len(calls))
    for call in calls:
        fn = call.get("function", call)
        put_string(buf, fn.get("name", ""))
        args = fn.get("arguments", "{}")
        if isinstance(args, str):
            args = json.loads(args) if args else {}
        buf += struct.pack("<I", len(args))
        for key, value in args.items():
            put_string(buf, key)
            if isinstance(value, str):
                buf += struct.pack("<B", 1)
                put_string(buf, value)
            else:
                buf += struct.pack("<B", 0)
                put_string(buf, json.dumps(value, ensure_ascii=False))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    golden_path = pathlib.Path(sys.argv[1])
    out_path = pathlib.Path(sys.argv[2])

    goldens = json.loads(golden_path.read_text(encoding="utf-8"))["fixtures"]
    if len(goldens) != len(golden.FIXTURES):
        sys.exit("fixture count mismatch: {} vs {}".format(
            len(goldens), len(golden.FIXTURES)))

    payload = bytearray()
    payload += struct.pack("<I", len(goldens))
    for (name, fixture), golden_entry in zip(golden.FIXTURES, goldens):
        if golden_entry["name"] != name:
            sys.exit("fixture name mismatch: {} vs {}".format(
                golden_entry["name"], name))
        put_string(payload, name)
        put_string(payload, golden_entry["rendered"])
        tokens = golden_entry["tokens"]
        payload += struct.pack("<I", len(tokens))
        payload += struct.pack("<{}I".format(len(tokens)), *tokens)
        tools = fixture.get("tools") or []
        payload += struct.pack("<I", len(tools))
        for tool in tools:
            pack_tool(payload, tool)
        messages = fixture["messages"]
        payload += struct.pack("<I", len(messages))
        for msg in messages:
            pack_message(payload, msg)

    out_path.write_bytes(payload)
    print("{} fixtures, {:.1f} KiB -> {}".format(
        len(goldens), len(payload) / 1024.0, out_path))


if __name__ == "__main__":
    main()
