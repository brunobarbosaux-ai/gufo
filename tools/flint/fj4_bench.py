#!/usr/bin/env python3
"""F-J4 benchmark client (0.1): Gufo flint vs the nemotron llama.cpp table.

Sends the three committed F-J4 prompts (msg / doc / long) N times each to
a running gufo serve endpoint at T = 0, streaming, and derives per run:

- prefill avg  = prompt_tokens / TTFT (first streamed token)
- prefill marginal = (TTFT_long - TTFT_msg) / (tokens_long - tokens_msg)
  — the per-token cost of prompt growth, fixed overheads cancelled
- decode = (completion_tokens - 1) / (T_last - T_first_token)

Also samples SoC temperature and current sclk around every run (0.2
protocol: every number carries its thermal context; cool vs sustained
are reported separately).

The comparison anchors live in nemotron35-llamacpp-bench.md (Flint ROCm,
UD-Q6_K_XL, b11804): 526.4 / 1444.4 / 1131.3 t/s prefill and
48.2 / 44.5 / 38.8 t/s decode for msg / doc / long.

Usage:
  python3 tools/flint/fj4_bench.py [--url URL] [--runs N] [--max-tokens N]
The endpoint must already serve the flint model (start it separately —
this script never owns the iGPU).
"""

import argparse
import glob
import json
import pathlib
import time
import urllib.request

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "fj4"
PROMPTS = (("msg", "msg.txt"), ("doc", "doc.txt"), ("long", "long.txt"))


def read_sys(path_glob: str) -> str:
    hits = glob.glob(path_glob)
    if not hits:
        return "?"
    try:
        return (pathlib.Path(hits[0]).read_text()).strip()
    except OSError:
        return "?"


def thermal() -> str:
    temp = read_sys("/sys/class/drm/card1/device/hwmon/hwmon*/temp1_input")
    try:
        temp_c = int(temp) / 1000.0
    except ValueError:
        temp_c = -1.0
    sclk = read_sys("/sys/class/drm/card1/device/pp_dpm_sclk")
    cur = "?"
    for line in sclk.splitlines():
        if "*" in line:
            mhz = line.split(":")[1].strip().split("Mhz")[0].strip()
            cur = mhz
    return f"SoC {temp_c:.1f}C sclk {cur}MHz"


def stream_once(url: str, model: str, text: str, max_tokens: int):
    payload = {
        "model": model, "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
        "stream": True, "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first_token_at = None
    usage = None
    with urllib.request.urlopen(request, timeout=1200) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            if first_token_at is None:
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if (delta.get("content") or delta.get("reasoning_content")
                            or delta.get("tool_calls")):
                        first_token_at = time.perf_counter()
                        break
    t1 = time.perf_counter()
    return t0, first_token_at, t1, usage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8006/v1/chat/completions")
    ap.add_argument("--model", default="t-flint-40b")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompts", default="msg,doc,long",
                    help="comma-separated subset of msg/doc/long")
    args = ap.parse_args()

    wanted = [s.strip() for s in args.prompts.split(",") if s.strip()]
    prompts = [(label, (FIXTURES / name).read_text())
               for label, name in PROMPTS if label in wanted]

    # Warm the engine (weight pages + code) with the short prompt once.
    stream_once(args.url, args.model, prompts[0][1], 8)

    rows = {}
    ttfts = {label: [] for label, _ in PROMPTS}
    for label, text in prompts:
        per_run = []
        for run in range(args.runs):
            before = thermal()
            t0, tft, t1, usage = stream_once(args.url, args.model, text,
                                             args.max_tokens)
            after = thermal()
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
            ttft_ms = (tft - t0) * 1e3
            gen_ms = (t1 - tft) * 1e3
            prefill = prompt_tokens / (ttft_ms / 1e3)
            decode = (completion_tokens - 1) / (gen_ms / 1e3)
            per_run.append({"prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "ttft_ms": ttft_ms, "gen_ms": gen_ms,
                            "prefill": prefill, "decode": decode,
                            "thermal": before})
            ttfts[label].append(ttft_ms)
            print(f"  [{label} run {run + 1}] {prompt_tokens} tok prompt, "
                  f"{completion_tokens} gen: prefill {prefill:.1f} t/s, "
                  f"decode {decode:.1f} t/s ({before})")
        rows[label] = per_run

    print("\n=== F-J4 summary (mean of", args.runs, "runs, T=0) ===")
    print(f"{'prompt':>6} {'tokens':>7} {'prefill avg':>12} "
          f"{'decode':>8}")
    means = {}
    for label, _ in prompts:
        p = [r["prompt_tokens"] for r in rows[label]]
        pr = sum(r["prefill"] for r in rows[label]) / len(rows[label])
        de = sum(r["decode"] for r in rows[label]) / len(rows[label])
        means[label] = (sum(p) / len(p), pr, de)
        print(f"{label:>6} {sum(p) / len(p):>7.0f} {pr:>10.1f}  {de:>8.1f}")
    fixed = sum(ttfts["msg"]) / len(ttfts["msg"]) if ttfts["msg"] else 0.0
    if ttfts["long"] and ttfts["msg"] and means.get("long") and means.get("msg"):
        long_ttft = sum(ttfts["long"]) / len(ttfts["long"])
        d_tok = means["long"][0] - means["msg"][0]
        if d_tok > 0:
            marginal = (long_ttft - fixed) / d_tok * 1e3
            print(f"prefill marginal (long vs msg): ~{marginal:.1f} t/s")


if __name__ == "__main__":
    main()
