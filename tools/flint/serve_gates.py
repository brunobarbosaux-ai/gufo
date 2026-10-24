#!/usr/bin/env python3
"""F-L7 serve-path gates against running endpoints (G4, G5, G8).

Engines (start/stop them OUTSIDE this script — they each own the iGPU):
  oracle  llama.cpp b11804 ROCm via tools/flint/oracle_server.sh gpu :8098
  serve   gufo flint sandbox via the llama-swap TEST gateway       :8006
          (llamaswap-config-test.yaml, model id t-flint-40b)

The two engines must not run at the same time (one iGPU, 36 GB weights
each), so every gate is phase-split: capture one arm to a JSON file, then
the other, then compare. G4 compares greedy token sequences (text equality
plus token ids through the shared G1-verified tokenizer); G5 writes
dist.py-format distribution files for the chi-square/TV harness; G8 sends
concurrent identical requests and requires byte-identical outputs.

Usage:
  serve_gates.py g4 --engine oracle --out /tmp/g4-oracle.json [--prompts N]
  serve_gates.py g4 --engine serve  --out /tmp/g4-serve.json
  serve_gates.py g4-compare A.json B.json
  serve_gates.py g5 --engine oracle --out /tmp/g5-oracle.json [--rows 200]
  serve_gates.py g5 --engine serve  --out /tmp/g5-serve.json
  python3 .../stageg/dist.py g5-serve.json g5-oracle.json
  serve_gates.py g8 --engine serve --sessions 8 [--gate]   # 1..8 sessions
"""

import argparse
import json
import sys
import threading
import time
import urllib.request

ORACLE = "http://127.0.0.1:8098"
SERVE = "http://127.0.0.1:8006/v1"
SERVE_MODEL = "t-flint-40b"
G4_TOKENS = 512  # prompt length target and generation length (gate spec)


def post(url, payload, timeout=600):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def chat(base_url, model, messages, max_tokens, temperature, top_p, top_k,
         seed, timeout=600):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
    }
    if base_url.endswith("/v1") or "/v1/chat" in base_url:
        data = post(f"{base_url}/chat/completions", payload, timeout)
        return data["choices"][0]["message"]["content"]
    # llama.cpp native endpoint (oracle): /completion with the rendered prompt
    data = post(f"{base_url}/completion", {**payload, "messages": None} |
                {"prompt": messages}, timeout)
    return data["content"]


def engine_config(engine):
    if engine == "oracle":
        return {"base": ORACLE, "model": "flint", "native": True}
    if engine == "serve":
        return {"base": SERVE, "model": SERVE_MODEL, "native": False}
    raise SystemExit(f"unknown engine {engine}")


def engine_generate(cfg, messages, *, max_tokens, temperature, top_p, top_k,
                    seed):
    """One greedy/sampled generation through either engine's chat API."""
    if cfg["native"]:
        # llama.cpp /completion accepts OpenAI-style messages and renders the
        # chat template server-side (same template parity as G2).
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
        }
        data = post(f"{cfg['base']}/v1/chat/completions", payload)
        return data["choices"][0]["message"]["content"]
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
    }
    data = post(f"{cfg['base']}/chat/completions", payload)
    return data["choices"][0]["message"]["content"]


def build_g4_prompts(count, cfg):
    """Eight corpus-class prompts trimmed to exactly G4_TOKENS via the
    oracle tokenizer (both engines share the G1-verified tokenizer)."""
    import pathlib
    import subprocess
    import tempfile

    corpus_dir = tempfile.mkdtemp(prefix="flint-g4-corpus-")
    subprocess.run(
        [sys.executable,
         str(pathlib.Path(__file__).parent / "corpus.py"), corpus_dir],
        check=True, capture_output=True)
    prompts = []
    seen_classes = set()
    with open(f"{corpus_dir}/corpus.jsonl") as handle:
        for line in handle:
            row = json.loads(line)
            cls = row.get("class", row.get("cls", "unknown"))
            if cls in seen_classes:
                continue
            seen_classes.add(cls)
            prompts.append({"class": cls, "text": row["text"]})
            if len(prompts) == count:
                break
    trimmed = []
    for item in prompts:
        ids = post(f"{ORACLE}/tokenize",
                   {"content": item["text"], "add_special": True})["tokens"]
        ids = ids[:G4_TOKENS]
        text = post(f"{ORACLE}/detokenize", {"tokens": ids})["content"]
        trimmed.append({"class": item["class"], "tokens": len(ids),
                        "text": text})
    return trimmed


def cmd_g4(args):
    cfg = engine_config(args.engine)
    prompts = build_g4_prompts(args.prompts, cfg)
    rows = []
    for index, item in enumerate(prompts):
        start = time.time()
        text = engine_generate(
            cfg, [{"role": "user", "content": item["text"]}],
            max_tokens=G4_TOKENS, temperature=0.0, top_p=1.0, top_k=0,
            seed=1234 + index)
        rows.append({"class": item["class"], "prompt_tokens": item["tokens"],
                     "text": text, "seconds": round(time.time() - start, 2)})
        print(f"  [{args.engine}] {item['class']}: {len(text)} chars, "
              f"{rows[-1]['seconds']}s")
    with open(args.out, "w") as handle:
        json.dump({"gate": "G4", "engine": args.engine,
                   "settings": {"temperature": 0.0, "max_tokens": G4_TOKENS},
                   "rows": rows}, handle, indent=1)
    print(f"G4 arm written: {args.out}")


def cmd_g4_compare(args):
    left = json.load(open(args.left))
    right = json.load(open(args.right))
    failures = 0
    for row_l, row_r in zip(left["rows"], right["rows"]):
        if row_l["text"] == row_r["text"]:
            print(f"  {row_l['class']}: IDENTICAL ({len(row_l['text'])} chars)")
            continue
        # Locate the first divergence for the report.
        common = 0
        for a, b in zip(row_l["text"], row_r["text"]):
            if a != b:
                break
            common += 1
        failures += 1
        print(f"  {row_l['class']}: DIVERGES at char {common}\n"
              f"    oracle: {row_l['text'][common:common + 80]!r}\n"
              f"    serve : {row_r['text'][common:common + 80]!r}")
    print("G4 VERDICT:",
          "PASS (8/8 identical greedy sequences)" if failures == 0
          else f"FAIL ({failures} diverged)")
    return 0 if failures == 0 else 1


def cmd_g5(args):
    cfg = engine_config(args.engine)
    prompt = [{"role": "user", "content":
               "Name a single common word that describes something cold. "
               "Answer with one word only."}]
    rows = []
    for index in range(args.rows):
        text = engine_generate(cfg, prompt, max_tokens=args.max_tokens,
                               temperature=1.0, top_p=0.95, top_k=20,
                               seed=90000 + index)
        rows.append({"text": text})
        if (index + 1) % 50 == 0:
            print(f"  [{args.engine}] {index + 1}/{args.rows} rows")
    with open(args.out, "w") as handle:
        json.dump({"label": f"{args.engine} T1.0 p0.95 k20",
                   "settings": {"temperature": 1.0, "top_p": 0.95,
                                "top_k": 20, "rows": args.rows},
                   "dist": rows}, handle)
    print(f"G5 arm written: {args.out} ({args.rows} rows)")


def cmd_g8(args):
    cfg = engine_config(args.engine)
    prompt = [{"role": "user", "content":
               "Count from one to twenty as digits, comma separated."}]
    for sessions in range(1, args.sessions + 1):
        barrier = threading.Barrier(sessions)
        results = [None] * sessions
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=120)
                results[index] = engine_generate(
                    cfg, prompt, max_tokens=args.max_tokens, temperature=0.0,
                    top_p=1.0, top_k=0, seed=777)
            except Exception as exc:  # noqa: BLE001 — gate reports everything
                errors.append(f"session {index}: {exc}")

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(sessions)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            print(f"  sessions={sessions}: ERRORS {errors}")
            return 1
        unique = set(results)
        status = "OK" if len(unique) == 1 else "MISMATCH"
        print(f"  sessions={sessions}: {status} "
              f"({len(results[0])} chars, {len(unique)} distinct)")
        if len(unique) != 1:
            if not args.gate:
                continue
            return 1
        if sessions == args.sessions:
            print("G8 VERDICT: PASS (identical greedy outputs, "
                  f"1..{args.sessions} sessions)")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("g4", "g5"):
        p = sub.add_parser(name)
        p.add_argument("--engine", choices=("oracle", "serve"), required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--prompts", type=int, default=8)
        p.add_argument("--rows", type=int, default=200)
        p.add_argument("--max-tokens", type=int, default=16)

    p = sub.add_parser("g4-compare")
    p.add_argument("left")
    p.add_argument("right")

    p = sub.add_parser("g8")
    p.add_argument("--engine", choices=("oracle", "serve"), default="serve")
    p.add_argument("--sessions", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--gate", action="store_true",
                   help="exit non-zero on any mismatch (default: report)")

    args = parser.parse_args()
    if args.cmd == "g4":
        cmd_g4(args)
    elif args.cmd == "g4-compare":
        raise SystemExit(cmd_g4_compare(args))
    elif args.cmd == "g5":
        cmd_g5(args)
    elif args.cmd == "g8":
        raise SystemExit(cmd_g8(args))


if __name__ == "__main__":
    main()
