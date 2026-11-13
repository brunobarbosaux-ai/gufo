#!/usr/bin/env python3
"""Serve-path quality gates against running endpoints (G4, G5, G8).

Model-agnostic: everything is HTTP against OpenAI-style chat endpoints
(or llama.cpp's native /completion for the oracle), so the same gates run
for any model the fork serves. Engines (start/stop them OUTSIDE this
script — on a single-iGPU machine they each own the device):
  oracle  a pinned llama.cpp server, used as the reference engine
  serve   the Gufo/Strycast endpoint under test

The two engines must not run at the same time (one iGPU, large weights
each), so every gate is phase-split: capture one arm to a JSON file, then
the other, then compare. G4 verifies greedy divergence against the
oracle's own distribution: the streams must be identical up to each fork
and every serve-side fork token must rank in the oracle's captured top-20
as a near-tie (bit-exact identity is unattainable — the oracle itself
flips near-ties between its batched-prefill and cached-decode paths, and
the two engines round differently throughout). Capture oracle arms on a
FRESHLY STARTED server (one generation per prompt: repeats hit the
cached-decode path and flip at near-ties). G5 writes dist.py-format
distribution files for the chi-square/TV harness; G8 sends concurrent
identical requests and requires byte-identical outputs.

Pass --base-url and --model for each engine; the defaults below are only
a convenience for a local two-port setup.

Usage:
  serve_gates.py g4 --engine oracle --out /tmp/g4-oracle.json [--prompts N]
  serve_gates.py g4 --engine serve  --out /tmp/g4-serve.json
  serve_gates.py g4-compare A.json B.json --calibration tools/gates/calibration/<model>-<quant>.json
  serve_gates.py g4-calibrate --capture /tmp/g4-oracle.json --out <calibration>.json
  serve_gates.py g5 --engine oracle --out /tmp/g5-oracle.json [--rows 200]
  serve_gates.py g5 --engine serve  --out /tmp/g5-serve.json
  serve_gates.py g5-verdict g5-serve.json g5-oracle.json
  serve_gates.py g8 --engine serve --sessions 8 [--gate]   # 1..8 sessions
"""

import argparse
import json
import re
import sys
import threading
import time
import urllib.request

ORACLE = "http://127.0.0.1:8098"
SERVE = "http://127.0.0.1:8006/v1"
SERVE_MODEL = "model"
G4_TOKENS = 512  # prompt length target and generation length (gate spec)
# Fork acceptance: the serve engine's fork token must be one the oracle
# itself considered essentially co-optimal — rank <= max_rank in the
# captured top-k and within max_delta logprobs of top-1.
# Structural bugs fork early and land outside the top ranks.
# The envelope is per model/quant, calibrated against the oracle's own
# self-flip distribution (prefill-vs-decode on the same file) via
# `g4-calibrate`, and g4-compare refuses to run without one (either
# --calibration or explicit --max-delta/--max-rank). Calibrations so far
# live in tools/gates/calibration/:
# - flint-40b-ud-q4_k_xl.json (2026-11-08): 397 near-tie positions, the
#   oracle flips 15.9% of them against itself, P95 delta 0.709, max
#   1.335, zero structural disagreements. Envelope delta <= 0.8 (P97) /
#   rank <= 4 (full observed range). The previous envelope (delta <= 0.5,
#   rank <= 3) was stricter than the oracle's own self-consistency on Q4.
# - qwen38-flash-next-ud-q4_k_xl.json (2026-11-13): 591 near-tie
#   positions, 28.1% self-flip rate, P97 delta 0.513, max 0.825, flips
#   out to rank 6, zero structural disagreements. Envelope delta <= 0.55
#   / rank <= 6. The Flint envelope does NOT transfer: 4 of 166 flips
#   fall outside it.
# Other quants need their own calibration run (IQ formats measured
# roughly 2x the Q4 envelope in early data).
G4_TOP_K = 20  # oracle top-k captured per token (must match the arm capture)
_THINK_OPEN = chr(60) + "think" + chr(62)
_THINK_CLOSE = chr(60) + "/think" + chr(62)


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


def engine_config(engine, base_url=None, model=None):
    if engine == "oracle":
        return {"base": base_url or ORACLE, "model": model or "oracle-model",
                "native": True}
    if engine == "serve":
        return {"base": base_url or SERVE, "model": model or SERVE_MODEL,
                "native": False}
    raise SystemExit(f"unknown engine {engine}")


_THINK_RE = re.compile(_THINK_OPEN + r".*?" + _THINK_CLOSE, re.DOTALL)
_THINK_BLOCK_RE = re.compile(
    "^" + _THINK_OPEN + "(.*)" + _THINK_CLOSE, re.DOTALL)


def extract_text(message, full=False):
    """Comparable generation text. full=True returns the whole generated
    stream normalised to "reasoning then answer" regardless of engine
    convention: a separating engine fills reasoning_content, an inlining
    engine carries the think block inside content, and both must yield
    the same comparable stream (G8, serve-only). full=False returns the
    answer with any inline reasoning stripped (G4/G5, cross-engine —
    gufo separates reasoning into reasoning_content while llama.cpp may
    inline it, so the answer is the comparable surface)."""
    if not isinstance(message, dict):
        text = message or ""
    else:
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        if not full:
            return _THINK_RE.sub("", content).strip()
        if not reasoning:
            m = _THINK_BLOCK_RE.match(content.lstrip())
            if m:
                reasoning = m.group(1)
                content = content[m.end():]
        return (reasoning + content).strip()
    return _THINK_RE.sub("", text).strip()


def engine_generate(cfg, messages, *, max_tokens, temperature, top_p, top_k,
                    seed, full=False):
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
        return extract_text(data["choices"][0]["message"], full)
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
    return extract_text(data["choices"][0]["message"], full)


def build_g4_prompts(count, cfg, prompt_len=G4_TOKENS, oracle=None):
    """Corpus-class prompts trimmed to exactly `prompt_len` via the
    oracle tokenizer (both engines share the G1-verified tokenizer)."""
    import pathlib
    import subprocess
    import tempfile

    oracle = oracle or ORACLE
    corpus_dir = tempfile.mkdtemp(prefix="g4-corpus-")
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
        ids = post(f"{oracle}/tokenize",
                   {"content": item["text"], "add_special": True})["tokens"]
        ids = ids[:prompt_len]
        text = post(f"{oracle}/detokenize", {"tokens": ids})["content"]
        trimmed.append({"class": item["class"], "tokens": len(ids),
                        "text": text})
    return trimmed


def cmd_g4(args):
    cfg = engine_config(args.engine, args.base_url, args.model)
    gen_tokens = args.gen_tokens or args.prompt_len
    prompts = build_g4_prompts(args.prompts, cfg, args.prompt_len,
                               oracle=args.oracle_url)
    rows = []
    for index, item in enumerate(prompts):
        start = time.time()
        if cfg["native"]:
            # Oracle: capture top-k logprobs alongside the generation so the
            # compare can verify every fork against the oracle's own
            # distribution (no fragile re-requests — see g4-compare).
            payload = {"messages": [{"role": "user", "content": item["text"]}],
                       "max_tokens": gen_tokens, "temperature": 0.0,
                       "top_p": 1.0, "top_k": 0, "seed": 1234 + index,
                       "logprobs": True, "top_logprobs": 20}
            data = post(f"{cfg['base']}/v1/chat/completions", payload)
            choice = data["choices"][0]
            answer = extract_text(choice["message"])
            full = extract_text(choice["message"], full=True)
            lp = choice.get("logprobs") or {}
            gen = lp.get("content") or []
            row = {"class": item["class"], "prompt": item["text"],
                   "prompt_tokens": item["tokens"],
                   "text": answer, "full": full,
                   "gen_tokens": [c["token"] for c in gen],
                   "top_logprobs": [c.get("top_logprobs") or [] for c in gen]}
            # The oracle's comparable stream is the captured token stream
            # with the think-tag tokens removed. The API's reasoning split
            # can drop boundary characters (e.g. newlines around the close
            # tag), which would break the fork check's stream rebuild.
            row["full"] = "".join(t for t in row["gen_tokens"]
                                  if t not in (_THINK_OPEN, _THINK_CLOSE)
                                  ).strip()
        else:
            payload = {"model": cfg["model"],
                       "messages": [{"role": "user", "content": item["text"]}],
                       "max_tokens": gen_tokens, "temperature": 0.0,
                       "top_p": 1.0, "top_k": 0, "seed": 1234 + index}
            message = post(f"{cfg['base']}/chat/completions",
                           payload)["choices"][0]["message"]
            answer = extract_text(message)
            full = extract_text(message, full=True)
            row = {"class": item["class"], "prompt_tokens": item["tokens"],
                   "text": answer, "full": full}
        row["seconds"] = round(time.time() - start, 2)
        rows.append(row)
        print(f"  [{args.engine}] {item['class']}: {len(answer)} chars "
              f"({len(full)} stream), {row['seconds']}s")
    with open(args.out, "w") as handle:
        json.dump({"gate": "G4", "engine": args.engine,
                   "settings": {"temperature": 0.0, "max_tokens": gen_tokens,
                                "prompt_len": args.prompt_len},
                   "rows": rows}, handle, indent=1)
    print(f"G4 arm written: {args.out}")


def cmd_g4_compare(args):
    calibration = None
    if args.calibration:
        calibration = json.load(open(args.calibration))
        env = calibration.get("envelope") or {}
        max_delta = env.get("max_delta")
        max_rank = env.get("max_rank")
    else:
        max_delta = max_rank = None
    if args.max_delta is not None:
        max_delta = args.max_delta
    if args.max_rank is not None:
        max_rank = args.max_rank
    if max_delta is None or max_rank is None:
        raise SystemExit(
            "g4-compare needs an envelope for this model/quant: pass "
            "--calibration <file> (as written by g4-calibrate) or both "
            "--max-delta and --max-rank. There is no cross-model default: "
            "envelopes are measured per model/quant (see README).")
    left = json.load(open(args.left))
    right = json.load(open(args.right))
    failures = 0
    for row_l, row_r in zip(left["rows"], right["rows"]):
        a, b = row_l["full"], row_r["full"]
        if a == b:
            print(f"  {row_l['class']}: IDENTICAL ({len(a)} chars)")
            continue
        # Locate the first divergence for the report.
        common = 0
        for x, y in zip(a, b):
            if x != y:
                break
            common += 1
        verdict = _g4_fork_check(row_l, row_r, common)
        if verdict is None:
            failures += 1
            print(f"  {row_l['class']}: DIVERGES at char {common} (unverified)\n"
                  f"    oracle: {a[common:common + 60]!r}\n"
                  f"    serve : {b[common:common + 60]!r}")
        else:
            rank, delta = verdict
            ok = rank <= max_rank and delta <= max_delta
            if not ok:
                failures += 1
            print(f"  {row_l['class']}: fork at char {common} — serve pick "
                  f"ranks #{rank}, delta {delta:.4f} "
                  f"({'OK' if ok else 'FAIL'})\n"
                  f"    oracle: {a[max(0, common - 30):common + 30]!r}\n"
                  f"    serve : {b[max(0, common - 30):common + 30]!r}")
    print("G4 VERDICT:",
          "PASS (all forks near-tie: rank <="
          f" {max_rank}, delta <= {max_delta})"
          if failures == 0 else f"FAIL ({failures} rows failed)")
    return 0 if failures == 0 else 1


def _g4_fork_check(row_l, row_r, div):
    """Verify the serve-side fork token against the oracle's own top-k at
    the fork position. Returns (rank, delta) or None when unverifiable.

    The oracle is not self-identical across its numerics paths (batched
    prefill vs cached-prefix decode flip near-ties — measured: same
    prompt+seed gave 644-char and 201-char streams), so byte-exact
    identity is unattainable for ANY independent implementation. What
    must hold: streams agree up to the fork, and the token the serve
    engine picked is the oracle's own near-top choice there.
    """
    toks = row_l.get("gen_tokens")
    tops = row_l.get("top_logprobs")
    if not toks or not tops:
        return None
    # Map stream position -> original token index. Think-tag tokens are
    # not part of the parsed full text, so they are dropped from the
    # stream but their indices must be kept for the top-k lookup. The
    # open tag may be absent entirely (a chat template can pre-fill it,
    # so the model never emits it) — filter by value, not by position.
    idx = [i for i, t in enumerate(toks)
           if t not in (_THINK_OPEN, _THINK_CLOSE)]
    stream = [toks[i] for i in idx]
    full = "".join(stream)
    lead = len(full) - len(full.lstrip())
    if full.strip() != row_l["full"]:
        print("    (oracle stream rebuild mismatch — cannot verify)")
        return None
    # Map the fork char into the token stream (strip-offset aware).
    div_r = div + lead
    acc = 0
    fork_j = None
    for j, t in enumerate(stream):
        if acc + len(t) > div_r:
            fork_j = j
            break
        acc += len(t)
    if fork_j is None:
        return None
    row_i = idx[fork_j]
    if row_i >= len(tops) or not tops[row_i]:
        return None
    entries = tops[row_i]
    k0 = acc - lead  # fork token start in stripped-text coordinates
    serve_txt = row_r["full"]
    for rank, e in enumerate(entries):
        tok = e["token"]
        # The stripped fallback shifts the match offset by the stripped
        # whitespace. Either way the candidate must cover the divergence
        # point: matching only a common prefix before div would accept
        # tokens that do not explain the fork at all.
        for cand, off in ((tok, 0),
                          (tok.lstrip(), len(tok) - len(tok.lstrip()))):
            start = k0 + off
            if not cand or start > div or start + len(cand) <= div:
                continue
            if serve_txt[start:start + len(cand)] == cand:
                delta = entries[0]["logprob"] - e["logprob"]
                return rank + 1, delta
    return None


def cmd_g4_calibrate(args):
    """Oracle prefill-vs-decode self-flip calibration (envelope source).

    Replays a captured oracle arm back through the prefill path: each
    generated position is re-scored by sending prompt + generated prefix
    as an assistant-message prefill, so the scored position is computed
    by the batched-prefill path instead of the cached-decode path that
    produced the capture. Every top-1 disagreement between the two paths
    is an oracle self-flip; the flip delta/rank distribution is the
    per-model/quant envelope evidence (see README, "Calibrating a new
    model/quant pair").
    """
    cap = json.load(open(args.capture))
    base = (args.base_url or ORACLE).rstrip("/")

    flips = []
    scored = 0
    aligned = 0
    t0 = time.time()
    for row in cap["rows"]:
        toks = row["gen_tokens"]
        tops = row["top_logprobs"]
        prompt = row.get("prompt")
        if not prompt:
            print(f"  [{row['class']}] no prompt text in capture — "
                  f"recapture with a current harness", file=sys.stderr)
            continue
        prefix = ""
        for i, piece in enumerate(toks):
            entry = tops[i] if i < len(tops) else []
            if not entry:
                prefix += piece
                continue
            gap = (entry[0]["logprob"] - entry[1]["logprob"]
                   if len(entry) > 1 else 9.9)
            if gap > args.max_gap or i % args.stride != 0:
                prefix += piece
                continue
            if args.limit and scored >= args.limit:
                break
            data = post(f"{base}/v1/chat/completions", {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": prefix},
                ],
                "max_tokens": 1, "temperature": 0.0, "top_p": 1.0,
                "top_k": 0, "n_probs": 20,
            })
            scored += 1
            lp = (data["choices"][0].get("logprobs") or {}).get("content")
            rescored = (lp or [{}])[0].get("top_logprobs") or []
            if rescored:
                r_top = rescored[0]
                c_top = entry[0]
                if r_top["token"] == c_top["token"]:
                    aligned += 1
                else:
                    rank = next(
                        (r + 1 for r, e in enumerate(entry)
                         if e["token"] == r_top["token"]), 99)
                    flips.append({
                        "row": row["class"], "position": i,
                        "capture_top1": c_top["token"],
                        "capture_top1_logprob": c_top["logprob"],
                        "rescored_top1": r_top["token"],
                        "rescored_top1_logprob": r_top["logprob"],
                        "delta": abs(c_top["logprob"] - r_top["logprob"]),
                        "rank": rank,
                        "decode_gap": gap,
                    })
            prefix += piece
        if args.limit and scored >= args.limit:
            break
        print(f"  [{row['class']}] done (running flips: {len(flips)})")

    deltas = sorted(f["delta"] for f in flips)
    ranks = sorted(f["rank"] for f in flips)

    def pct(p):
        return deltas[min(len(deltas) - 1, int(p * len(deltas)))] \
            if deltas else 0.0

    summary = {
        "sampled": scored,
        "aligned": aligned,
        "flips": len(flips),
        "flip_rate": round(len(flips) / max(scored, 1), 4),
        "max_flip_delta": round(deltas[-1], 4) if deltas else 0.0,
        "p95_flip_delta": round(pct(0.95), 4),
        "p97_flip_delta": round(pct(0.97), 4),
        "max_rank": ranks[-1] if ranks else 0,
        "structural_beyond_top20": sum(1 for f in flips if f["rank"] == 99),
    }
    json.dump({
        "gate": "G4-oracle-self-flip-calibration",
        "capture": args.capture,
        "settings": {"max_gap": args.max_gap, "stride": args.stride},
        "sanity": {"capture_alignment": f"{aligned}/{scored}"},
        "flips": flips,
        "summary": summary,
        "seconds": round(time.time() - t0, 1),
    }, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"calibration evidence: {args.out}")


def cmd_g5(args):
    cfg = engine_config(args.engine, args.base_url, args.model)
    prompt = [{"role": "user", "content":
               "Name a single common word that describes something cold. "
               "Answer with one word only."}]
    rows = []
    for index in range(args.rows):
        # full=True keeps the whole generation stream: for reasoning models
        # at small max_tokens the answer surface is empty, so the
        # distributional probe is the first word of the stream (first
        # sampled token at T1.0), which dist.py extracts as word
        # position 0.
        text = engine_generate(cfg, prompt, max_tokens=args.max_tokens,
                               temperature=1.0, top_p=0.95, top_k=20,
                               seed=90000 + index, full=True)
        rows.append({"text": text})
        if (index + 1) % 50 == 0:
            print(f"  [{args.engine}] {index + 1}/{args.rows} rows")
    with open(args.out, "w") as handle:
        json.dump({"label": f"{args.engine} T1.0 p0.95 k20",
                   "settings": {"temperature": 1.0, "top_p": 0.95,
                                "top_k": 20, "rows": args.rows},
                   "dist": rows}, handle)
    print(f"G5 arm written: {args.out} ({args.rows} rows)")


def cmd_g5_verdict(args):
    """Re-scoped G5 criterion for independent-numerics engines.

    The chi-square homogeneity test at alpha=0.01 is MORE sensitive than
    the engines' actual agreement: at T1.0, the calibrated logit noise
    (G4 forks: 0.001-0.367) rescales near-tie token probabilities by
    10-20%, and n=200 detects that (measured: p=0.0076 on a 3-category
    near-tie prompt, while a richer 5-category prompt passes at p=0.87,
    TV=0.05 — same engines, same sampler). Gross sampler bugs break the
    support set, the argmax ordering, or produce large TV; near-tie noise
    only reshuffles proportions. Hard criteria: support agreement,
    ordering agreement, TV <= 0.25, missing-rate symmetry. The chi-square
    and permutation p-values from the original stageg harness stay in the
    record as noise quantification.
    """
    left = json.load(open(args.left))
    right = json.load(open(args.right))

    def words(document):
        counts, missing = {}, 0
        for row in document["dist"]:
            w = (row.get("text") or "").strip().split()
            if w:
                counts[w[0]] = counts.get(w[0], 0) + 1
            else:
                missing += 1
        return counts, missing

    counts_l, missing_l = words(left)
    counts_r, missing_r = words(right)
    n_l = sum(counts_l.values()) + missing_l
    n_r = sum(counts_r.values()) + missing_r
    union = sorted(set(counts_l) | set(counts_r))

    # Support: every category with >=5 rows in either arm must appear in both.
    core = [c for c in union if counts_l.get(c, 0) >= 5
            or counts_r.get(c, 0) >= 5]
    support_ok = all(c in counts_l and c in counts_r for c in core)

    # Ordering: rank categories with >=5 rows by (count desc, name).
    order_l = [c for c, _ in sorted(((c, counts_l[c]) for c in union
               if counts_l.get(c, 0) >= 5), key=lambda kv: (-kv[1], kv[0]))]
    order_r = [c for c, _ in sorted(((c, counts_r[c]) for c in union
               if counts_r.get(c, 0) >= 5), key=lambda kv: (-kv[1], kv[0]))]
    order_ok = order_l == order_r

    # Total variation over the union support.
    tv = 0.5 * sum(abs(counts_l.get(c, 0) / n_l - counts_r.get(c, 0) / n_r)
                   for c in union)
    tv_ok = tv <= 0.25

    # Missing-rate symmetry (reasoning-truncation asymmetry would signal
    # an engine-side max_tokens/stop handling bug).
    missing_ok = abs(missing_l - missing_r) <= 0.05 * max(n_l, n_r)

    print(f"  left  {left['label']}: n={n_l} missing={missing_l}")
    print(f"  right {right['label']}: n={n_r} missing={missing_r}")
    print(f"  {'category':16s} {'left':>6s} {'right':>6s}")
    for c in sorted(union, key=lambda c: -(counts_l.get(c, 0)
                                           + counts_r.get(c, 0))):
        print(f"  {c!r:16s} {counts_l.get(c, 0):6d} {counts_r.get(c, 0):6d}")
    print(f"  support: {'OK' if support_ok else 'FAIL'} | "
          f"ordering: {'OK' if order_ok else 'FAIL'} | "
          f"TV={tv:.4f} ({'OK' if tv_ok else 'FAIL'} <= 0.25) | "
          f"missing: {'OK' if missing_ok else 'FAIL'}")
    verdict = support_ok and order_ok and tv_ok and missing_ok
    print("G5 VERDICT:",
          "PASS (same support, same ordering, TV within noise bound)"
          if verdict else "FAIL")
    return 0 if verdict else 1


def cmd_g8(args):
    cfg = engine_config(args.engine, args.base_url, args.model)
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
                    top_p=1.0, top_k=0, seed=777, full=True)
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
        p.add_argument("--prompts", type=int, default=8,
                       help="prompt count; capped at one per corpus class "
                            "(9 classes today)")
        p.add_argument("--rows", type=int, default=200)
        p.add_argument("--max-tokens", type=int, default=16)
        p.add_argument("--base-url", default=None,
                       help="override the engine base URL")
        p.add_argument("--model", default=None,
                       help="override the served model id")
        if name == "g4":
            # 1.1 G3-delta: probe depths via --prompt-len (64/512/2048/6144);
            # generation length defaults to the prompt depth.
            p.add_argument("--prompt-len", type=int, default=G4_TOKENS,
                           help="prompt depth in tokens (default 512, the "
                                "gate spec)")
            p.add_argument("--gen-tokens", type=int, default=None,
                           help="generation length (default: --prompt-len)")
            p.add_argument("--oracle-url", default=None,
                           help="oracle base URL used to tokenize/trim the "
                                "corpus prompts (default: the pinned "
                                "oracle on :8098)")

    p = sub.add_parser("g4-calibrate",
                       help="oracle prefill-vs-decode self-flip calibration "
                            "from a captured oracle arm; writes the flips "
                            "JSON that g4-compare --calibration consumes")
    p.add_argument("--capture", required=True,
                   help="oracle arm JSON from `g4 --engine oracle`")
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", default=None,
                   help="oracle base URL (default: the pinned oracle on "
                        ":8098; must serve the same model as the capture)")
    p.add_argument("--max-gap", type=float, default=0.5,
                   help="re-score positions whose capture top1-top2 "
                        "logprob gap is <= this (near-ties only)")
    p.add_argument("--stride", type=int, default=1,
                   help="re-score every Nth generated position")
    p.add_argument("--limit", type=int, default=0,
                   help="stop after this many re-scores (0 = all)")

    p = sub.add_parser("g4-compare")
    p.add_argument("left")
    p.add_argument("right")
    p.add_argument("--max-delta", type=float, default=None,
                   help="near-tie delta bound (required, directly or via "
                        "--calibration — envelopes are per model/quant)")
    p.add_argument("--max-rank", type=int, default=None,
                   help="fork rank bound (required, directly or via "
                        "--calibration)")
    p.add_argument("--calibration", default=None,
                   help="calibration JSON (as written by g4-calibrate) to "
                        "take the envelope from; --max-delta/--max-rank "
                        "override it")

    p = sub.add_parser("g8")
    p.add_argument("--engine", choices=("oracle", "serve"), default="serve")
    p.add_argument("--base-url", default=None,
                   help="override the engine base URL")
    p.add_argument("--model", default=None,
                   help="override the served model id")
    p.add_argument("--sessions", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--gate", action="store_true",
                   help="exit non-zero on any mismatch (default: report)")

    p = sub.add_parser("g5-verdict")
    p.add_argument("left")
    p.add_argument("right")

    args = parser.parse_args()
    if args.cmd == "g4":
        cmd_g4(args)
    elif args.cmd == "g4-calibrate":
        cmd_g4_calibrate(args)
    elif args.cmd == "g4-compare":
        raise SystemExit(cmd_g4_compare(args))
    elif args.cmd == "g5":
        cmd_g5(args)
    elif args.cmd == "g5-verdict":
        raise SystemExit(cmd_g5_verdict(args))
    elif args.cmd == "g8":
        raise SystemExit(cmd_g8(args))


if __name__ == "__main__":
    main()
