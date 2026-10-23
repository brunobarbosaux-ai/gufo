#!/usr/bin/env python3
"""Oracle goldens for tokenizer (G1 / M0.6) and chat template (G2 / M0.7).

Captures llama.cpp b11804 behaviour against the pinned Flint artifact:

  tokenize  /tokenize for every corpus case, add_special true and false
  template  /apply-template for the G2 conversation fixtures, then /tokenize
            of the rendered prompt (byte-identical template parity is judged
            on prompt token ids, so the golden stores both)

Goldens are artifacts and stay outside Git (see tools/ds4/README.md);
the manifest records sha256 + oracle build + model id. Regenerate with
the same pinned oracle_server.sh build whenever the oracle moves.

Usage:
  python3 tools/flint/golden.py tokenize  OUTDIR [--base-url URL] [--limit N]
  python3 tools/flint/golden.py template  OUTDIR [--base-url URL]
  python3 tools/flint/golden.py compare   A.json B.json
"""

import argparse
import hashlib
import json
import pathlib
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
PORT = 8098
BASE = "http://127.0.0.1:{}".format(PORT)
HEALTH_TIMEOUT_S = 300  # first mmap of the 34 GiB artifact is the slow part


def _post(url, payload, timeout=120):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError("POST {} -> {}: {}".format(url, e.code, body)) from None


def _get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class Oracle:
    """Running oracle server, started on demand unless --base-url was given."""

    def __init__(self, base_url=None, gpu=False):
        self.external = base_url is not None
        self.gpu = gpu
        self.base = base_url or BASE
        self.proc = None
        self.props = None

    def __enter__(self):
        if not self.external:
            self._spawn()
        self._wait_health()
        self.props = _get(self.base + "/props")
        return self

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _spawn(self):
        if shutil.which("curl") is None:
            raise RuntimeError("curl not found; needed for the health check")
        mode = "gpu" if self.gpu else "cpu"
        self.proc = subprocess.Popen(
            [str(HERE / "oracle_server.sh"), mode],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)

    def _wait_health(self):
        timeout_s = 10 if self.external else HEALTH_TIMEOUT_S
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError("oracle server exited during startup")
            try:
                _get(self.base + "/health", timeout=5)
                return
            except (urllib.error.URLError, OSError):
                time.sleep(2)
        raise RuntimeError("oracle server not healthy after {}s".format(timeout_s))

    def tokenize(self, content, add_special):
        return _post(self.base + "/tokenize",
                     {"content": content, "add_special": add_special})["tokens"]

    def apply_template(self, messages, **extra):
        payload = {"messages": messages}
        payload.update(extra)
        return _post(self.base + "/apply-template", payload)["prompt"]


def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(outdir, name, extra):
    manifest_path = outdir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[name] = extra
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- tokenize

def cmd_tokenize(args):
    outdir = args.outdir.resolve()
    corpus_path = outdir / "corpus.jsonl"
    if not corpus_path.exists():
        sys.exit("missing {} — run: python3 tools/flint/corpus.py {}".format(corpus_path, outdir))

    cases = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    if args.limit:
        cases = cases[:args.limit]

    golden = {"format": 1, "cases": []}
    with Oracle(args.base_url, gpu=False) as oracle:
        model_path = oracle.props.get("model_path", "unknown")
        print("oracle: model_path={}".format(model_path))
        for i, case in enumerate(cases):
            entry = {
                "id": case["id"],
                "class": case["class"],
                "special_true": oracle.tokenize(case["text"], True),
                "special_false": oracle.tokenize(case["text"], False),
            }
            golden["cases"].append(entry)
            if (i + 1) % 500 == 0:
                print("  {}/{}".format(i + 1, len(cases)))

    path = outdir / "tokenize_golden.json"
    path.write_text(json.dumps(golden, ensure_ascii=False), encoding="utf-8")
    _write_manifest(outdir, "tokenize_golden", {
        "file": path.name, "sha256": _sha(path), "cases": len(golden["cases"]),
    })
    print("{}: {} cases".format(path, len(golden["cases"])))


# ---------------------------------------------------------------- template

# G2 fixtures. Byte parity is judged on rendered prompt token ids; these
# conversations exist to expose every branch of the template: roles,
# reasoning, tools, observation, escaping, malformed calls.
FIXTURES = [
    ("basic_system_user_assistant", {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 17*23?"},
            {"role": "assistant", "content": "391."},
            {"role": "user", "content": "Thanks!"},
        ]}),
    ("developer_role", {
        "messages": [
            {"role": "system", "content": "System line."},
            {"role": "developer", "content": "Follow the style guide."},
            {"role": "user", "content": "Hello"},
        ]}),
    ("reasoning_turn", {
        "messages": [
            {"role": "user", "content": "Plan a route."},
            {"role": "assistant", "content": "Done.",
             "reasoning_content": "First check distances, then pick the shortest."},
        ]}),
    ("tool_roundtrip", {
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Get weather for a town",
            "parameters": {"type": "object", "properties": {
                "town": {"type": "string"}, "unit": {"type": "string"}},
                "required": ["town"]}}}],
        "messages": [
            {"role": "system", "content": "Use tools when helpful."},
            {"role": "user", "content": "Weather in Curitiba?"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"type": "function", "function": {
                 "name": "get_weather",
                 "arguments": "{\"town\": \"Curitiba\", \"unit\": \"c\"}"}}]},
            {"role": "tool", "content": "{\"temp_c\": 21, \"sky\": \"clear\"}"},
        ]}),
    ("tool_nested_unicode_args", {
        "tools": [{"type": "function", "function": {
            "name": "run_query",
            "description": "Run a query",
            "parameters": {"type": "object", "properties": {
                "filter": {"type": "object"}, "tag": {"type": "string"}}}}}],
        "messages": [
            {"role": "user", "content": "Query it"},
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {
                "name": "run_query",
                "arguments": "{\"filter\": {\"k\": [1, 2, {\"deep\": \"名前\"}]}, \"tag\": \"🚀\"}"}}]},
            {"role": "tool", "content": "{\"rows\": 3}"},
        ]}),
    ("tool_empty_args", {
        "tools": [{"type": "function", "function": {
            "name": "ping", "description": "Ping",
            "parameters": {"type": "object", "properties": {}}}}],
        "messages": [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {
                "name": "ping", "arguments": "{}"}}]},
            {"role": "tool", "content": "{\"pong\": true}"},
        ]}),
    ("malformed_tool_call_in_content", {
        "messages": [
            {"role": "user", "content": "try it"},
            {"role": "assistant",
             "content": "<get_weather broken=\"unterminated"},
        ]}),
    ("marker_literals_in_user_text", {
        "messages": [
            {"role": "user",
             "content": "Explain  tags. Also this line contains  literally."},
        ]}),
    ("empty_user_message", {
        "messages": [
            {"role": "user", "content": ""},
        ]}),
    ("system_only", {
        "messages": [
            {"role": "system", "content": "Only a system prompt."},
        ]}),
    ("multiline_system", {
        "messages": [
            {"role": "system", "content": "Line one\nLine two\n\nLine four\ttabbed"},
            {"role": "user", "content": "ok"},
        ]}),
    ("non_ascii_tool_names", {
        "tools": [{"type": "function", "function": {
            "name": "get_weather_cité",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"ville": {"type": "string"}}}}}],
        "messages": [
            {"role": "user", "content": "Météo à Curitiba?"},
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {
                "name": "get_weather_cité", "arguments": "{\"ville\": \"Curitiba\"}"}}]},
            {"role": "user", "content": "Merci"},
        ]}),
]


def cmd_template(args):
    outdir = args.outdir.resolve()
    golden = {"format": 1, "fixtures": []}
    with Oracle(args.base_url, gpu=False) as oracle:
        # /apply-template must exist on the pinned build; fail loudly here
        # rather than silently degrading to a weaker golden.
        probe = oracle.apply_template([{"role": "user", "content": "probe"}])
        if not isinstance(probe, str):
            sys.exit("/apply-template returned unexpected payload: {!r}".format(probe))
        for name, payload in FIXTURES:
            rendered = oracle.apply_template(**payload)
            entry = {
                "name": name,
                "rendered": rendered,
                "tokens": oracle.tokenize(rendered, False),
            }
            golden["fixtures"].append(entry)
            print("  {}: {} chars, {} tokens".format(name, len(rendered), len(entry["tokens"])))

    path = outdir / "template_golden.json"
    path.write_text(json.dumps(golden, ensure_ascii=False), encoding="utf-8")
    _write_manifest(outdir, "template_golden", {
        "file": path.name, "sha256": _sha(path), "fixtures": len(golden["fixtures"]),
    })
    print("{}: {} fixtures".format(path, len(golden["fixtures"])))


# ---------------------------------------------------------------- compare

def cmd_compare(args):
    a = json.loads(pathlib.Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(pathlib.Path(args.b).read_text(encoding="utf-8"))
    if a.get("format") != b.get("format") or "cases" not in a:
        sys.exit("compare expects two tokenize goldens")
    idx = {c["id"]: c for c in b["cases"]}
    diverged = 0
    first = None
    for case in a["cases"]:
        other = idx.get(case["id"])
        if other is None:
            continue
        for key in ("special_true", "special_false"):
            if case[key] != other[key]:
                diverged += 1
                if first is None:
                    first = (case["id"], key, case[key][:16], other[key][:16])
    if diverged:
        print("DIVERGED: {} case-keys differ".format(diverged))
        cid, key, left, right = first
        print("first: {} {}  A={}  B={}".format(cid, key, left, right))
        sys.exit(1)
    print("IDENTICAL over {} cases".format(len(a["cases"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tokenize", help="capture /tokenize goldens for the corpus")
    p.add_argument("outdir", type=pathlib.Path)
    p.add_argument("--base-url", default=None, help="use a running oracle instead of spawning one")
    p.add_argument("--limit", type=int, default=0, help="only the first N cases (smoke)")
    p.set_defaults(func=cmd_tokenize)

    p = sub.add_parser("template", help="capture /apply-template goldens for the G2 fixtures")
    p.add_argument("outdir", type=pathlib.Path)
    p.add_argument("--base-url", default=None)
    p.set_defaults(func=cmd_template)

    p = sub.add_parser("compare", help="compare two tokenize goldens")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
