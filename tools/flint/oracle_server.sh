#!/usr/bin/env bash
# Pinned Flint oracle server — llama.cpp b11804 ROCm build.
#
# The oracle for every Flint parity gate (G1–G4). Pin the build, the model and
# the flags here; never derive goldens from an unpinned binary.
#
# Modes:
#   cpu (default) — --device none, no GPU touched. Sufficient for tokenize and
#                   template goldens (M0.6, M0.7) and small logit smokes.
#   gpu           — full offload for logit dumps at depth (M0.5) and G4 identity.
#
# Port 8098: 8096 is gufoperf, 8097 is gufo-ornith, 8005 is llama-swap.

set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-/home/bruno/AI/llama/llama-b11804-rocm}"
MODEL="${MODEL:-/home/bruno/AI/GGUF/flint-UD-Q6_K_XL.gguf}"
PORT="${PORT:-8098}"
HOST="${HOST:-127.0.0.1}"
MODE="${1:-cpu}"
CTX_CPU="${CTX_CPU:-8192}"
CTX_GPU="${CTX_GPU:-32768}"
THREADS="${THREADS:-$(nproc)}"

case "$MODE" in
  cpu)
    FLAGS=(--device none -ngl 0 -c "$CTX_CPU" -t "$THREADS" -tb "$THREADS")
    ;;
  gpu)
    FLAGS=(--mmap mmap -ngl 99 -fa off -c "$CTX_GPU")
    ;;
  *)
    echo "usage: $0 [cpu|gpu]" >&2; exit 2 ;;
esac

echo "oracle: llama.cpp $( "$LLAMA_DIR/llama-cli" --version 2>&1 | head -1 )"
if [[ "${VERIFY_SHA:-0}" == "1" ]]; then
  echo "oracle: model ${MODEL##*/} sha256 $(sha256sum "$MODEL" | cut -c1-16)… (may take ~1 min)"
else
  echo "oracle: model ${MODEL##*/} (set VERIFY_SHA=1 to hash the 34 GiB artifact)"
fi
echo "oracle: mode=$MODE host=$HOST port=$PORT"

exec "$LLAMA_DIR/llama-server" \
  --model "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --jinja \
  --no-webui \
  "${FLAGS[@]}"
