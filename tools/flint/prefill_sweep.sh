#!/usr/bin/env bash
# GPU-window sweep for Flint prefill (optimization-plan items 2.5 + 2.7 A/B).
#
# Runs the real-artifact chunk smoke across the chunk-size sweep, once per
# MoE route and once with the scalar router, so the router/attention deltas
# are attributable. Each run prints the per-stage breakdown (0.3) and the
# G6 split bit-exactness line; capture everything to a log for BENCHMARKS.md.
#
# Requires the iGPU window and the 36 GB artifact. Cool vs sustained is a
# separate protocol (0.2) — this is the cool/burst sweep only.
#
# Usage:  GUFO_FLINT_GGUF=/home/bruno/AI/GGUF/flint-UD-Q6_K_XL.gguf \
#         tools/flint/prefill_sweep.sh [ctx]
#         (ctx = arena max_context; default 32768 to match the serve path)
set -euo pipefail

BIN="${BIN:-build/gpu-test/tests/models/qwen/qwen_flint_real_smoke_test}"
CTX="${1:-32768}"
OUT="${OUT:-/tmp/flint-prefill-sweep-$(date +%Y%m%d-%H%M%S).log}"

: "${GUFO_FLINT_GGUF:?set GUFO_FLINT_GGUF to the flint artifact}"
[[ -x "$BIN" ]] || { echo "no test binary at $BIN (ninja qwen_flint_real_smoke_test)" >&2; exit 1; }

run() {  # run <label> <n_tokens> [env assignments...]
  local label="$1" n="$2"; shift 2
  echo "===== $label  n=$n  ctx=$CTX  env: $* =====" | tee -a "$OUT"
  # Continue past gate failures: a failed parity gate must not kill the
  # remaining A/B runs — the isolation data is the point of the sweep.
  env "$@" "$BIN" chunk "$CTX" "$n" 2>&1 | tee -a "$OUT" || true
  echo | tee -a "$OUT"
}

echo "flint prefill sweep -> $OUT (artifact $GUFO_FLINT_GGUF)"
# Fast device gate first (seconds, no big load) so a bad build fails before the
# 36 GB artifact load.
echo "===== synthetic gate =====" | tee -a "$OUT"
build/gpu-test/tests/models/qwen/qwen_flint_moe_wmma_ops_test 2>&1 | tee -a "$OUT"
echo | tee -a "$OUT"

# Diagnostic first: scalar router at 181 isolates the tiled-router kernel
# (the W2 sweep run that failed the head gate); 512 with the tiled router
# isolates the chunk size. Then the remaining sweep cells.
run "wmma+SCALAR-router" 181 GUFO_FLINT_ROUTER_SCALAR=1
run "wmma+tiled-router" 512
run "wmma+tiled-router" 256
run "wmma+SCALAR-router" 512 GUFO_FLINT_ROUTER_SCALAR=1
# Isolation: both routes scalar (W1-era kernels + the PV unroll only).
run "SCALAR-moe+SCALAR-router" 181 GUFO_FLINT_MOE_SCALAR=1 GUFO_FLINT_ROUTER_SCALAR=1
# MoE route A/B (2.2 re-check at the small end): scalar grouped MoE.
run "SCALAR-moe+tiled-router" 181 GUFO_FLINT_MOE_SCALAR=1

echo "sweep complete: $OUT"
