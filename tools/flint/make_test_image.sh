#!/usr/bin/env bash
# Build the flint sandbox test image: the freshly built gufo binary on the
# ROCm 10.1 gfx1151 test base (gufo/test:flint-f-l4, itself build-deps +
# runtime libs + ENTRYPOINT gufo / CMD serve llm).
#
# Usage: tools/flint/make_test_image.sh [build-dir] [tag]
#   build-dir defaults to build/gpu-test, tag to gufo/test:flint-<short-sha>.
set -euo pipefail

BUILD_DIR="${1:-build/gpu-test}"
TAG="${2:-gufo/test:flint-$(git rev-parse --short HEAD)}"
BIN="$BUILD_DIR/gufo"
BASE="gufo/test:flint-f-l4"

if [[ ! -f "$BIN" ]]; then
  echo "no binary at $BIN (run: ninja -C $BUILD_DIR gufo)" >&2
  exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp "$BIN" "$STAGE/gufo"
printf 'FROM %s\nCOPY gufo /usr/local/bin/gufo\n' "$BASE" > "$STAGE/Dockerfile"

docker build -q -t "$TAG" "$STAGE"
echo "built $TAG from $BIN"
