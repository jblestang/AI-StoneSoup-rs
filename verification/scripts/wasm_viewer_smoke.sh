#!/usr/bin/env bash
# Headless WASM viewer smoke: partial steps, reset, full finish (requires Trunk dist).
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
dist="${DIST:-apps/stonesoup-viewer/dist}"
if [[ ! -f "$dist/index.html" ]]; then
  echo "wasm smoke: missing $dist/index.html (run trunk build first)"
  exit 1
fi
export DIST="$dist"
exec python3 "$root/verification/scripts/wasm_viewer_smoke.py"
