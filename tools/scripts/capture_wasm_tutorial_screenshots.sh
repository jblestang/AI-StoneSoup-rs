#!/usr/bin/env bash
# Build WASM viewer and capture final-state UI for catalogue 01–27 into screenshot/wasm/.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
export PORT="${PORT:-8765}"
export DIST="$root/apps/stonesoup-viewer/dist"
export OUT="$root/screenshot/wasm"

mkdir -p "$OUT"
cd "$root/apps/stonesoup-viewer"
unset NO_COLOR
trunk build --release

exec python3 "$root/tools/scripts/capture_wasm_tutorial_screenshots.py"
