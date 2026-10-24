#!/usr/bin/env bash
# Viewer-style matplotlib PNGs for catalogue tutorials 01–17 (plot panel; matches WASM trails).
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
export OUT="${OUT:-$root/screenshot/python}"
export MPLBACKEND=Agg
exec python3 "$root/tools/python/capture_python_tutorial_screenshots.py"
