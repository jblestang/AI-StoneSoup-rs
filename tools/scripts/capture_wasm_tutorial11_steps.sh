#!/usr/bin/env bash
# Capture WASM viewer PNG after each partial step for tutorial 11 (Lot 5 resampling).
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
export PORT="${PORT:-8767}"
export DIST="$root/apps/stonesoup-viewer/dist"
export OUT="$root/screenshot/wasm/tutorial_11_steps"

mkdir -p "$OUT"
cd "$root/apps/stonesoup-viewer"
unset NO_COLOR
trunk build --release

python3 - <<'PY'
import os
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

port = os.environ["PORT"]
dist = Path(os.environ["DIST"])
out = Path(os.environ["OUT"])
out.mkdir(parents=True, exist_ok=True)
server = subprocess.Popen(
    [sys.executable, "-m", "http.server", port, "--directory", str(dist)],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(1.5)
base = f"http://127.0.0.1:{port}"
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for n in range(1, 7):
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(f"{base}/?tutorial=11&steps={n}", wait_until="networkidle", timeout=120_000)
            page.wait_for_function(
                f"document.title === 'stonesoup-viewer-step-{n}'",
                timeout=120_000,
            )
            page.wait_for_timeout(500)
            dest = out / f"step_{n}.png"
            page.screenshot(path=str(dest), full_page=False)
            print("wrote", dest)
            page.close()
        browser.close()
finally:
    server.terminate()
    server.wait(timeout=10)
PY
