#!/usr/bin/env python3
"""Capture WASM viewer final-state PNGs for tutorials 01–08."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

READY_TITLE = "stonesoup-viewer-ready"


def main() -> int:
    port = os.environ.get("PORT", "8765")
    dist = Path(os.environ.get("DIST", "apps/stonesoup-viewer/dist"))
    out = Path(os.environ.get("OUT", "screenshot/wasm"))
    out.mkdir(parents=True, exist_ok=True)

    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", port, "--directory", str(dist)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.5)
    try:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "playwright", "-q"]
            )
            subprocess.check_call(
                [sys.executable, "-m", "playwright", "install", "chromium"]
            )
            from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            for code in ("01", "02", "03", "04", "05", "06", "07", "08"):
                page = browser.new_page(viewport={"width": 1280, "height": 800})
                url = f"http://127.0.0.1:{port}/?tutorial={code}&finish=1"
                page.goto(url, wait_until="networkidle", timeout=120_000)
                page.wait_for_function(
                    f"document.title === '{READY_TITLE}'",
                    timeout=120_000,
                )
                page.wait_for_timeout(800)
                dest = out / f"tutorial_{code}.png"
                page.screenshot(path=str(dest), full_page=False)
                print("wrote", dest.resolve())
                page.close()
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
