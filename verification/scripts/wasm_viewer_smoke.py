#!/usr/bin/env python3
"""Playwright smoke test for the WASM viewer (step, reset, finish)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

READY_TITLE = "stonesoup-viewer-ready"
RESET_TITLE = "stonesoup-viewer-reset"
STEP3_TITLE = "stonesoup-viewer-step-3"
ROOT = Path(__file__).resolve().parents[2]


def ensure_playwright() -> None:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright", "-q"])
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])


def main() -> int:
    ensure_playwright()
    from playwright.sync_api import sync_playwright

    port = os.environ.get("PORT", "8766")
    dist = Path(os.environ.get("DIST", ROOT / "apps/stonesoup-viewer/dist"))
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
            page = browser.new_page(viewport={"width": 1280, "height": 800})

            page.goto(f"{base}/?tutorial=01&steps=3", wait_until="networkidle", timeout=120_000)
            page.wait_for_function(
                f"document.title === '{STEP3_TITLE}'",
                timeout=120_000,
            )

            page.goto(
                f"{base}/?tutorial=01&reset=1",
                wait_until="networkidle",
                timeout=120_000,
            )
            page.wait_for_function(
                f"document.title === '{RESET_TITLE}'",
                timeout=30_000,
            )

            page.goto(f"{base}/?tutorial=01&finish=1", wait_until="networkidle", timeout=120_000)
            page.wait_for_function(
                f"document.title === '{READY_TITLE}'",
                timeout=120_000,
            )

            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    print("wasm_viewer_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
