#!/usr/bin/env python3
"""Verify Rust viewer particle graphics match Stone Soup for all tutorial 11 steps."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "verification" / "fixtures" / "lot5_resampling.json"

# Reuse Stone Soup runner from plot script
sys.path.insert(0, str(ROOT / "tools" / "python"))
from plot_lot5_resampling_reference import run_cases  # noqa: E402


def main() -> int:
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
    live = run_cases()
    if len(live) != len(fx):
        print("FAIL: case count", len(live), "vs fixture", len(fx))
        return 1

    print("Tutorial 11 — index → state (Rust viewer uses same coordinates as below)\n")
    print(f"{'Step':>4}  {'Case':<32}  {'N':>3}  Points (index, state)")
    print("-" * 72)
    all_ok = True
    for step, (case, ref) in enumerate(zip(live, fx, strict=True), start=1):
        ref_pts = [(i, row[0]) for i, row in enumerate(ref["state_vector"])]
        live_pts = list(zip(case["index"], case["state"], strict=True))
        match = live_pts == ref_pts
        all_ok &= match
        status = "ok" if match else "MISMATCH"
        print(f"{step:4d}  {case['name']:<32}  {case['nparts']:3d}  {live_pts}  [{status}]")

    print()
    if all_ok:
        print("Graphical data parity: all 6 steps match Stone Soup and lot5_resampling.json.")
        print("Rust WASM: scatter (index, state) in central panel; sidebar shows resampler name.")
        print("Python reference plots: screenshot/python/tutorial_11_step_*.png")
        print("Rust step captures: screenshot/wasm/tutorial_11_steps/step_*.png")
        return 0
    print("FAIL: at least one step diverged from fixture.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
