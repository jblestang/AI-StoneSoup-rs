#!/usr/bin/env python3
"""Compare Stone Soup (Python export) vs Rust reduced GM-PHD mixtures per step."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "verification" / "fixtures" / "lot7_gm_phd.json"
STRONG_W = 0.2


def mean_xy(mean: list[float]) -> tuple[float, float]:
    return mean[0], mean[2]


def run_rust_dump() -> list[dict]:
    proc = subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "stonesoup-scenarios",
            "--test",
            "lot7_rust_dump",
            "--",
            "--nocapture",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"rust dump failed (exit {proc.returncode})")
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("[{"):
            return json.loads(line)
    raise SystemExit("no JSON array found in rust dump output")


def match_dist(py_c: dict, rs_c: dict) -> float:
    px, py = mean_xy(py_c["mean"])
    rx, ry = mean_xy(rs_c["mean"])
    return math.hypot(px - rx, py - ry)


def greedy_match(py_comps: list[dict], rs_comps: list[dict]) -> list[tuple[dict, dict, float]]:
    """Match by smallest position distance (each used once)."""
    pairs: list[tuple[dict, dict, float]] = []
    used_rs: set[int] = set()
    for py_c in sorted(py_comps, key=lambda c: -c["weight"]):
        best_j = None
        best_d = float("inf")
        for j, rs_c in enumerate(rs_comps):
            if j in used_rs:
                continue
            d = match_dist(py_c, rs_c)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is not None:
            used_rs.add(best_j)
            pairs.append((py_c, rs_comps[best_j], best_d))
    return pairs


def main() -> int:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    py_steps = payload["reduced_by_step"]
    rs_steps = run_rust_dump()

    print("Lot 7 GM-PHD — Python (Stone Soup export) vs Rust")
    print(f"fixture: {FIXTURE.relative_to(ROOT)}  seed={payload['seed']}  steps={payload['num_steps']}")
    print()
    hdr = (
        "step",
        "py_n",
        "rs_n",
        "py_wΣ",
        "rs_wΣ",
        "strong_py",
        "strong_rs",
        "max_strong_xy_m",
        "max_strong_Δw",
    )
    print("\t".join(hdr))

    max_xy = 0.0
    count_mismatch = 0
    for step, (py_comps, rs_block) in enumerate(zip(py_steps, rs_steps, strict=True)):
        rs_comps = rs_block["components"]
        py_n = len(py_comps)
        rs_n = rs_block["n"]
        py_ws = sum(c["weight"] for c in py_comps)
        rs_ws = sum(c["weight"] for c in rs_comps)
        if py_n != rs_n:
            count_mismatch += 1
        py_strong = [c for c in py_comps if c["weight"] >= STRONG_W]
        rs_strong = [c for c in rs_comps if c["weight"] >= STRONG_W]
        pairs = greedy_match(py_strong, rs_strong)
        step_max_xy = max((d for _, _, d in pairs), default=0.0)
        step_max_dw = max(
            (abs(p["weight"] - r["weight"]) for p, r, _ in pairs),
            default=0.0,
        )
        max_xy = max(max_xy, step_max_xy)
        print(
            "\t".join(
                str(x)
                for x in (
                    step,
                    py_n,
                    rs_n,
                    f"{py_ws:.6f}",
                    f"{rs_ws:.6f}",
                    len(py_strong),
                    len(rs_strong),
                    f"{step_max_xy:.3f}",
                    f"{step_max_dw:.4f}",
                )
            )
        )

    print()
    print(f"steps with component count mismatch: {count_mismatch}/{len(py_steps)}")
    print(f"max strong-mode (w≥{STRONG_W}) matched position error: {max_xy:.3f} m")
    print()
    print("Step 0 detail (sorted by weight):")
    py0 = sorted(py_steps[0], key=lambda c: -c["weight"])
    rs0 = sorted(rs_steps[0]["components"], key=lambda c: -c["weight"])
    print("  Python:")
    for c in py0[:6]:
        x, y = mean_xy(c["mean"])
        print(f"    w={c['weight']:.6f}  xy=({x:.3f}, {y:.3f})  tag={c['tag']}")
    print("  Rust:")
    for c in rs0[:6]:
        x, y = mean_xy(c["mean"])
        print(f"    w={c['weight']:.6f}  xy=({x:.3f}, {y:.3f})  tag={c['tag']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
