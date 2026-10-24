#!/usr/bin/env python3
"""Plot Stone Soup resampling tutorial outputs like the Rust WASM viewer (index → state).

Uses the same RNG preamble and resampler sequence as ``export_lot5_resampling.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "verification" / "fixtures" / "lot5_resampling.json"
OUT_DIR = ROOT / "screenshot" / "python"


def replay_tutorial_preamble(n_particles: int) -> None:
    import numpy as np

    _ = np.random.rand(5)
    _ = np.random.uniform(0, 1 / n_particles)
    s_lb = np.arange(n_particles) * (1 / n_particles)
    _ = np.random.uniform(s_lb, s_lb + (1 / n_particles))
    _ = np.random.rand(2)


def run_cases() -> list[dict]:
    import numpy as np
    from stonesoup.resampler.particle import (
        ESSResampler,
        MultinomialResampler,
        ResidualMethod,
        ResidualResampler,
        StratifiedResampler,
        SystematicResampler,
    )
    from stonesoup.types.particle import Particle

    n_particles = 5
    np.random.seed(1991)
    replay_tutorial_preamble(n_particles)
    equal_particles = [Particle(np.array([[float(i)]]), weight=1 / 5) for i in range(n_particles)]
    cases: list[dict] = []

    state = MultinomialResampler().resample(equal_particles)
    cases.append(("multinomial_equal_n5", state))

    state = MultinomialResampler().resample(equal_particles, nparts=10)
    cases.append(("multinomial_upsample_n10", state))

    np.random.seed(1991)
    replay_tutorial_preamble(n_particles)
    state = SystematicResampler().resample(equal_particles)
    cases.append(("systematic_equal_n5", state))

    np.random.seed(1991)
    replay_tutorial_preamble(n_particles)
    state = StratifiedResampler().resample(equal_particles)
    cases.append(("stratified_equal_n5", state))

    weights = np.array([1.0, 5.0, 1.5, 0.5, 2.0])
    norm = weights / weights.sum()
    weighted = [Particle(np.array([[float(i)]]), weight=float(w)) for i, w in enumerate(norm)]
    np.random.seed(1991)
    replay_tutorial_preamble(n_particles)
    state = ResidualResampler(residual_method=ResidualMethod.SYSTEMATIC).resample(weighted)
    cases.append(("residual_systematic_weighted_n5", state))

    np.random.seed(1991)
    replay_tutorial_preamble(n_particles)
    ess = ESSResampler(resampler=ResidualResampler(residual_method=ResidualMethod.SYSTEMATIC))
    state = ess.resample(equal_particles)
    cases.append(("ess_residual_equal_n5", state))

    out: list[dict] = []
    for name, st in cases:
        sv = st.state_vector.T.tolist()
        out.append(
            {
                "name": name,
                "nparts": len(sv),
                "index": list(range(len(sv))),
                "state": [row[0] for row in sv],
            }
        )
    return out


def plot_cases(cases: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    fig.suptitle("Stone Soup ResamplingTutorial — viewer-style (index → state)", fontsize=12)
    for ax, case in zip(axes.flat, cases, strict=True):
        idx = case["index"]
        state = case["state"]
        ax.scatter(idx, state, s=80, c="#78b4ff", edgecolors="#3060a0", linewidths=0.8)
        ax.set_title(case["name"], fontsize=9)
        ax.set_xlabel("index")
        ax.set_ylabel("state")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.35)
        ax.set_xlim(-0.5, max(idx) + 0.5 if idx else 4.5)
        ymin, ymax = min(state), max(state)
        pad = 0.5 if ymax > ymin else 1.0
        ax.set_ylim(ymin - pad, ymax + pad)
    grid_path = OUT_DIR / "tutorial_11_all_cases.png"
    fig.savefig(grid_path, dpi=120)
    plt.close(fig)

    for i, case in enumerate(cases):
        fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
        ax.scatter(case["index"], case["state"], s=100, c="#78b4ff", edgecolors="#3060a0")
        ax.set_title(f"Resampled particles (index → state)\n{case['name']}")
        ax.set_xlabel("index")
        ax.set_ylabel("state")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.35)
        path = OUT_DIR / f"tutorial_11_step_{i + 1}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print("wrote", path)

def compare_to_fixture(cases: list[dict]) -> int:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fx_cases = data["cases"]
    if len(fx_cases) != len(cases):
        print("case count mismatch", len(fx_cases), len(cases), file=sys.stderr)
        return 1
    ok = True
    for i, (live, fx) in enumerate(zip(cases, fx_cases, strict=True)):
        fx_state = [row[0] for row in fx["state_vector"]]
        if live["state"] != fx_state:
            print(f"case {i} state mismatch:\n  got {live['state']}\n  ref {fx_state}", file=sys.stderr)
            ok = False
        if live["nparts"] != fx["nparts"]:
            ok = False
    if ok:
        print("numeric parity with lot5_resampling.json: ok")
    return 0 if ok else 1


def main() -> int:
    cases = run_cases()
    plot_cases(cases)
    return compare_to_fixture(cases)


if __name__ == "__main__":
    raise SystemExit(main())
