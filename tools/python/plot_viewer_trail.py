#!/usr/bin/env python3
"""Plot a dump_viewer_trail JSON payload like the Rust WASM viewer (plot panel only)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


def ellipse_ring(
    center: list[float], semi: tuple[float, float], angle_rad: float, segments: int = 48
) -> list[tuple[float, float]]:
    a, b = semi
    cos_t, sin_t = math.cos(angle_rad), math.sin(angle_rad)
    cx, cy = center
    ring: list[tuple[float, float]] = []
    for i in range(segments):
        phi = 2.0 * math.pi * i / segments
        lx, ly = a * math.cos(phi), b * math.sin(phi)
        ring.append((cx + lx * cos_t - ly * sin_t, cy + lx * sin_t + ly * cos_t))
    return ring


def plot_trail(payload: dict[str, Any], dest: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    dest.parent.mkdir(parents=True, exist_ok=True)

    if payload.get("is_lot5"):
        fig, ax = plt.subplots(figsize=(12.8, 8.0), constrained_layout=True)
        ax.set_title("Resampled particles (index → state)")
        pts = payload.get("resample_plot") or []
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(xs, ys, s=80, c="#78b4ff", edgecolors="#3060a0", linewidths=0.8, label="Particles")
        ax.set_xlabel("index")
        ax.set_ylabel("state")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper right")
        fig.savefig(dest, dpi=100)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(12.8, 8.0), constrained_layout=True)
    ax.set_title("Tracking view (x–y)")

    sensor = payload.get("sensor_xy")
    if sensor:
        ax.scatter(
            [sensor[0]],
            [sensor[1]],
            s=80,
            c="yellow",
            edgecolors="black",
            linewidths=0.5,
            label="Sensor",
            zorder=5,
        )

    truth = payload.get("truth_trail") or []
    if truth:
        ax.plot(
            [p[0] for p in truth],
            [p[1] for p in truth],
            color="#d3d3d3",
            linewidth=2,
            label="Ground truth",
        )

    is_lot3_meas = payload.get("is_lot3_meas")
    clutter = payload.get("step_clutter") or []
    if is_lot3_meas and clutter:
        ax.scatter(
            [p[0] for p in clutter],
            [p[1] for p in clutter],
            s=25,
            c="#b47850",
            label="Clutter",
            zorder=3,
        )

    true_dets = payload.get("step_true_dets") or []
    if is_lot3_meas and true_dets:
        ax.scatter(
            [p[0] for p in true_dets],
            [p[1] for p in true_dets],
            s=50,
            c="#add8e6",
            edgecolors="#3060a0",
            label="True detections (step)",
            zorder=4,
        )

    meas = payload.get("meas_trail") or []
    if not is_lot3_meas and meas:
        ax.scatter(
            [p[0] for p in meas],
            [p[1] for p in meas],
            s=40,
            c="#add8e6",
            edgecolors="#3060a0",
            label="Detections (Cartesian)",
            zorder=3,
        )

    if payload.get("is_lot3_nn"):
        assoc = payload.get("step_associated")
        if assoc:
            ax.scatter(
                [assoc[0]],
                [assoc[1]],
                s=80,
                c="#ffc83c",
                edgecolors="#806020",
                label="Associated detection",
                zorder=6,
            )

    est = payload.get("est_trail") or []
    if est:
        ax.plot(
            [p[0] for p in est],
            [p[1] for p in est],
            color="green",
            linewidth=2,
            label="Track A",
        )

    truth_b = payload.get("truth_b_trail") or []
    if truth_b:
        ax.plot(
            [p[0] for p in truth_b],
            [p[1] for p in truth_b],
            color="#c8c8ff",
            linewidth=2,
            label="Ground truth B",
        )

    est_b = payload.get("est_b_trail") or []
    if est_b:
        ax.plot(
            [p[0] for p in est_b],
            [p[1] for p in est_b],
            color="#ff8c50",
            linewidth=2,
            label="Track B",
        )

    ell = payload.get("last_ellipse")
    if ell:
        ring = ellipse_ring(
            ell["center"],
            (ell["semi_axes"][0], ell["semi_axes"][1]),
            ell["angle_rad"],
        )
        poly = Polygon(
            ring,
            closed=True,
            facecolor=(80 / 255, 200 / 255, 120 / 255, 0.16),
            edgecolor=(80 / 255, 200 / 255, 120 / 255, 1.0),
            linewidth=1.0,
            label="Covariance (2σ)",
        )
        ax.add_patch(poly)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(dest, dpi=100)
    plt.close(fig)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: plot_viewer_trail.py <trail.json> <output.png>", file=sys.stderr)
        return 2
    payload = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    plot_trail(payload, Path(argv[2]))
    print("wrote", Path(argv[2]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
