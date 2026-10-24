#!/usr/bin/env python3
"""Render dual_radar_viz.json as an x–y tracking overview PNG."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("verification/reports/dual_radar_viz.json")
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/opt/cursor/artifacts/screenshots/dual_radar_tracking_result.png")
    payload = json.loads(src.read_text())
    plot(payload, dest)
    print(dest)


def plot(payload: dict[str, Any], dest: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge

    dest.parent.mkdir(parents=True, exist_ok=True)
    ra = float(payload["range_a_km"])
    rb = float(payload["range_b_km"])
    truths: list[dict[str, Any]] = payload["truths"]
    tracks: list[dict[str, Any]] = payload["track_polylines"]

    fig, ax = plt.subplots(figsize=(14, 14), constrained_layout=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_title(
        "Dual radar tracking (seed 42, 60 s)\n"
        f"{payload['continuous_track_count']} continuous tracks / "
        f"{payload['truths_with_any_plot']} plotted / "
        f"{payload['num_truths']} truths"
    )

    ax.add_patch(Circle((0, 0), ra, fill=False, color="#888888", linewidth=1.0, linestyle="--", label=f"Radar A {ra:.0f} km"))
    ax.add_patch(Circle((0, 0), rb, fill=False, color="#aaaaaa", linewidth=0.8, linestyle=":", label=f"Radar B {rb:.0f} km"))
    ax.scatter([0], [0], s=120, c="black", marker="^", zorder=5, label="Co-located radars")

    colors = {
        "none": "#cccccc",
        "a_only": "#e45756",
        "b_only": "#4c78a8",
        "both": "#59a14f",
    }
    for t in truths:
        c = colors.get(t["plot_visibility"], "#999999")
        size = 8 if t["plot_visibility"] == "none" else 12
        alpha = 0.35 if t["plot_visibility"] == "none" else 0.75
        ax.scatter([t["x_km"]], [t["y_km"]], s=size, c=c, alpha=alpha, edgecolors="none", zorder=2)

    for tr in tracks:
        pts = tr["points_km"]
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color="#1f1f1f", alpha=0.12, linewidth=0.6, zorder=3)

    lim = ra * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.25)

    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["both"], markersize=8, label="Plotted both radars"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["a_only"], markersize=8, label="Plotted A only"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["b_only"], markersize=8, label="Plotted B only"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["none"], markersize=6, label="No plot"),
        Line2D([0], [0], color="#1f1f1f", alpha=0.4, linewidth=1, label="Continuous filter path (sampled)"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=9)

    fig.savefig(dest, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
