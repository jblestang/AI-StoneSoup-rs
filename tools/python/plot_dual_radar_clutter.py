#!/usr/bin/env python3
"""Render dual_radar_viz.json as an x–y tracking overview PNG."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("verification/reports/dual_radar_viz.json")
    dest = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("/opt/cursor/artifacts/screenshots/dual_radar_tracking_result.png")
    )
    payload = json.loads(src.read_text())
    plot(payload, dest)
    print(dest)


def bearing_north_cw_to_matplotlib_deg(bearing_deg: float) -> float:
    """North-clockwise bearing → matplotlib angle (degrees CCW from +x/east)."""
    return 90.0 - bearing_deg


def plot(payload: dict[str, Any], dest: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Wedge

    dest.parent.mkdir(parents=True, exist_ok=True)
    ra = float(payload["range_a_km"])
    rb = float(payload["range_b_km"])
    fov = float(payload.get("fov_deg", 90.0))
    beams = payload.get("radar_beam_deg", [0.0, 0.0])
    truths: list[dict[str, Any]] = payload["truths"]
    truth_paths: list[dict[str, Any]] = payload.get("truth_trajectories") or []
    filter_tracks: list[dict[str, Any]] = payload.get("filter_tracks") or []

    # Back-compat with older JSON
    if not filter_tracks and payload.get("track_polylines"):
        for tr in payload["track_polylines"]:
            filter_tracks.append(
                {
                    "track_id": 0,
                    "truth_id": tr.get("truth_id"),
                    "continuous": True,
                    "is_false_track": False,
                    "points_km": tr["points_km"],
                }
            )

    fig, ax = plt.subplots(figsize=(14, 14), constrained_layout=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    untracked_plotted = int(payload.get("plotted_without_continuous_count", 0))
    ax.set_title(
        "Dual radar tracking (seed 42, 60 s)\n"
        f"{payload['continuous_track_count']} continuous tracks · "
        f"{untracked_plotted} plotted but untracked · "
        f"{payload['truths_with_any_plot']} plotted / {payload['num_truths']} truths"
    )

    ax.add_patch(
        Circle((0, 0), ra, fill=False, color="#888888", linewidth=1.0, linestyle="--")
    )
    ax.add_patch(
        Circle((0, 0), rb, fill=False, color="#aaaaaa", linewidth=0.8, linestyle=":")
    )
    ax.scatter([0], [0], s=120, c="black", marker="^", zorder=8)

    half = fov / 2.0
    wedge_style = [("#c44e52", 0.07), ("#4c72b0", 0.07)]
    for beam, (color, alpha), r_km, label in zip(
        beams,
        wedge_style,
        [ra, rb],
        ["Radar A beam", "Radar B beam"],
    ):
        center = bearing_north_cw_to_matplotlib_deg(float(beam))
        ax.add_patch(
            Wedge(
                (0, 0),
                r_km,
                center - half,
                center + half,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=0.5,
                zorder=0,
            )
        )

    # Ground-truth motion (full 60 s history)
    for traj in truth_paths:
        pts = traj.get("points_km") or []
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color="#9a9a9a", alpha=0.38, linewidth=0.55, zorder=1)

    # Filter histories: non-continuous / false first, continuous on top
    for tr in filter_tracks:
        if tr.get("continuous"):
            continue
        pts = tr.get("points_km") or []
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if tr.get("is_false_track"):
            ax.plot(xs, ys, color="#e15759", alpha=0.55, linewidth=0.9, linestyle=":", zorder=3)
        elif tr.get("truth_id") is not None:
            ax.plot(xs, ys, color="#f28e2b", alpha=0.45, linewidth=0.75, linestyle="--", zorder=3)
        else:
            ax.plot(xs, ys, color="#bab0ac", alpha=0.35, linewidth=0.6, linestyle="--", zorder=2)

    for tr in filter_tracks:
        if not tr.get("continuous"):
            continue
        pts = tr.get("points_km") or []
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color="#222222", alpha=0.72, linewidth=1.35, zorder=4)

    vis_colors = {
        "both": "#59a14f",
        "a_only": "#e45756",
        "b_only": "#4c78a8",
        "none": "#d4d4d4",
    }

    for t in truths:
        if t.get("plotted_untracked"):
            ax.scatter(
                [t["x_km"]],
                [t["y_km"]],
                s=70,
                facecolors="none",
                edgecolors="#b07aa1",
                linewidths=1.4,
                marker="o",
                zorder=6,
            )
            ax.scatter(
                [t["x_km"]],
                [t["y_km"]],
                s=28,
                c="#b07aa1",
                marker="x",
                linewidths=1.2,
                zorder=7,
            )
            continue
        if t.get("has_continuous_track"):
            c = vis_colors.get(t["plot_visibility"], "#333333")
            ax.scatter([t["x_km"]], [t["y_km"]], s=14, c=c, alpha=0.85, edgecolors="none", zorder=6)
            continue
        c = vis_colors.get(t["plot_visibility"], "#999999")
        size = 6 if t["plot_visibility"] == "none" else 10
        alpha = 0.45 if t["plot_visibility"] == "none" else 0.65
        ax.scatter([t["x_km"]], [t["y_km"]], s=size, c=c, alpha=alpha, edgecolors="none", zorder=5)

    lim = ra * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.25)

    legend_elems = [
        Line2D([0], [0], color="#b0b0b0", alpha=0.5, linewidth=1, label="Truth trajectory (60 s)"),
        Line2D([0], [0], color="#1a1a1a", alpha=0.7, linewidth=1.5, label="Continuous filter track"),
        Line2D([0], [0], color="#f28e2b", alpha=0.7, linewidth=1.2, linestyle="--", label="Filter fragment (truth assoc.)"),
        Line2D([0], [0], color="#e15759", alpha=0.7, linewidth=1.2, linestyle=":", label="False filter track"),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#b07aa1",
            linestyle="None",
            markersize=8,
            label="Plotted, no continuous track",
        ),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=vis_colors["none"], markersize=6, label="Never plotted"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8)

    fig.savefig(dest, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
