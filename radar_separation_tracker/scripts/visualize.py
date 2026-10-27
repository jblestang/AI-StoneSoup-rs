#!/usr/bin/env python3
"""Visualise trajectoires (vérité + pistes) et événements."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar_tracker.config import load_config
from radar_tracker.coordinates import spherical_to_cartesian
from radar_tracker.csv_reader import read_detections_csv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "config" / "demo.yaml"))
    p.add_argument("--detections", required=True)
    p.add_argument("--tracks", default=str(ROOT / "output" / "tracks.json"))
    p.add_argument("--events", default=str(ROOT / "output" / "events.json"))
    p.add_argument("--output", default=str(ROOT / "output" / "plot.png"))
    args = p.parse_args()
    cfg = load_config(args.config)
    radar_pos = np.array(cfg.radar.position_m)
    dets = read_detections_csv(args.detections)
    xs, ys = [], []
    for d in dets[:: max(1, len(dets) // 2000)]:
        p3 = spherical_to_cartesian(d.range_m, d.azimuth_rad, d.elevation_rad, radar_pos)
        xs.append(p3[0])
        ys.append(p3[1])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(xs, ys, s=4, c="lightgray", label="détections (sous-échantillon)")
    if Path(args.tracks).exists():
        tracks = json.loads(Path(args.tracks).read_text())
        for tr in tracks:
            if len(tr.get("state", [])) >= 6:
                ax.plot(tr["state"][0], tr["state"][2], "o", label=tr["track_id"])
    if Path(args.events).exists():
        evs = json.loads(Path(args.events).read_text())
        ax.set_title(f"Suivi radar — {len(evs)} événement(s) confirmé(s)")
    else:
        ax.set_title("Suivi radar")
    ax.set_xlabel("x Est (m)")
    ax.set_ylabel("y Nord (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
