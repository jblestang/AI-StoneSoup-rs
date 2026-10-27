#!/usr/bin/env python3
"""Génère un scénario CSV + vérité terrain (JSON)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar_tracker.config import load_config
from radar_tracker.simulator.sensor import simulate_scan, write_detections_csv
from radar_tracker.simulator.truth import generate_scenario_truth


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "config" / "demo.yaml"))
    p.add_argument("--scenario", default=None)
    p.add_argument("--output-dir", "-o", default=str(ROOT / "output"))
    args = p.parse_args()
    cfg = load_config(args.config)
    scenario = args.scenario or cfg.simulation.get("scenario", "single_target")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    duration = float(cfg.simulation.get("duration_s", 120))
    interval = float(cfg.simulation.get("scan_interval_s", 1.0))
    seed = int(cfg.simulation.get("seed", 42))
    truth = generate_scenario_truth(scenario, duration, interval, seed=seed)
    rng = np.random.default_rng(seed)
    n_scans = int(duration / interval) + 1
    flags = {}
    if scenario == "missed_detections":
        flags["miss_burst"] = True
    if scenario == "dense_clutter":
        flags["clutter"] = True
    all_dets = []
    for si in range(n_scans):
        all_dets.extend(simulate_scan(truth, si, cfg, rng, flags))
    csv_path = out / f"{scenario}_detections.csv"
    write_detections_csv(str(csv_path), all_dets)
    gt_path = out / f"{scenario}_ground_truth.json"
    gt = {
        "name": truth.name,
        "events": [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "time": e.time.isoformat(),
                "parent_ids": e.parent_ids,
                "child_ids": e.child_ids,
            }
            for e in truth.events
        ],
    }
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path} ({len(all_dets)} detections)")
    print(f"Wrote {gt_path}")


if __name__ == "__main__":
    main()
