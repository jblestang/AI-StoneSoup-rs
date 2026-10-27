#!/usr/bin/env python3
"""Exécute le suivi sur un CSV de détections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar_tracker.config import load_config
from radar_tracker.csv_reader import read_scans_with_gaps
from radar_tracker.events import export_events_json
from radar_tracker.tracker import RadarSeparationTracker


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "config" / "demo.yaml"))
    p.add_argument("--detections", required=True)
    p.add_argument("--output-dir", default=str(ROOT / "output"))
    args = p.parse_args()
    cfg = load_config(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    interval = float(cfg.simulation.get("scan_interval_s", 1.0))
    scans = read_scans_with_gaps(args.detections, expected_interval_s=interval)
    tracker = RadarSeparationTracker(cfg)
    result = tracker.run(scans)
    tracks_out = []
    for tr in result.tracks:
        sv = tr.state.state_vector.ravel().tolist() if tr.state else []
        tracks_out.append(
            {
                "track_id": tr.track_id,
                "life": tr.life.value,
                "score": tr.score,
                "state": sv,
                "parent_track_id": tr.parent_track_id,
            }
        )
    (out / "tracks.json").write_text(json.dumps(tracks_out, indent=2), encoding="utf-8")
    export_events_json(result.events, str(out / "events.json"))
    metrics = [
        {
            "processing_time_s": m.processing_time_s,
            "n_detections": m.n_detections,
            "n_tracks": m.n_tracks,
            "n_hypotheses": m.n_hypotheses,
        }
        for m in result.scan_metrics
    ]
    (out / "scan_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Tracks: {len(result.tracks)}, Events: {len(result.events)}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
