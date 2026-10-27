"""Benchmark performance par scan."""

from __future__ import annotations

import platform
import statistics
import time
from dataclasses import asdict, dataclass

import numpy as np

from radar_tracker.config import AppConfig, load_config
from radar_tracker.csv_reader import ScanBatch
from radar_tracker.simulator.sensor import simulate_scan
from radar_tracker.simulator.truth import generate_scenario_truth
from radar_tracker.tracker import RadarSeparationTracker


@dataclass
class BenchmarkReport:
    host: str
    median_s: float
    p95_s: float
    max_s: float
    throughput_scans_per_s: float
    max_memory_mb: float
    mean_active_tracks: float
    mean_hypotheses: float


def run_benchmark(cfg: AppConfig, scenario: str = "stress_1000_det", n_scans: int = 60) -> BenchmarkReport:
    import resource

    truth = generate_scenario_truth(
        scenario,
        duration_s=n_scans,
        scan_interval_s=cfg.simulation.get("scan_interval_s", 1.0),
        seed=int(cfg.simulation.get("seed", 42)),
    )
    rng = np.random.default_rng(99)
    tracker = RadarSeparationTracker(cfg)
    times = []
    hyps = []
    tracks = []
    mem = 0.0
    for si in range(n_scans):
        dets = simulate_scan(truth, si, cfg, rng, {"n_clutter": 900})
        batch = ScanBatch(
            timestamp=dets[0].timestamp if dets else truth.start_time,
            radar_id=cfg.radar.id,
            scan_id=f"S{si:05d}",
            detections=[],
        )
        from radar_tracker.csv_reader import DetectionRecord

        batch.detections = [
            DetectionRecord(
                timestamp=d.timestamp,
                radar_id=d.radar_id,
                scan_id=d.scan_id,
                detection_id=d.detection_id,
                range_m=d.range_m,
                azimuth_rad=d.azimuth_rad,
                elevation_rad=d.elevation_rad,
            )
            for d in dets
        ]
        m = tracker.process_scan(batch)
        times.append(m.processing_time_s)
        hyps.append(m.n_hypotheses)
        tracks.append(m.n_tracks)
        mem = max(mem, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)

    times_sorted = sorted(times)
    p95_i = int(0.95 * len(times_sorted)) - 1
    total = sum(times)
    return BenchmarkReport(
        host=platform.platform(),
        median_s=statistics.median(times),
        p95_s=times_sorted[max(0, p95_i)],
        max_s=max(times),
        throughput_scans_per_s=len(times) / max(total, 1e-9),
        max_memory_mb=mem,
        mean_active_tracks=statistics.mean(tracks) if tracks else 0,
        mean_hypotheses=statistics.mean(hyps) if hyps else 0,
    )


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    report = run_benchmark(cfg)
    import json

    print(json.dumps(asdict(report), indent=2))
