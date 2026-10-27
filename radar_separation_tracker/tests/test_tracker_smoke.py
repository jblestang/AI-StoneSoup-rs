from pathlib import Path

import numpy as np

from radar_tracker.config import load_config
from radar_tracker.simulator.sensor import simulate_scan, write_detections_csv
from radar_tracker.simulator.truth import generate_scenario_truth
from radar_tracker.tracker import RadarSeparationTracker
from radar_tracker.csv_reader import read_scans_with_gaps


def test_end_to_end_single_target(tmp_path):
    cfg = load_config("config/demo.yaml")
    truth = generate_scenario_truth("single_target", 30, 1.0, seed=1)
    rng = np.random.default_rng(1)
    dets = []
    for si in range(31):
        dets.extend(simulate_scan(truth, si, cfg, rng))
    csv = tmp_path / "d.csv"
    write_detections_csv(str(csv), dets)
    scans = read_scans_with_gaps(csv, 1.0)
    out = RadarSeparationTracker(cfg).run(scans)
    assert out.scan_metrics
    assert out.scan_metrics[0].processing_time_s >= 0
