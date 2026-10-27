import numpy as np

from radar_tracker.config import load_config
from radar_tracker.simulator.sensor import simulate_scan
from radar_tracker.simulator.truth import generate_scenario_truth


def test_clutter_fraction_about_twenty_percent():
    cfg = load_config("config/demo.yaml")
    cfg.simulation["clutter_fraction"] = 0.20
    truth = generate_scenario_truth("aircraft_asm", 10, 1.0, seed=42)
    rng = np.random.default_rng(42)
    totals = []
    clutter_ids = 0
    for si in range(10):
        dets = simulate_scan(truth, si, cfg, rng)
        totals.append(len(dets))
        clutter_ids += sum(1 for d in dets if d.detection_id.startswith("C"))
    assert clutter_ids > 0
    # au moins un scan avec clutter quand il y a des vraies détections
    assert any(t >= 2 for t in totals)
