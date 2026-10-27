import numpy as np

from radar_tracker.config import load_config
from radar_tracker.simulator.sensor import _resolve_groups


def test_unresolved_merge():
    cfg = load_config("config/demo.yaml")
    # two targets within resolution
    meas = [
        ("T1", 30000.0, 0.5, 0.1),
        ("T2", 30010.0, 0.5005, 0.1005),
    ]
    groups = _resolve_groups(meas, cfg)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_resolved_split():
    cfg = load_config("config/demo.yaml")
    meas = [
        ("T1", 30000.0, 0.5, 0.1),
        ("T2", 30500.0, 0.6, 0.1),
    ]
    groups = _resolve_groups(meas, cfg)
    assert len(groups) == 2
