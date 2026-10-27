import numpy as np

from radar_tracker.config import load_config
from radar_tracker.scoring import common_origin_score, innovation_log_likelihood


def test_innovation_angle_wrap():
    cfg = load_config("config/demo.yaml")
    inn = np.array([[0.01], [3.0], [1.0]])
    S = np.diag([1e-4, 1e-4, 25.0])
    ll = innovation_log_likelihood(inn, S)
    assert ll < 0


def test_common_origin_compatible():
    cfg = load_config("config/demo.yaml")
    p = [np.array([0, 0, 0]), np.array([1, 1, 0])]
    c = [np.array([0.5, 0.5, 0]), np.array([2, 2, 0])]
    pc = [np.diag([100, 1, 100, 1, 100, 1])] * 2
    cc = [np.diag([100, 1, 100, 1, 100, 1])] * 2
    r = common_origin_score(p, c, pc, cc, cfg)
    assert r.compatible
