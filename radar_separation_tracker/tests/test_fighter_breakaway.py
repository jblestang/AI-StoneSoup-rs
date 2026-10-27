import numpy as np

from radar_tracker.simulator.truth import generate_scenario_truth


def test_breakaway_not_instant_reverse():
    truth = generate_scenario_truth("probable_detach_opposite", 120, 1.0, seed=42)
    t1 = next(t for t in truth.targets if t.target_id == "T1")
    t2 = next(t for t in truth.targets if t.target_id == "T2")
    sep = 60
    v1 = np.array([t1.states[sep][1], t1.states[sep][3]])
    v2_0 = np.array([t2.states[0][1], t2.states[0][3]])
    np.testing.assert_allclose(v1, v2_0, rtol=0.01)
    v2_1 = np.array([t2.states[1][1], t2.states[1][3]])
    dot = float(np.dot(v2_0, v2_1))
    assert dot > 0, "pas de demi-tour instantané après 1 s"
    v2_late = np.array([t2.states[min(25, len(t2.states) - 1)][1], t2.states[min(25, len(t2.states) - 1)][3]])
    assert float(np.dot(v1, v2_late)) < 0, "cap opposé atteint après la cassure"
