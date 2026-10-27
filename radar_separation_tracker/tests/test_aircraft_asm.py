import numpy as np

from radar_tracker.simulator.truth import generate_scenario_truth


def test_aircraft_no_instant_reverse():
    truth = generate_scenario_truth("aircraft_asm", 120, 1.0, seed=42)
    ac = next(t for t in truth.targets if t.target_id == "AC")
    sep = 60
    v0 = np.array([ac.states[sep][1], ac.states[sep][3]])
    v1 = np.array([ac.states[sep + 1][1], ac.states[sep + 1][3]])
    assert float(np.dot(v0, v1)) > 0.99 * float(np.linalg.norm(v0) * np.linalg.norm(v1))


def test_missile_launch_and_dive():
    truth = generate_scenario_truth("aircraft_asm", 120, 1.0, seed=42)
    ac = next(t for t in truth.targets if t.target_id == "AC")
    ms = next(t for t in truth.targets if t.target_id == "MS")
    sep = 60
    np.testing.assert_allclose(ac.states[sep][:6], ms.states[0][:6], rtol=0.02)
    z_late = ms.states[min(20, len(ms.states) - 1)][4]
    assert z_late < ac.states[sep][4] - 500.0
    speed_ms = float(np.linalg.norm(ms.states[5][[1, 3, 5]]))
    speed_ac = float(np.linalg.norm(ac.states[65][[1, 3, 5]]))
    assert speed_ms > speed_ac
