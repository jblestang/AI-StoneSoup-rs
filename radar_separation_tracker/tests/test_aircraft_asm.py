import numpy as np

from radar_tracker.simulator.truth import generate_scenario_truth


def test_aircraft_no_instant_reverse():
    truth = generate_scenario_truth("aircraft_asm", 120, 1.0, seed=42)
    ac = next(t for t in truth.targets if t.target_id == "AC")
    sep = 60
    v0 = np.array([ac.states[sep][1], ac.states[sep][3]])
    v1 = np.array([ac.states[sep + 1][1], ac.states[sep + 1][3]])
    np.testing.assert_allclose(v0, v1, rtol=0.01)


def test_aircraft_asm_straight_leg():
    truth = generate_scenario_truth("aircraft_asm", 120, 1.0, seed=42)
    ac = next(t for t in truth.targets if t.target_id == "AC")
    v0 = np.array([ac.states[0][1], ac.states[0][3]])
    v1 = np.array([ac.states[119][1], ac.states[119][3]])
    assert float(np.dot(v0, v1)) > 0.999 * float(np.linalg.norm(v0) * np.linalg.norm(v1))


def test_aircraft_turn_rate_15_deg_per_s_after_launch():
    truth = generate_scenario_truth("aircraft_half_turn_asm", 120, 1.0, seed=42)
    ac = next(t for t in truth.targets if t.target_id == "AC")
    import math

    sep = 60
    v_straight = np.array([ac.states[sep - 1][1], ac.states[sep - 1][3]], dtype=float)
    v_turn = np.array([ac.states[sep][1], ac.states[sep][3]], dtype=float)
    ang0 = math.degrees(math.atan2(v_straight[1], v_straight[0]))
    ang1 = math.degrees(math.atan2(v_turn[1], v_turn[0]))
    delta = (ang1 - ang0 + 180) % 360 - 180
    assert abs(delta) < 0.5
    v1 = np.array([ac.states[sep][1], ac.states[sep][3]], dtype=float)
    v2 = np.array([ac.states[sep + 1][1], ac.states[sep + 1][3]], dtype=float)
    a1 = math.degrees(math.atan2(v1[1], v1[0]))
    a2 = math.degrees(math.atan2(v2[1], v2[0]))
    d2 = (a2 - a1 + 180) % 360 - 180
    assert abs(d2 - 15.0) < 0.5
    v_turn_start = np.array([ac.states[sep][1], ac.states[sep][3]], dtype=float)
    v_after_half = np.array([ac.states[sep + 12][1], ac.states[sep + 12][3]], dtype=float)
    v_steady = np.array([ac.states[sep + 13][1], ac.states[sep + 13][3]], dtype=float)
    dot_half = float(np.dot(v_turn_start, v_after_half))
    n = float(np.linalg.norm(v_turn_start) * np.linalg.norm(v_after_half))
    assert dot_half < -0.95 * n
    assert float(np.dot(v_after_half, v_steady)) > 0.99 * n


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
