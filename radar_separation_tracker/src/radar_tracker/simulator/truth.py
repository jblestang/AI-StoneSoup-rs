"""Vérité terrain pour scénarios reproductibles."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np


@dataclass
class TruthTarget:
    target_id: str
    states: list[np.ndarray]  # each [x,vx,y,vy,z,vz]
    timestamps: list[datetime]
    birth_time: datetime
    death_time: datetime | None = None
    parent_id: str | None = None
    separation_time: datetime | None = None


@dataclass
class GroundTruthEvent:
    event_id: str
    event_type: str  # unresolved_disperses | probable_detach | crossing | independent_birth
    time: datetime
    parent_ids: list[str]
    child_ids: list[str]
    first_resolvable_time: datetime | None = None


@dataclass
class ScenarioTruth:
    name: str
    targets: list[TruthTarget]
    events: list[GroundTruthEvent]
    scan_interval_s: float
    start_time: datetime


def _integrate_cv(state: np.ndarray, dt: float) -> np.ndarray:
    x, vx, y, vy, z, vz = state
    return np.array([x + vx * dt, vx, y + vy * dt, vy, z + vz * dt, vz])


def _fighter_breakaway_states(
    initial: np.ndarray,
    n_steps: int,
    sep_idx: int,
    times: list,
    dt: float,
    *,
    max_load_g: float = 4.0,
    climb_accel_mps2: float = 2.5,
    lateral_break_bias: float = 0.65,
) -> tuple[list[np.ndarray], list]:
    """Cassure chasseur post-séparation : même vitesse initiale, virage chargé limité en g.

    Hypothèse simulation (pas mesurée) : charge max ``max_load_g``, reprendre le cap
    opposé horizontalement sur ~15–25 s via accélération bornée (pas de flip instantané).
    """
    g = 9.81
    max_accel = max_load_g * g
    state = initial.astype(float).copy()
    states: list[np.ndarray] = []
    ts = []
    v0 = np.array([state[1], state[3], state[5]], dtype=float)
    v0_h = v0.copy()
    v0_h[2] = 0.0
    v0_norm = float(np.linalg.norm(v0_h))
    if v0_norm < 1.0:
        target_h = np.array([-1.0, 0.0, 0.0])
    else:
        target_h = -v0_h / v0_norm

    for i in range(sep_idx, n_steps):
        states.append(state.copy())
        ts.append(times[i])
        vx, vy, vz = state[1], state[3], state[5]
        v_h = np.array([vx, vy, 0.0])
        speed = float(np.linalg.norm(v_h))
        if speed < 5.0:
            state = _integrate_cv(state, dt)
            continue
        cur = v_h / speed
        # Cassure : composante latérale (bias droit) + rétrogradation progressive vers cap opposé
        lateral = np.array([cur[1], -cur[0], 0.0])
        lat_n = float(np.linalg.norm(lateral))
        if lat_n > 1e-6:
            lateral /= lat_n
        blend = min(1.0, (i - sep_idx) / max(1, int(20 / dt)))
        desired_dir = (1.0 - blend) * cur + blend * target_h
        desired_dir = desired_dir + lateral_break_bias * lateral
        desired_dir[2] = 0.0
        dn = float(np.linalg.norm(desired_dir))
        if dn > 1e-6:
            desired_dir /= dn
        desired_v = desired_dir * speed
        accel_h = desired_v - v_h
        a_norm = float(np.linalg.norm(accel_h))
        if a_norm > max_accel:
            accel_h *= max_accel / a_norm
        state[1] += accel_h[0] * dt
        state[3] += accel_h[1] * dt
        state[5] += climb_accel_mps2 * dt
        state[0] += state[1] * dt
        state[2] += state[3] * dt
        state[4] += state[5] * dt

    return states, ts


def generate_scenario_truth(
    name: str,
    duration_s: float,
    scan_interval_s: float,
    start_time: datetime | None = None,
    seed: int = 42,
) -> ScenarioTruth:
    rng = np.random.default_rng(seed)
    t0 = start_time or datetime(2026, 1, 1, 12, 0, 0)
    n_steps = int(duration_s / scan_interval_s) + 1
    times = [t0 + timedelta(seconds=i * scan_interval_s) for i in range(n_steps)]
    targets: list[TruthTarget] = []
    events: list[GroundTruthEvent] = []

    def make_track(tid: str, x0, v, t_start_idx: int = 0, parent=None, sep_time=None):
        states = []
        ts = []
        for i in range(t_start_idx, n_steps):
            dt = (times[i] - (times[t_start_idx] if t_start_idx else times[0])).total_seconds()
            s0 = np.array([x0[0], v[0], x0[1], v[1], x0[2], v[2]])
            states.append(_integrate_cv(s0, dt - (times[t_start_idx] - times[0]).total_seconds() if i > t_start_idx else 0))
            # recompute properly from absolute
        # simpler loop
        state = np.array([x0[0], v[0], x0[1], v[1], x0[2], v[2]], dtype=float)
        states = []
        ts = []
        for i in range(t_start_idx, n_steps):
            states.append(state.copy())
            ts.append(times[i])
            state = _integrate_cv(state, scan_interval_s)
        targets.append(
            TruthTarget(
                target_id=tid,
                states=states,
                timestamps=ts,
                birth_time=times[t_start_idx],
                parent_id=parent,
                separation_time=sep_time,
            )
        )

    if name == "single_target":
        make_track("T1", (30000.0, 20000.0, 8000.0), (120.0, -80.0, 5.0))

    elif name == "unresolved_group_disperses":
        sep_idx = int(40 / scan_interval_s)
        v1 = (100.0, 60.0, 0.0)
        v2 = (80.0, 40.0, 0.0)
        center = (35000.0, 25000.0, 9000.0)
        off = 5.0
        # T1 et T2 proches puis séparation cinématique
        state1 = np.array([center[0] - off, v1[0], center[1], v1[1], center[2], v1[2]])
        state2 = np.array([center[0] + off, v2[0], center[1], v2[1], center[2], v2[2]])
        s1, ts1, s2, ts2 = [], [], [], []
        for i in range(n_steps):
            if i < sep_idx:
                # même centroid + petit offset (non résolu en mesure)
                c = _integrate_cv(
                    np.array([center[0], v1[0], center[1], v1[1], center[2], v1[2]]),
                    i * scan_interval_s,
                )
                s1.append(np.array([c[0] - off, v1[0], c[2], v1[1], c[4], v1[2]]))
                s2.append(np.array([c[0] + off, v1[0], c[2], v1[1], c[4], v1[2]]))
            else:
                state1 = _integrate_cv(state1, scan_interval_s)
                state2 = _integrate_cv(state2, scan_interval_s)
                s1.append(state1.copy())
                s2.append(state2.copy())
            ts1.append(times[i])
            ts2.append(times[i])
        targets.append(
            TruthTarget("T1", s1, ts1, times[0], parent_id="GROUP", separation_time=times[sep_idx])
        )
        targets.append(
            TruthTarget("T2", s2, ts2, times[0], parent_id="GROUP", separation_time=times[sep_idx])
        )
        events.append(
            GroundTruthEvent(
                event_id="E1",
                event_type="unresolved_disperses",
                time=times[sep_idx],
                parent_ids=["GROUP"],
                child_ids=["T1", "T2"],
                first_resolvable_time=times[min(sep_idx + 3, n_steps - 1)],
            )
        )

    elif name == "probable_detach":
        make_track("T1", (28000.0, 15000.0, 7000.0), (150.0, 90.0, 0.0))
        sep_idx = int(50 / scan_interval_s)
        parent_state = targets[0].states[sep_idx]
        make_track(
            "T2",
            (parent_state[0], parent_state[2], parent_state[4]),
            (parent_state[1] + 20, parent_state[3] - 15, 2.0),
            sep_idx,
            parent="T1",
            sep_time=times[sep_idx],
        )
        events.append(
            GroundTruthEvent(
                event_id="E1",
                event_type="probable_detach",
                time=times[sep_idx],
                parent_ids=["T1"],
                child_ids=["T1", "T2"],
            )
        )

    elif name in ("probable_detach_opposite", "detach_opposite_1min"):
        # Scénario B : 120 s, séparation à 60 s — leader poursuit, #2 casse en chasseur
        # (virage chargé ~4 g, remise de cap opposée sur ~20 s, légère montée).
        v_continue = (155.0, 85.0, 0.0)  # ~310 m/s horizontal (hyp. démo subsonique)
        make_track("T1", (32000.0, 22000.0, 8500.0), v_continue)
        sep_idx = int(60 / scan_interval_s)
        if sep_idx >= n_steps:
            sep_idx = max(1, n_steps // 2)
        parent_state = targets[0].states[sep_idx]
        s2, ts2 = _fighter_breakaway_states(
            parent_state,
            n_steps,
            sep_idx,
            times,
            scan_interval_s,
            max_load_g=4.0,
            climb_accel_mps2=2.5,
            lateral_break_bias=0.55,
        )
        targets.append(
            TruthTarget(
                target_id="T2",
                states=s2,
                timestamps=ts2,
                birth_time=times[sep_idx],
                parent_id="T1",
                separation_time=times[sep_idx],
            )
        )
        events.append(
            GroundTruthEvent(
                event_id="E1",
                event_type="probable_detach",
                time=times[sep_idx],
                parent_ids=["T1"],
                child_ids=["T1", "T2"],
                first_resolvable_time=times[min(sep_idx + 3, n_steps - 1)],
            )
        )

    elif name == "crossing":
        make_track("T1", (20000.0, 10000.0, 6000.0), (200.0, 100.0, 0.0))
        make_track("T2", (40000.0, 30000.0, 6000.0), (-180.0, -120.0, 0.0))

    elif name == "independent_birth_near":
        make_track("T1", (30000.0, 20000.0, 8000.0), (100.0, 50.0, 0.0))
        make_track("T2", (30100.0, 20050.0, 8000.0), (-50.0, 80.0, 0.0), int(30 / scan_interval_s))

    elif name == "dense_clutter":
        make_track("T1", (30000.0, 20000.0, 8000.0), (100.0, 50.0, 0.0))

    elif name == "missed_detections":
        make_track("T1", (30000.0, 20000.0, 8000.0), (100.0, 50.0, 0.0))

    elif name == "azimuth_wrap":
        make_track("T1", (1000.0, -500.0, 2000.0), (-200.0, 50.0, 0.0))

    elif name == "stress_1000_det":
        make_track("T1", (30000.0, 20000.0, 8000.0), (100.0, 50.0, 0.0))
        for k in range(5):
            make_track(
                f"T{k+2}",
                (30000 + 5000 * k, 20000 + 3000 * k, 8000.0),
                (100.0 + k * 5, 50.0, 0.0),
            )

    else:
        make_track("T1", (30000.0, 20000.0, 8000.0), (100.0, 50.0, 0.0))

    return ScenarioTruth(
        name=name,
        targets=targets,
        events=events,
        scan_interval_s=scan_interval_s,
        start_time=t0,
    )
