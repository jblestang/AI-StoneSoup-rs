#!/usr/bin/env python3
"""Export Tutorial 04 particle-filter weighted-mean track (seed 1991, 1000 particles).

Measurements are taken from ``lot2_ekf.json`` so the Rust port can avoid replaying
the truth/measurement RNG preamble and still match Stone Soup filter dynamics.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

UPSTREAM_SHA = "a4336b920a799cfe0a77ecb05867c5deeb371c7a"
ROOT = Path(__file__).resolve().parents[2]
EKF = ROOT / "verification" / "fixtures" / "lot2_ekf.json"
OUT = ROOT / "verification" / "fixtures" / "lot2_pf.json"


def main() -> int:
    import numpy as np
    from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
    from stonesoup.models.transition.linear import (
        CombinedLinearGaussianTransitionModel,
        ConstantVelocity,
    )
    from stonesoup.types.array import StateVectors
    from stonesoup.types.detection import Detection
    from stonesoup.types.numeric import Probability
    from stonesoup.types.state import ParticleState

    ekf = json.loads(EKF.read_text(encoding="utf-8"))
    start_time = datetime.now().replace(microsecond=0)
    np.random.seed(1991)

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.05), ConstantVelocity(0.05)]
    )
    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.diag([np.radians(5) ** 2, 1.0]),
        translation_offset=np.array([[50], [0]]),
    )
    measurements = [
        Detection(
            np.array(m).reshape(-1, 1),
            timestamp=start_time + timedelta(seconds=k),
            measurement_model=measurement_model,
        )
        for k, m in enumerate(ekf["measurements"])
    ]

    number_particles = 1000
    mean = np.array([0, 1, 0, 1])
    std = np.sqrt([1.5, 0.5, 1.5, 0.5])
    # Diagonal prior with legacy NumPy normals (matches ``stonesoup_sim`` / RandomKit).
    samples = np.random.standard_normal((number_particles, 4)) * std + mean
    prior = ParticleState(
        state_vector=StateVectors(samples.T),
        weight=np.array([Probability(1 / number_particles)] * number_particles),
        timestamp=start_time,
    )

    # Manual predict/update matching ``stonesoup_sim`` RandomKit + diagonal log-likelihood.
    LOT1_A_Q = np.array(
        [
            [-0.1187916679626401, -0.2219642619541573, 0.0, 0.0],
            [0.0, 0.0, -0.1187916679626401, -0.2219642619541573],
            [-0.050549048352273925, 0.027053029685161423, 0.0, 0.0],
            [0.0, 0.0, -0.050549048352273925, 0.027053029685161423],
        ]
    )
    F = transition_model.matrix(time_interval=timedelta(seconds=1))
    v0, v1 = np.radians(5) ** 2, 1.0
    log_det = np.log(2 * np.pi * v0 * v1)
    sx, sy = 50.0, 0.0

    def mod_bearing(x: float) -> float:
        return (x + np.pi) % (2 * np.pi) - np.pi

    def mvn4_q(z: np.ndarray) -> np.ndarray:
        return z @ LOT1_A_Q

    def h(state: np.ndarray) -> np.ndarray:
        dx, dy = state[0] - sx, state[2] - sy
        return np.array([mod_bearing(np.arctan2(dy, dx)), np.hypot(dx, dy)])

    sv = prior.state_vector.copy()
    log_w = prior.log_weight.copy()
    ts_prev = prior.timestamp

    track = []
    for measurement in measurements:
        dt = (measurement.timestamp - ts_prev).total_seconds()
        if dt > 0:
            f = transition_model.matrix(time_interval=timedelta(seconds=dt))
            for i in range(number_particles):
                z = np.random.standard_normal(4)
                sv[:, i] = f @ sv[:, i] + mvn4_q(z)
        m = measurement.state_vector.ravel().astype(float)
        for i in range(number_particles):
            z = h(sv[:, i])
            innov0 = mod_bearing(m[0] - z[0])
            innov1 = m[1] - z[1]
            log_w[i] += -0.5 * (innov0**2 / v0 + innov1**2 / v1 + log_det)
        log_w -= np.log(np.exp(log_w - log_w.max()).sum()) + log_w.max()
        w = np.exp(log_w)
        w /= w.sum()
        mean = (sv * w).sum(axis=1)
        track.append([float(x) for x in mean.ravel()])
        ts_prev = measurement.timestamp

    payload = {
        "upstream_sha": UPSTREAM_SHA,
        "seed": 1991,
        "n_particles": number_particles,
        "measurements_from": "verification/fixtures/lot2_ekf.json",
        "track": track,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print("wrote", OUT, "sha256", hashlib.sha256(text.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
