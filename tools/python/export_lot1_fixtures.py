#!/usr/bin/env python3
"""Export Tutorial 01 Kalman fixtures from pinned Stone Soup (v1.9.1)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

UPSTREAM_SHA = "a4336b920a799cfe0a77ecb05867c5deeb371c7a"
OUT = Path(__file__).resolve().parents[2] / "verification" / "fixtures" / "lot1_kalman.json"


def main() -> int:
    import numpy as np
    from stonesoup.models.measurement.linear import LinearGaussian
    from stonesoup.models.transition.linear import (
        CombinedLinearGaussianTransitionModel,
        ConstantVelocity,
    )
    from stonesoup.predictor.kalman import KalmanPredictor
    from stonesoup.types.detection import Detection
    from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
    from stonesoup.types.hypothesis import SingleHypothesis
    from stonesoup.types.state import GaussianState
    from stonesoup.updater.kalman import KalmanUpdater

    np.random.seed(1991)
    start_time = datetime.now().replace(microsecond=0)
    q_x, q_y = 0.05, 0.05
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(q_x), ConstantVelocity(q_y)]
    )
    timesteps = [start_time]
    truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=timesteps[0])])
    num_steps = 20
    for k in range(1, num_steps + 1):
        timesteps.append(start_time + timedelta(seconds=k))
        truth.append(
            GroundTruthState(
                transition_model.function(
                    truth[k - 1], noise=True, time_interval=timedelta(seconds=1)
                ),
                timestamp=timesteps[k],
            )
        )

    measurement_model = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.array([[5.0, 0.0], [0.0, 5.0]]),
    )
    measurements = []
    for state in truth:
        measurement = measurement_model.function(state, noise=True)
        measurements.append(
            Detection(measurement, timestamp=state.timestamp, measurement_model=measurement_model)
        )

    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    prior = GaussianState(
        [[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time
    )
    track_states = []
    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)
        post = updater.update(hypothesis)
        track_states.append(post)
        prior = post

    def sv(state) -> list[float]:
        return [float(x) for x in state.state_vector]

    payload = {
        "upstream_sha": UPSTREAM_SHA,
        "seed": 1991,
        "q_x": q_x,
        "q_y": q_y,
        "dt_seconds": 1.0,
        "prior_mean": [0.0, 1.0, 0.0, 1.0],
        "prior_cov_diag": [1.5, 0.5, 1.5, 0.5],
        "measurement_noise_diag": [5.0, 5.0],
        "truth": [sv(s) for s in truth],
        "measurements": [m.state_vector.ravel().tolist() for m in measurements],
        "track": [sv(s) for s in track_states],
        "track_cov_diag_last": [float(track_states[-1].covar[i, i]) for i in range(4)],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print("wrote", OUT, "sha256", hashlib.sha256(text.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
