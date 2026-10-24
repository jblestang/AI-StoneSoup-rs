#!/usr/bin/env python3
"""Export Tutorial 02 EKF fixtures (bearing–range sensor, seed 1991)."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

UPSTREAM_SHA = "a4336b920a799cfe0a77ecb05867c5deeb371c7a"
OUT = Path(__file__).resolve().parents[2] / "verification" / "fixtures" / "lot2_ekf.json"


def main() -> int:
    import numpy as np
    from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
    from stonesoup.models.transition.linear import (
        CombinedLinearGaussianTransitionModel,
        ConstantVelocity,
    )
    from stonesoup.predictor.kalman import ExtendedKalmanPredictor
    from stonesoup.types.detection import Detection
    from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
    from stonesoup.types.hypothesis import SingleHypothesis
    from stonesoup.types.state import GaussianState
    from stonesoup.updater.kalman import ExtendedKalmanUpdater

    np.random.seed(1991)
    start_time = datetime.now().replace(microsecond=0)
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.05), ConstantVelocity(0.05)]
    )
    truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])
    for k in range(1, 21):
        ts = start_time + timedelta(seconds=k)
        truth.append(
            GroundTruthState(
                transition_model.function(
                    truth[k - 1], noise=True, time_interval=timedelta(seconds=1)
                ),
                timestamp=ts,
            )
        )

    sensor_x, sensor_y = 50, 0
    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.diag([np.radians(5) ** 2, 1.0]),
        translation_offset=np.array([[sensor_x], [sensor_y]]),
    )
    measurements = []
    for state in truth:
        m = measurement_model.function(state, noise=True)
        measurements.append(
            Detection(m, timestamp=state.timestamp, measurement_model=measurement_model)
        )

    predictor = ExtendedKalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model)
    prior = GaussianState(
        [[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time
    )
    track = []
    for m in measurements:
        pred = predictor.predict(prior, timestamp=m.timestamp)
        prior = updater.update(SingleHypothesis(pred, m))
        track.append(prior)

    def sv(state) -> list[float]:
        return [float(x) for x in state.state_vector.ravel()]

    payload = {
        "upstream_sha": UPSTREAM_SHA,
        "seed": 1991,
        "sensor_xy": [sensor_x, sensor_y],
        "truth": [sv(s) for s in truth],
        "measurements": [[float(x) for x in m.state_vector.ravel()] for m in measurements],
        "track": [sv(s) for s in track],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print("wrote", OUT, "sha256", hashlib.sha256(text.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
