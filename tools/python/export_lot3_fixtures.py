#!/usr/bin/env python3
"""Export Tutorial 05 (clutter + nearest neighbour) fixtures from Stone Soup v1.9.1."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

UPSTREAM_SHA = "a4336b920a799cfe0a77ecb05867c5deeb371c7a"
OUT = Path(__file__).resolve().parents[2] / "verification" / "fixtures" / "lot3_clutter.json"
# Fixed clock so JSON metadata is stable across machines (dynamics do not depend on it).
START_TIME = datetime(2020, 1, 1, 12, 0, 0)


def main() -> int:
    import numpy as np
    from scipy.stats import uniform

    from stonesoup.dataassociator.neighbour import NearestNeighbour
    from stonesoup.hypothesiser.distance import DistanceHypothesiser
    from stonesoup.measures import Mahalanobis
    from stonesoup.models.measurement.linear import LinearGaussian
    from stonesoup.models.transition.linear import (
        CombinedLinearGaussianTransitionModel,
        ConstantVelocity,
    )
    from stonesoup.predictor.kalman import KalmanPredictor
    from stonesoup.types.detection import Clutter, TrueDetection
    from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
    from stonesoup.types.state import GaussianState
    from stonesoup.types.track import Track
    from stonesoup.updater.kalman import KalmanUpdater

    np.random.seed(1991)
    q_x, q_y = 0.005, 0.005
    prob_det = 0.9
    gate = 3.0

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(q_x), ConstantVelocity(q_y)]
    )
    measurement_model = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.array([[0.75, 0.0], [0.0, 0.75]]),
    )

    timesteps = [START_TIME]
    truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=timesteps[0])])
    num_steps = 21
    for k in range(1, num_steps):
        timesteps.append(START_TIME + timedelta(seconds=k))
        truth.append(
            GroundTruthState(
                transition_model.function(
                    truth[k - 1], noise=True, time_interval=timedelta(seconds=1)
                ),
                timestamp=timesteps[k],
            )
        )

    all_measurements = []
    for state in truth:
        measurement_set = set()
        if np.random.rand() <= prob_det:
            measurement = measurement_model.function(state, noise=True)
            measurement_set.add(
                TrueDetection(
                    state_vector=measurement,
                    groundtruth_path=truth,
                    timestamp=state.timestamp,
                    measurement_model=measurement_model,
                )
            )
        truth_x = float(state.state_vector[0])
        truth_y = float(state.state_vector[2])
        for _ in range(np.random.randint(10)):
            x = float(uniform.rvs(truth_x - 10, 20))
            y = float(uniform.rvs(truth_y - 10, 20))
            measurement_set.add(
                Clutter(
                    np.array([[x], [y]]),
                    timestamp=state.timestamp,
                    measurement_model=measurement_model,
                )
            )
        all_measurements.append(measurement_set)

    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=Mahalanobis(), missed_distance=gate
    )
    data_associator = NearestNeighbour(hypothesiser)

    prior = GaussianState(
        [[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=START_TIME
    )
    track = Track([prior])
    for n, measurements in enumerate(all_measurements):
        hypotheses = data_associator.associate(
            [track], measurements, START_TIME + timedelta(seconds=n)
        )
        hypothesis = hypotheses[track]
        if hypothesis.measurement:
            post = updater.update(hypothesis)
            track.append(post)
        else:
            track.append(hypothesis.prediction)

    def sv(state) -> list[float]:
        return [float(x) for x in state.state_vector]

    def det_row(d) -> dict:
        kind = "true" if isinstance(d, TrueDetection) else "clutter"
        xy = d.state_vector.ravel().tolist()
        return {"xy": [float(xy[0]), float(xy[1])], "kind": kind}

    meas_sets = []
    for i, mset in enumerate(all_measurements):
        # Stable order for Rust embedding (sort by x, y).
        rows = sorted((det_row(d) for d in mset), key=lambda r: (r["xy"][0], r["xy"][1]))
        meas_sets.append({"step": i, "measurements": rows})

    # Posterior / prediction states after each association step (21 steps).
    track_states = [sv(s) for s in track[1:]]

    payload = {
        "upstream_sha": UPSTREAM_SHA,
        "seed": 1991,
        "start_time_iso": START_TIME.isoformat(),
        "q_x": q_x,
        "q_y": q_y,
        "prob_det": prob_det,
        "gate_mahalanobis": gate,
        "measurement_noise_diag": [0.75, 0.75],
        "prior_mean": [0.0, 1.0, 0.0, 1.0],
        "prior_cov_diag": [1.5, 0.5, 1.5, 0.5],
        "truth": [sv(s) for s in truth],
        "measurement_sets": meas_sets,
        "track": track_states,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print("wrote", OUT, "sha256", hashlib.sha256(text.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
