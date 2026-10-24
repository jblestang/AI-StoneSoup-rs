#!/usr/bin/env python3
"""Export Tutorials 06–08 fixtures (GNN, PDA, JPDA) from Stone Soup v1.9.1."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

UPSTREAM_SHA = "a4336b920a799cfe0a77ecb05867c5deeb371c7a"
ROOT = Path(__file__).resolve().parents[2]
START_TIME = datetime(2020, 1, 1, 12, 0, 0)


def sv(state) -> list[float]:
    return [float(x) for x in state.state_vector]


def det_row(d) -> dict:
    from stonesoup.types.detection import TrueDetection

    kind = "true" if isinstance(d, TrueDetection) else "clutter"
    xy = d.state_vector.ravel().tolist()
    return {"xy": [float(xy[0]), float(xy[1])], "kind": kind}


def simulate_single_target_measurements(seed: int = 1991) -> tuple[list, list]:
    import numpy as np
    from scipy.stats import uniform

    from stonesoup.models.measurement.linear import LinearGaussian
    from stonesoup.models.transition.linear import (
        CombinedLinearGaussianTransitionModel,
        ConstantVelocity,
    )
    from stonesoup.types.detection import Clutter, TrueDetection
    from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

    np.random.seed(seed)
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    timesteps = [START_TIME]
    truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=timesteps[0])])
    for k in range(1, 21):
        timesteps.append(START_TIME + timedelta(seconds=k))
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
        noise_covar=np.array([[0.75, 0.0], [0.0, 0.75]]),
    )
    prob_det = 0.9
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
    return truth, all_measurements


def simulate_multi_target_measurements(seed: int = 1991) -> tuple[set, list]:
    import numpy as np
    from ordered_set import OrderedSet
    from scipy.stats import uniform

    from stonesoup.models.measurement.linear import LinearGaussian
    from stonesoup.models.transition.linear import (
        CombinedLinearGaussianTransitionModel,
        ConstantVelocity,
    )
    from stonesoup.types.detection import Clutter, TrueDetection
    from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

    np.random.seed(seed)
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    timesteps = [START_TIME]
    truth_a = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=timesteps[0])])
    for k in range(1, 21):
        timesteps.append(START_TIME + timedelta(seconds=k))
        truth_a.append(
            GroundTruthState(
                transition_model.function(
                    truth_a[k - 1], noise=True, time_interval=timedelta(seconds=1)
                ),
                timestamp=timesteps[k],
            )
        )
    truths = OrderedSet([truth_a])
    truth_b = GroundTruthPath([GroundTruthState([0, 1, 20, -1], timestamp=timesteps[0])])
    for k in range(1, 21):
        truth_b.append(
            GroundTruthState(
                transition_model.function(
                    truth_b[k - 1], noise=True, time_interval=timedelta(seconds=1)
                ),
                timestamp=timesteps[k],
            )
        )
    truths.add(truth_b)
    measurement_model = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.array([[0.75, 0.0], [0.0, 0.75]]),
    )
    all_measurements = []
    for k in range(20):
        measurement_set = set()
        for truth in truths:
            if np.random.rand() <= 0.9:
                measurement = measurement_model.function(truth[k], noise=True)
                measurement_set.add(
                    TrueDetection(
                        state_vector=measurement,
                        groundtruth_path=truth,
                        timestamp=truth[k].timestamp,
                        measurement_model=measurement_model,
                    )
                )
            truth_x = float(truth[k].state_vector[0])
            truth_y = float(truth[k].state_vector[2])
            for _ in range(np.random.randint(10)):
                x = float(uniform.rvs(truth_x - 10, 20))
                y = float(uniform.rvs(truth_y - 10, 20))
                measurement_set.add(
                    Clutter(
                        np.array([[x], [y]]),
                        timestamp=truth[k].timestamp,
                        measurement_model=measurement_model,
                    )
                )
        all_measurements.append(measurement_set)
    return truths, all_measurements


def meas_sets_json(all_measurements) -> list:
    out = []
    for i, mset in enumerate(all_measurements):
        rows = sorted((det_row(d) for d in mset), key=lambda r: (r["xy"][0], r["xy"][1]))
        out.append({"step": i, "measurements": rows})
    return out


def export_gnn() -> None:
    from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
    from stonesoup.hypothesiser.distance import DistanceHypothesiser
    from stonesoup.measures import Mahalanobis
    from stonesoup.models.transition.linear import (
        CombinedLinearGaussianTransitionModel,
        ConstantVelocity,
    )
    from stonesoup.predictor.kalman import KalmanPredictor
    from stonesoup.types.state import GaussianState
    from stonesoup.types.track import Track
    from stonesoup.updater.kalman import KalmanUpdater
    from stonesoup.models.measurement.linear import LinearGaussian
    import numpy as np

    truths, all_measurements = simulate_multi_target_measurements()
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    measurement_model = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.array([[0.75, 0.0], [0.0, 0.75]]),
    )
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=Mahalanobis(), missed_distance=3
    )
    data_associator = GlobalNearestNeighbour(hypothesiser)
    prior1 = GaussianState(
        [[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=START_TIME
    )
    prior2 = GaussianState(
        [[0], [1], [20], [-1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=START_TIME
    )
    track_low = Track([prior1])
    track_high = Track([prior2])
    tracks = {track_low, track_high}
    for n, measurements in enumerate(all_measurements):
        hypotheses = data_associator.associate(
            tracks, measurements, START_TIME + timedelta(seconds=n)
        )
        for track in tracks:
            hypothesis = hypotheses[track]
            if hypothesis.measurement:
                track.append(updater.update(hypothesis))
            else:
                track.append(hypothesis.prediction)
    truth_list = list(truths)
    payload = {
        "upstream_sha": UPSTREAM_SHA,
        "seed": 1991,
        "num_steps": 20,
        "gate_mahalanobis": 3.0,
        "measurement_sets": meas_sets_json(all_measurements),
        "truth_a": [sv(s) for s in truth_list[0]],
        "truth_b": [sv(s) for s in truth_list[1]],
        "track_a": [sv(s) for s in track_low[1:]],
        "track_b": [sv(s) for s in track_high[1:]],
    }
    write_json("lot3_multitarget.json", payload)


def export_pda() -> None:
    from stonesoup.dataassociator.probability import PDA
    from stonesoup.functions import gm_reduce_single
    from stonesoup.hypothesiser.probability import PDAHypothesiser
    from stonesoup.models.measurement.linear import LinearGaussian
    from stonesoup.models.transition.linear import (
        CombinedLinearGaussianTransitionModel,
        ConstantVelocity,
    )
    from stonesoup.predictor.kalman import KalmanPredictor
    from stonesoup.types.array import StateVectors
    from stonesoup.types.state import GaussianState
    from stonesoup.types.track import Track
    from stonesoup.types.update import GaussianStateUpdate
    from stonesoup.updater.kalman import KalmanUpdater
    import numpy as np

    truth, all_measurements = simulate_single_target_measurements()
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    measurement_model = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.array([[0.75, 0.0], [0.0, 0.75]]),
    )
    prob_detect = 0.9
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    hypothesiser = PDAHypothesiser(
        predictor=predictor,
        updater=updater,
        clutter_spatial_density=0.125,
        prob_detect=prob_detect,
    )
    data_associator = PDA(hypothesiser=hypothesiser)
    prior = GaussianState(
        [[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=START_TIME
    )
    track = Track([prior])
    for n, measurements in enumerate(all_measurements):
        hypotheses = data_associator.associate({track}, measurements, START_TIME + timedelta(seconds=n))
        hypotheses = hypotheses[track]
        posterior_states = []
        posterior_state_weights = []
        for hypothesis in hypotheses:
            if not hypothesis:
                posterior_states.append(hypothesis.prediction)
            else:
                posterior_states.append(updater.update(hypothesis))
            posterior_state_weights.append(hypothesis.probability)
        means = StateVectors([state.state_vector for state in posterior_states])
        covars = np.stack([state.covar for state in posterior_states], axis=2)
        weights = np.asarray(posterior_state_weights)
        post_mean, post_covar = gm_reduce_single(means, covars, weights)
        track.append(
            GaussianStateUpdate(
                post_mean,
                post_covar,
                hypotheses,
                hypotheses[0].measurement.timestamp,
            )
        )
    payload = {
        "upstream_sha": UPSTREAM_SHA,
        "seed": 1991,
        "num_steps": len(all_measurements),
        "prob_detect": prob_detect,
        "clutter_spatial_density": 0.125,
        "measurement_sets": meas_sets_json(all_measurements),
        "truth": [sv(s) for s in truth],
        "track": [sv(s) for s in track[1:]],
    }
    write_json("lot3_pda.json", payload)


def export_jpda() -> None:
    from stonesoup.dataassociator.probability import JPDA
    from stonesoup.functions import gm_reduce_single
    from stonesoup.hypothesiser.probability import PDAHypothesiser
    from stonesoup.models.measurement.linear import LinearGaussian
    from stonesoup.models.transition.linear import (
        CombinedLinearGaussianTransitionModel,
        ConstantVelocity,
    )
    from stonesoup.predictor.kalman import KalmanPredictor
    from stonesoup.types.array import StateVectors
    from stonesoup.types.state import GaussianState
    from stonesoup.types.track import Track
    from stonesoup.types.update import GaussianStateUpdate
    from stonesoup.updater.kalman import KalmanUpdater
    import numpy as np

    truths, all_measurements = simulate_multi_target_measurements()
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    measurement_model = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.array([[0.75, 0.0], [0.0, 0.75]]),
    )
    prob_detect = 0.9
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    hypothesiser = PDAHypothesiser(
        predictor=predictor,
        updater=updater,
        clutter_spatial_density=0.125,
        prob_detect=prob_detect,
    )
    data_associator = JPDA(hypothesiser=hypothesiser)
    prior1 = GaussianState(
        [[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=START_TIME
    )
    prior2 = GaussianState(
        [[0], [1], [20], [-1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=START_TIME
    )
    track_low = Track([prior1])
    track_high = Track([prior2])
    tracks = {track_low, track_high}
    for n, measurements in enumerate(all_measurements):
        hypotheses = data_associator.associate(
            tracks, measurements, START_TIME + timedelta(seconds=n)
        )
        for track in tracks:
            track_hypotheses = hypotheses[track]
            posterior_states = []
            posterior_state_weights = []
            for hypothesis in track_hypotheses:
                if not hypothesis:
                    posterior_states.append(hypothesis.prediction)
                else:
                    posterior_states.append(updater.update(hypothesis))
                posterior_state_weights.append(hypothesis.probability)
            means = StateVectors([state.state_vector for state in posterior_states])
            covars = np.stack([state.covar for state in posterior_states], axis=2)
            weights = np.asarray(posterior_state_weights)
            post_mean, post_covar = gm_reduce_single(means, covars, weights)
            track.append(
                GaussianStateUpdate(
                    post_mean,
                    post_covar,
                    track_hypotheses,
                    track_hypotheses[0].measurement.timestamp,
                )
            )
    truth_list = list(truths)
    payload = {
        "upstream_sha": UPSTREAM_SHA,
        "seed": 1991,
        "num_steps": 20,
        "prob_detect": prob_detect,
        "clutter_spatial_density": 0.125,
        "measurement_sets": meas_sets_json(all_measurements),
        "truth_a": [sv(s) for s in truth_list[0]],
        "truth_b": [sv(s) for s in truth_list[1]],
        "track_a": [sv(s) for s in track_low[1:]],
        "track_b": [sv(s) for s in track_high[1:]],
    }
    write_json("lot3_jpda.json", payload)


def write_json(name: str, payload: dict) -> None:
    path = ROOT / "verification" / "fixtures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")
    print("wrote", path, "sha256", hashlib.sha256(text.encode()).hexdigest())


def main() -> int:
    export_gnn()
    export_pda()
    export_jpda()
    return 0


if __name__ == "__main__":
    sys.exit(main())
