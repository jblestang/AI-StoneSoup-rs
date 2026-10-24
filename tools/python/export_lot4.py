#!/usr/bin/env python3
"""Export Lot 4 tutorials 09 (initiators/deleters) and 10 (simulation stack) reference tracks."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

UPSTREAM_SHA = "a4336b920a799cfe0a77ecb05867c5deeb371c7a"
ROOT = Path(__file__).resolve().parents[2]
OUT09 = ROOT / "verification" / "fixtures" / "lot4_tutorial09.json"
OUT10 = ROOT / "verification" / "fixtures" / "lot4_tutorial10.json"
def track_means(tracks) -> list[list[float]]:
    rows = []
    for t in tracks:
        s = t.states[-1]
        rows.append([float(s.state_vector[i, 0]) for i in range(4)])
    rows.sort(key=lambda r: (r[0], r[2]))
    return rows


def export_09() -> dict:
    import numpy as np
    from ordered_set import OrderedSet
    from scipy.stats import uniform
    from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
    from stonesoup.deleter.error import CovarianceBasedDeleter
    from stonesoup.hypothesiser.distance import DistanceHypothesiser
    from stonesoup.initiator.simple import MultiMeasurementInitiator
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
    from stonesoup.updater.kalman import KalmanUpdater

    np.random.seed(1991)
    start_time = datetime.now().replace(microsecond=0)
    truths = OrderedSet()
    current_truths = set()
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    timesteps = []
    for k in range(20):
        timesteps.append(start_time + timedelta(seconds=k))
        for truth in current_truths.copy():
            if np.random.rand() <= 0.05:
                current_truths.remove(truth)
        for truth in current_truths:
            truth.append(
                GroundTruthState(
                    transition_model.function(
                        truth[-1], noise=True, time_interval=timedelta(seconds=1)
                    ),
                    timestamp=timesteps[k],
                )
            )
        for _ in range(np.random.poisson(0.6)):
            x, y = np.random.rand(2) * [20, 20]
            x_vel, y_vel = np.random.rand(2) * 2 - 1
            state = GroundTruthState([x, x_vel, y, y_vel], timestamp=timesteps[k])
            truth = GroundTruthPath([state])
            current_truths.add(truth)
            truths.add(truth)

    measurement_model = LinearGaussian(
        4, [0, 2], np.array([[0.25, 0], [0, 0.25]])
    )
    all_measurements = []
    for k in range(20):
        measurement_set = set()
        timestamp = start_time + timedelta(seconds=k)
        for truth in truths:
            try:
                truth_state = truth[timestamp]
            except IndexError:
                continue
            if np.random.rand() <= 0.9:
                measurement = measurement_model.function(truth_state, noise=True)
                measurement_set.add(
                    TrueDetection(
                        state_vector=measurement,
                        groundtruth_path=truth,
                        timestamp=truth_state.timestamp,
                        measurement_model=measurement_model,
                    )
                )
            truth_x = truth_state.state_vector[0, 0]
            truth_y = truth_state.state_vector[2, 0]
            for _ in range(np.random.randint(2)):
                x = uniform.rvs(truth_x - 10, 20)
                y = uniform.rvs(truth_y - 10, 20)
                measurement_set.add(
                    Clutter(
                        np.array([[x], [y]]),
                        timestamp=timestamp,
                        measurement_model=measurement_model,
                    )
                )
        all_measurements.append(measurement_set)

    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=Mahalanobis(), missed_distance=3
    )
    data_associator = GNNWith2DAssignment(hypothesiser)
    deleter = CovarianceBasedDeleter(covar_trace_thresh=4)
    initiator = MultiMeasurementInitiator(
        prior_state=GaussianState([[0], [0], [0], [0]], np.diag([0, 1, 0, 1])),
        measurement_model=measurement_model,
        deleter=deleter,
        data_associator=data_associator,
        updater=updater,
        min_points=2,
    )

    tracks, all_tracks = set(), set()
    snapshots = []
    meas_flat = []
    offsets = [0]
    for n, measurements in enumerate(all_measurements):
        for m in measurements:
            meas_flat.append(
                [float(m.state_vector[0, 0]), float(m.state_vector[1, 0])]
            )
        offsets.append(len(meas_flat))
        hypotheses = data_associator.associate(
            tracks, measurements, start_time + timedelta(seconds=n)
        )
        associated_measurements = set()
        for track in tracks:
            hypothesis = hypotheses[track]
            if hypothesis.measurement:
                post = updater.update(hypothesis)
                track.append(post)
                associated_measurements.add(hypothesis.measurement)
            else:
                track.append(hypothesis.prediction)
        tracks -= deleter.delete_tracks(tracks)
        tracks |= initiator.initiate(
            measurements - associated_measurements, start_time + timedelta(seconds=n)
        )
        all_tracks |= tracks
        snapshots.append({"track_means": track_means(tracks)})

    return {
        "upstream_sha": UPSTREAM_SHA,
        "seed": 1991,
        "num_steps": 20,
        "gate": 3.0,
        "deleter_trace": 4.0,
        "min_points": 2,
        "process_noise": 0.005,
        "meas_var": 0.25,
        "meas_offsets": offsets,
        "meas_xy": meas_flat,
        "snapshots": snapshots,
        "final_track_means": track_means(all_tracks),
    }


def export_10() -> dict:
    import numpy as np
    from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
    from stonesoup.deleter.error import CovarianceBasedDeleter
    from stonesoup.hypothesiser.distance import DistanceHypothesiser
    from stonesoup.initiator.simple import MultiMeasurementInitiator
    from stonesoup.measures import Mahalanobis
    from stonesoup.models.measurement.linear import LinearGaussian
    from stonesoup.models.transition.linear import (
        CombinedLinearGaussianTransitionModel,
        ConstantVelocity,
    )
    from stonesoup.predictor.kalman import KalmanPredictor
    from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator, SimpleDetectionSimulator
    from stonesoup.tracker.simple import MultiTargetTracker
    from stonesoup.types.array import CovarianceMatrix, StateVector
    from stonesoup.types.state import GaussianState
    from stonesoup.updater.kalman import KalmanUpdater

    np.random.seed(1991)
    start_time = datetime.now().replace(microsecond=0)
    timestep_size = timedelta(seconds=5)
    number_of_steps = 20
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.05), ConstantVelocity(0.05)]
    )
    initial_state = GaussianState(
        StateVector([[0], [0], [0], [0]]),
        CovarianceMatrix(np.diag([4, 0.5, 4, 0.5])),
        start_time,
    )
    groundtruth_sim = MultiTargetGroundTruthSimulator(
        transition_model=transition_model,
        initial_state=initial_state,
        timestep=timestep_size,
        number_steps=number_of_steps,
        birth_rate=0.3,
        death_probability=0.05,
    )
    measurement_model = LinearGaussian(4, [0, 2], np.diag([0.25, 0.25]))
    detection_sim = SimpleDetectionSimulator(
        groundtruth=groundtruth_sim,
        measurement_model=measurement_model,
        detection_probability=0.9,
        meas_range=np.array([[-1, 1], [-1, 1]]) * 30,
        clutter_rate=1,
    )
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=Mahalanobis(), missed_distance=3
    )
    data_associator = GNNWith2DAssignment(hypothesiser)
    deleter = CovarianceBasedDeleter(covar_trace_thresh=2)
    initiator = MultiMeasurementInitiator(
        prior_state=GaussianState([[0], [0], [0], [0]], np.diag([0, 0.5, 0, 0.5])),
        measurement_model=measurement_model,
        deleter=deleter,
        data_associator=data_associator,
        updater=updater,
        min_points=3,
    )
    tracker = MultiTargetTracker(
        initiator=initiator,
        deleter=deleter,
        detector=detection_sim,
        data_associator=data_associator,
        updater=updater,
    )

    snapshots = []
    meas_flat = []
    offsets = [0]
    all_tracks = set()
    for time, tracks in tracker:
        step_dets = [d for d in detection_sim.detections if d.timestamp == time]
        for d in step_dets:
            meas_flat.append([float(d.state_vector[0, 0]), float(d.state_vector[1, 0])])
        offsets.append(len(meas_flat))
        snapshots.append({"track_means": track_means(tracks)})
        all_tracks |= tracks

    return {
        "upstream_sha": UPSTREAM_SHA,
        "seed": 1991,
        "num_steps": number_of_steps,
        "timestep_seconds": 5,
        "gate": 3.0,
        "deleter_trace": 2.0,
        "min_points": 3,
        "process_noise": 0.05,
        "meas_var": 0.25,
        "meas_offsets": offsets,
        "meas_xy": meas_flat,
        "snapshots": snapshots,
        "final_track_means": track_means(all_tracks),
    }


def main() -> int:
    p09 = export_09()
    p10 = export_10()
    OUT09.write_text(json.dumps(p09, indent=2) + "\n", encoding="utf-8")
    OUT10.write_text(json.dumps(p10, indent=2) + "\n", encoding="utf-8")
    import subprocess

    subprocess.run(
        [sys.executable, str(ROOT / "tools/python/lot4_fixtures_to_rust.py")],
        check=True,
    )
    for path in (OUT09, OUT10):
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        print("wrote", path, "sha256", h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
