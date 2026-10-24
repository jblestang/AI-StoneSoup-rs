#!/usr/bin/env python3
"""
Two co-located rotating radars with clutter — 250 plots per radar, track statistics.

Radar A: 6 s rotation, 400 km range (10 RPM)
Radar B: 3 s rotation, 200 km range (~25% area overlap with A when co-located)
Clutter: Poisson rate tuned for ~10% of all plots

Writes verification/reports/dual_radar_clutter_stats.json
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "verification" / "reports" / "dual_radar_clutter_stats.json"

PLOTS_PER_RADAR = 250
CLUTTER_FRACTION_TARGET = 0.10
OVERLAP_FRACTION = 0.20
RANGE_A_M = 400e3
RANGE_B_M = 200e3
PERIOD_A_S = 6.0
PERIOD_B_S = 3.0
RPM_A = 60.0 / PERIOD_A_S
RPM_B = 60.0 / PERIOD_B_S
FOV_DEG = 90.0
DT_S = 0.25
NUM_TRUTHS = 4
CLUTTER_RATE = 0.52
SEED = 42


def track_duration_s(track) -> float:
    if len(track) < 2:
        return 0.0
    return (track[-1].timestamp - track[0].timestamp).total_seconds()


def main() -> int:
    from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
    from stonesoup.dataassociator.tracktotrack import TrackToTruth
    from stonesoup.deleter.time import UpdateTimeDeleter
    from stonesoup.hypothesiser.distance import DistanceHypothesiser
    from stonesoup.initiator.simple import MultiMeasurementInitiator
    from stonesoup.measures import Mahalanobis
    from stonesoup.models.clutter.clutter import ClutterModel
    from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
    from stonesoup.models.transition.linear import (
        CombinedLinearGaussianTransitionModel,
        ConstantVelocity,
    )
    from stonesoup.platform import FixedPlatform
    from stonesoup.predictor.kalman import ExtendedKalmanPredictor
    from stonesoup.sensor.radar.radar import RadarRotatingBearingRange
    from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
    from stonesoup.tracker.simple import MultiTargetTracker
    from stonesoup.types.angle import Angle
    from stonesoup.types.array import StateVector
    from stonesoup.types.state import GaussianState, State
    from stonesoup.types.detection import TrueDetection
    from stonesoup.updater.kalman import ExtendedKalmanUpdater

    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    start = datetime(2026, 1, 1)
    transition = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.5), ConstantVelocity(0.5)]
    )

    truths_sim = MultiTargetGroundTruthSimulator(
        transition_model=transition,
        birth_rate=0.0,
        death_probability=0.0,
        initial_state=GaussianState(
            StateVector([[0], [0], [0], [0]]),
            np.diag([1.0, 0.1, 1.0, 0.1]),
            timestamp=start,
        ),
        preexisting_states=[
            StateVector([r * math.cos(a), rng.uniform(-6, 6), r * math.sin(a), rng.uniform(-6, 6)])
            for r, a in (
                (80e3, 0.0),
                (80e3, math.pi / 2),
                (80e3, math.pi),
                (80e3, 3 * math.pi / 2),
            )
        ],
        number_steps=50_000,
        timestep=timedelta(seconds=DT_S),
    )

    def clutter_dist(low, high):
        return rng.uniform(low, high)

    def make_clutter(max_range: float) -> ClutterModel:
        lim = max_range * 0.95
        return ClutterModel(
            clutter_rate=CLUTTER_RATE,
            dist_params=((-lim, lim), (-lim, lim)),
            distribution=clutter_dist,
            seed=SEED,
        )

    def make_radar(rpm: float, max_range: float) -> RadarRotatingBearingRange:
        dwell_step = 2.0 * math.pi * rpm * (DT_S / 60.0)
        radar = RadarRotatingBearingRange(
            ndim_state=4,
            position_mapping=(0, 2),
            noise_covar=np.diag([np.radians(0.5) ** 2, (80.0) ** 2]),
            rpm=rpm,
            max_range=max_range,
            fov_angle=np.radians(FOV_DEG),
            dwell_centre=StateVector([0.0]),
            resolution=Angle(dwell_step),
            clutter_model=make_clutter(max_range),
        )
        radar.timestamp = start
        return radar

    radar_a = make_radar(RPM_A, RANGE_A_M)
    radar_b = make_radar(RPM_B, RANGE_B_M)
    sensor_key = {radar_a: "radar_a", radar_b: "radar_b"}
    platform = FixedPlatform(
        State(StateVector([0, 0, 0, 0]), timestamp=start),
        position_mapping=(0, 2),
        sensors=[radar_a, radar_b],
    )

    meas_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.diag([np.radians(0.5) ** 2, 80.0**2]),
    )
    updater = ExtendedKalmanUpdater(meas_model)
    predictor = ExtendedKalmanPredictor(
        CombinedLinearGaussianTransitionModel(
            [ConstantVelocity(0.5), ConstantVelocity(0.5)]
        )
    )
    hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=Mahalanobis(), missed_distance=6.0
    )
    data_associator = GNNWith2DAssignment(hypothesiser)
    deleter = UpdateTimeDeleter(timedelta(seconds=8), delete_last_pred=True)
    prior = GaussianState(
        StateVector([0, 0, 0, 0]), np.diag([1e6, 10, 1e6, 10]), timestamp=start
    )
    initiator = MultiMeasurementInitiator(
        prior_state=prior,
        measurement_model=None,
        deleter=deleter,
        data_associator=data_associator,
        updater=updater,
        min_points=3,
    )

    plot_counts = {"radar_a": 0, "radar_b": 0}
    total_true = 0
    total_clutter = 0
    truths_detected = set()

    class PlotBudgetDetector:
        def __iter__(self):
            self._it = iter(truths_sim)
            return self

        def __next__(self):
            nonlocal total_true, total_clutter, truths_detected
            if (
                plot_counts["radar_a"] >= PLOTS_PER_RADAR
                and plot_counts["radar_b"] >= PLOTS_PER_RADAR
            ):
                raise StopIteration
            time, truth_states = next(self._it)
            platform.timestamp = time
            merged = set()
            for sensor in platform.sensors:
                key = sensor_key[sensor]
                if plot_counts[key] >= PLOTS_PER_RADAR:
                    continue
                sensor.act(time)
                dets = sensor.measure(set(truth_states), noise=True)
                budget = PLOTS_PER_RADAR - plot_counts[key]
                batch = list(dets)[:budget]
                plot_counts[key] += len(batch)
                for d in batch:
                    if isinstance(d, TrueDetection):
                        total_true += 1
                        truths_detected.add(d.groundtruth_path)
                    else:
                        total_clutter += 1
                merged.update(batch)
            return time, merged

    tracker = MultiTargetTracker(
        initiator=initiator,
        deleter=deleter,
        detector=PlotBudgetDetector(),
        data_associator=data_associator,
        updater=updater,
    )

    all_tracks = set()
    steps = 0
    for _t, current in tracker:
        all_tracks.update(current)
        steps += 1

    truth_path_set = truths_sim.groundtruth_paths
    associator = TrackToTruth(association_threshold=2_000.0)
    association_set = associator.associate_tracks(all_tracks, truth_path_set)
    associated_tracks = {assoc.objects[0] for assoc in association_set}
    associated_truths = {assoc.objects[1] for assoc in association_set}

    true_tracks = [t for t in all_tracks if t in associated_tracks]
    false_tracks = [t for t in all_tracks if t not in associated_tracks]

    dur_true = [track_duration_s(t) for t in true_tracks]
    dur_false = [track_duration_s(t) for t in false_tracks]
    dur_all = [track_duration_s(t) for t in all_tracks]
    total_plots = total_true + total_clutter

    summary = {
        "parameters": {
            "plots_per_radar": PLOTS_PER_RADAR,
            "clutter_fraction_target": CLUTTER_FRACTION_TARGET,
            "clutter_rate_poisson": CLUTTER_RATE,
            "overlap_fraction_area_nominal": OVERLAP_FRACTION,
            "overlap_area_fraction_co_located_disks": (RANGE_B_M / RANGE_A_M) ** 2,
            "range_a_km": RANGE_A_M / 1e3,
            "range_b_km": RANGE_B_M / 1e3,
            "period_a_s": PERIOD_A_S,
            "period_b_s": PERIOD_B_S,
            "rpm_a": RPM_A,
            "rpm_b": RPM_B,
            "fov_deg": FOV_DEG,
            "dt_s": DT_S,
            "num_truths": NUM_TRUTHS,
            "truth_layout": "four targets at 80 km on cardinal bearings",
            "seed": SEED,
        },
        "plots": {
            "radar_a_plots": plot_counts["radar_a"],
            "radar_b_plots": plot_counts["radar_b"],
            "total_true_detections": total_true,
            "total_clutter_detections": total_clutter,
            "clutter_fraction_actual": total_clutter / total_plots if total_plots else 0.0,
            "simulation_steps": steps,
        },
        "truths": {
            "simulated": len(truth_path_set),
            "with_any_plot": len(truths_detected),
            "with_confirmed_track": len(associated_truths),
        },
        "tracks": {
            "total_unique_tracks": len(all_tracks),
            "truth_associated_tracks": len(true_tracks),
            "false_tracks": len(false_tracks),
            "mean_duration_s_all": float(np.mean(dur_all)) if dur_all else 0.0,
            "mean_duration_s_true": float(np.mean(dur_true)) if dur_true else 0.0,
            "mean_duration_s_false": float(np.mean(dur_false)) if dur_false else 0.0,
            "median_duration_s_true": float(np.median(dur_true)) if dur_true else 0.0,
            "median_duration_s_false": float(np.median(dur_false)) if dur_false else 0.0,
            "max_duration_s": max(dur_all, default=0.0),
            "min_duration_s": min(dur_all, default=0.0),
            "durations_s_true": dur_true,
            "durations_s_false": dur_false,
        },
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
