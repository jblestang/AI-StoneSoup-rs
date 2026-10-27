"""Modèles Stone Soup (mesure + transition + filtre)."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.predictor.kalman import ExtendedKalmanPredictor, UnscentedKalmanPredictor
from stonesoup.types.array import CovarianceMatrix
from stonesoup.updater.kalman import ExtendedKalmanUpdater, UnscentedKalmanUpdater

from radar_tracker.config import AppConfig


def build_measurement_model(cfg: AppConfig) -> CartesianToElevationBearingRange:
    m = cfg.measurement
    noise = np.diag(
        [
            m.sigma_elevation_rad**2,
            m.sigma_azimuth_rad**2,
            m.sigma_range_m**2,
        ]
    )
    return CartesianToElevationBearingRange(
        ndim_state=6,
        mapping=(0, 2, 4),
        noise_covar=CovarianceMatrix(noise),
        rotation_offset=np.array(cfg.radar.rotation_offset_rad, dtype=float).reshape(3, 1),
    )


def build_transition_model(cfg: AppConfig) -> CombinedLinearGaussianTransitionModel:
    cv = ConstantVelocity(cfg.tracking.process_noise_accel_mps2)
    return CombinedLinearGaussianTransitionModel([cv, cv, cv])


def build_predictor_updater(cfg: AppConfig, measurement_model: CartesianToElevationBearingRange):
    trans = build_transition_model(cfg)
    if cfg.tracking.filter.lower() == "ukf":
        return (
            UnscentedKalmanPredictor(trans),
            UnscentedKalmanUpdater(measurement_model=measurement_model),
        )
    return (
        ExtendedKalmanPredictor(trans),
        ExtendedKalmanUpdater(measurement_model=measurement_model),
    )


def gate_threshold(cfg: AppConfig, ndim: int = 3) -> float:
    return float(chi2.ppf(cfg.tracking.gate_probability, ndim))
