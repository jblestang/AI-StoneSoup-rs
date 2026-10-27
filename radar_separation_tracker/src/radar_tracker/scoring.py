"""Scores de vraisemblance et origine commune (non calibrés en probabilité)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import multivariate_normal

from radar_tracker.coordinates import wrap_angle_rad
from radar_tracker.config import AppConfig


def innovation_log_likelihood(
    innovation: np.ndarray,
    covar: np.ndarray,
    angle_indices: tuple[int, ...] = (0, 1),
) -> float:
    inn = innovation.ravel().astype(float)
    for idx in angle_indices:
        inn[idx] = wrap_angle_rad(inn[idx])
    try:
        return float(multivariate_normal.logpdf(inn, cov=np.asarray(covar, dtype=float)))
    except np.linalg.LinAlgError:
        return -1e6


def missed_detection_log_score(cfg: AppConfig, n_miss: int) -> float:
    pd = cfg.tracking.probability_detection
    if n_miss <= 0:
        return 0.0
    return n_miss * math.log(max(1e-12, 1.0 - pd))


def detection_log_score(cfg: AppConfig, n_det: int) -> float:
    pd = cfg.tracking.probability_detection
    if n_det <= 0:
        return 0.0
    return n_det * math.log(max(1e-12, pd))


def clutter_log_score(cfg: AppConfig, n_clutter: int, gate_volume_m3: float) -> float:
    if n_clutter <= 0:
        return 0.0
    lam = cfg.tracking.clutter_density_per_m3 * max(gate_volume_m3, 1.0)
    return n_clutter * math.log(max(1e-12, lam))


@dataclass
class CommonOriginResult:
    score: float
    compatible: bool
    details: str


def common_origin_score(
    parent_pos_history: list[np.ndarray],
    child_pos_history: list[np.ndarray],
    parent_covs: list[np.ndarray],
    child_covs: list[np.ndarray],
    cfg: AppConfig,
) -> CommonOriginResult:
    """Compare rétrospectivement tubes d'incertitude (approximation ellipsoïde diagonale)."""
    if len(parent_pos_history) < 2 or len(child_pos_history) < 2:
        return CommonOriginResult(0.0, False, "historique insuffisant")
    n = min(
        len(parent_pos_history),
        len(child_pos_history),
        cfg.hypothesis.common_origin_lookback_scans,
    )
    mult = cfg.hypothesis.common_origin_sigma_multiplier
    min_dist = float("inf")
    for i in range(-n, 0):
        pp = parent_pos_history[i]
        cp = child_pos_history[i]
        pc = np.diag(parent_covs[i])[[0, 2, 4]] if parent_covs[i].shape[0] >= 6 else np.ones(3) * 100
        cc = np.diag(child_covs[i])[[0, 2, 4]] if child_covs[i].shape[0] >= 6 else np.ones(3) * 100
        thresh = mult * (np.sqrt(pc) + np.sqrt(cc))
        d = np.abs(pp - cp)
        sep = float(np.max(d / np.maximum(thresh, 1.0)))
        min_dist = min(min_dist, sep)
    compatible = min_dist < 1.0
    score = 4.0 * (1.0 - min(1.0, min_dist)) if compatible else -2.0 * min_dist
    return CommonOriginResult(
        score=score,
        compatible=compatible,
        details=f"min_normalised_sep={min_dist:.3f}",
    )
