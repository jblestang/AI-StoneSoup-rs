"""État d'une piste individuelle (filtre Stone Soup)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import GaussianState

from radar_tracker.config import AppConfig
from radar_tracker.coordinates import spherical_to_cartesian
from radar_tracker.events import TrackLifeState
from radar_tracker.models import build_measurement_model


@dataclass
class Track:
    track_id: str
    state: GaussianState | None = None
    life: TrackLifeState = TrackLifeState.CANDIDATE
    score: float = 0.0
    missed_scans: int = 0
    age_scans: int = 0
    parent_track_id: str | None = None
    group_id: str | None = None
    position_history: list[np.ndarray] = field(default_factory=list)
    cov_history: list[np.ndarray] = field(default_factory=list)
    last_update_scan: str | None = None
    init_scan_count: int = 0

    def record_history(self) -> None:
        if self.state is None:
            return
        sv = self.state.state_vector.ravel()
        self.position_history.append(np.array([sv[0], sv[2], sv[4]]))
        self.cov_history.append(np.array(self.state.covar))


def init_track_from_detections(
    track_id: str,
    detections: list[tuple[datetime, np.ndarray]],
    cfg: AppConfig,
    measurement_model,
) -> Track | None:
    if len(detections) < 2:
        return None
    detections = sorted(detections, key=lambda x: x[0])
    p0 = []
    for _, vec in detections:
        v = np.asarray(vec, dtype=float).ravel()
        el, az, r = float(v[0]), float(v[1]), float(v[2])
        p0.append(spherical_to_cartesian(r, az, el, np.array(cfg.radar.position_m)))
    p0 = np.mean(p0, axis=0)
    dt = (detections[-1][0] - detections[0][0]).total_seconds()
    if dt <= 0:
        vel = np.zeros(3)
    else:
        v0 = np.asarray(detections[0][1], dtype=float).ravel()
        v1 = np.asarray(detections[-1][1], dtype=float).ravel()
        p_first = spherical_to_cartesian(v0[2], v0[1], v0[0], np.array(cfg.radar.position_m))
        p_last = spherical_to_cartesian(v1[2], v1[1], v1[0], np.array(cfg.radar.position_m))
        vel = (p_last - p_first) / dt
    sig_p = cfg.tracking.init_position_sigma_m**2
    sig_v = cfg.tracking.init_velocity_sigma_mps**2
    cov = np.diag([sig_p, sig_v, sig_p, sig_v, sig_p, sig_v])
    state = GaussianState(
        StateVector([[p0[0]], [vel[0]], [p0[1]], [vel[1]], [p0[2]], [vel[2]]]),
        CovarianceMatrix(cov),
        timestamp=detections[-1][0],
    )
    tr = Track(track_id=track_id, state=state, init_scan_count=len(detections))
    tr.record_history()
    return tr


def detection_from_record(rec, measurement_model) -> Detection:
    from stonesoup.types.array import StateVector

    return Detection(
        StateVector(rec.stonesoup_vector),
        timestamp=rec.timestamp,
        measurement_model=measurement_model,
    )
