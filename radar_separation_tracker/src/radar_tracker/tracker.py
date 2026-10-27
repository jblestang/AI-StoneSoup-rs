"""Tracker multi-hypothèses principal."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy.stats import chi2
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import GaussianState

from radar_tracker.config import AppConfig
from radar_tracker.coordinates import spherical_to_cartesian, wrap_angle_rad
from radar_tracker.csv_reader import ScanBatch
from radar_tracker.events import (
    EventType,
    SeparationEvent,
    TrackLifeState,
    new_event_id,
)
from radar_tracker.hypothesis_tree import (
    HypothesisKind,
    generate_local_hypotheses,
    trim_window,
)
from radar_tracker.models import build_measurement_model, build_predictor_updater, gate_threshold
from radar_tracker.scoring import (
    clutter_log_score,
    common_origin_score,
    detection_log_score,
    innovation_log_likelihood,
    missed_detection_log_score,
)
from radar_tracker.spatial_index import SpatialGrid, detection_cartesian
from radar_tracker.track_state import Track, detection_from_record, init_track_from_detections

logger = logging.getLogger(__name__)


@dataclass
class ScanMetrics:
    processing_time_s: float
    n_detections: int
    n_tracks: int
    n_hypotheses: int
    pruned_hypotheses: int


@dataclass
class TrackerOutput:
    tracks: list[Track]
    events: list[SeparationEvent]
    scan_metrics: list[ScanMetrics] = field(default_factory=list)


class RadarSeparationTracker:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.measurement_model = build_measurement_model(cfg)
        self.predictor, self.updater = build_predictor_updater(cfg, self.measurement_model)
        self.gate_threshold = gate_threshold(cfg)
        self.radar_pos = np.array(cfg.radar.position_m, dtype=float)
        self.tracks: dict[str, Track] = {}
        self.events: list[SeparationEvent] = []
        self._pending_separations: dict[str, dict] = {}
        self._scan_count = 0
        self._hypothesis_counter = 0
        self._track_counter = 0
        self._init_buffer: list[tuple[str, datetime, np.ndarray]] = []

    def _new_track_id(self) -> str:
        self._track_counter += 1
        return f"P{self._track_counter:04d}"

    def _new_hyp_id(self) -> str:
        self._hypothesis_counter += 1
        return f"H{self._hypothesis_counter:04d}"

    def _predict_track(self, track: Track, timestamp: datetime) -> GaussianState | None:
        if track.state is None:
            return None
        if track.state.timestamp >= timestamp:
            return track.state
        return self.predictor.predict(track.state, timestamp=timestamp)

    def _gate(self, predicted: GaussianState, detection: Detection) -> tuple[bool, float]:
        meas_pred = np.asarray(self.measurement_model.function(predicted, noise=False), dtype=float).ravel()
        det_vec = np.asarray(detection.state_vector, dtype=float).ravel()
        H = np.asarray(self.measurement_model.jacobian(predicted), dtype=float)
        S = np.asarray(H @ predicted.covar @ H.T + self.measurement_model.covar(), dtype=float)
        inn = det_vec - meas_pred
        inn[1] = wrap_angle_rad(float(inn[1]))
        inn[0] = wrap_angle_rad(float(inn[0]))
        d2 = float(inn @ np.linalg.inv(S) @ inn)
        return d2 <= self.gate_threshold, d2

    def _update_track(self, track: Track, detection: Detection) -> None:
        pred = self._predict_track(track, detection.timestamp)
        if pred is None:
            return
        hyp = SingleHypothesis(pred, detection)
        post = self.updater.update(hyp)
        track.state = post
        track.missed_scans = 0
        track.age_scans += 1
        track.record_history()
        _, d2 = self._gate(pred, detection)
        S = self.measurement_model.jacobian(pred) @ pred.covar @ self.measurement_model.jacobian(pred).T
        S = S + self.measurement_model.covar()
        inn = detection.state_vector - self.measurement_model.function(pred, noise=False)
        inn[1, 0] = wrap_angle_rad(float(inn[1, 0]))
        ll = innovation_log_likelihood(inn, S)
        track.score += ll + detection_log_score(self.cfg, 1)

    def _life_cycle_update(self, track: Track) -> None:
        tcfg = self.cfg.tracking
        if track.life == TrackLifeState.CANDIDATE:
            if track.age_scans >= tcfg.min_scans_to_confirm and track.score >= tcfg.confirmation_score_threshold:
                track.life = TrackLifeState.CONFIRMED
            elif track.age_scans >= tcfg.min_scans_to_reject and track.score <= tcfg.rejection_score_threshold:
                track.life = TrackLifeState.REJECTED
        if track.missed_scans > tcfg.max_missed_scans_before_coast:
            track.life = TrackLifeState.REJECTED

    def process_scan(self, batch: ScanBatch) -> ScanMetrics:
        t0 = time.perf_counter()
        if batch.missing:
            for tr in self.tracks.values():
                tr.missed_scans += 1
                tr.score += missed_detection_log_score(self.cfg, 1)
                self._life_cycle_update(tr)
            dt = time.perf_counter() - t0
            return ScanMetrics(dt, 0, len(self.tracks), 0, 0)

        dets = batch.detections
        n_pruned = 0
        n_hyps = 0

        # Spatial index for preselection
        cell = max(200.0, self.cfg.measurement.sigma_range_m * 10)
        grid = SpatialGrid(cell_size_m=cell)
        det_xyz = []
        for i, d in enumerate(dets):
            xyz = detection_cartesian(d.range_m, d.azimuth_rad, d.elevation_rad, self.radar_pos)
            det_xyz.append(xyz)
            grid.insert(i, xyz[0], xyz[1], xyz[2])

        assigned_det: set[int] = set()
        ambiguous_tracks: set[str] = set()
        candidate_pairs: list[tuple[str, int, float]] = []

        for tid, track in list(self.tracks.items()):
            if track.life == TrackLifeState.REJECTED:
                continue
            pred = self._predict_track(track, batch.timestamp)
            if pred is None:
                continue
            sv = pred.state_vector.ravel()
            px, py, pz = sv[0], sv[2], sv[4]
            gate_r = max(500.0, float(sv[0]) * 0.02)
            idxs = grid.query_ball(px, py, pz, gate_r)
            in_gate = []
            for i in idxs:
                if i in assigned_det:
                    continue
                det = detection_from_record(dets[i], self.measurement_model)
                ok, d2 = self._gate(pred, det)
                if ok:
                    in_gate.append((i, d2))
            if len(in_gate) == 1:
                i, _ = in_gate[0]
                det = detection_from_record(dets[i], self.measurement_model)
                self._update_track(track, det)
                assigned_det.add(i)
                track.last_update_scan = batch.scan_id
            elif len(in_gate) > 1:
                ambiguous_tracks.add(tid)
                in_gate.sort(key=lambda x: x[1])
                i_best, _ = in_gate[0]
                det_best = detection_from_record(dets[i_best], self.measurement_model)
                self._update_track(track, det_best)
                assigned_det.add(i_best)
                track.last_update_scan = batch.scan_id
                for i_extra, _ in in_gate[1:]:
                    if i_extra not in assigned_det:
                        self._propose_detach(track, dets[i_extra], batch)
                        assigned_det.add(i_extra)
                for i, d2 in in_gate:
                    candidate_pairs.append((tid, i, d2))
            else:
                track.missed_scans += 1
                track.score += missed_detection_log_score(self.cfg, 1)
            self._life_cycle_update(track)

        unassigned = [i for i in range(len(dets)) if i not in assigned_det]

        # 2e détection dans le scan : poursuite + branche fille (détachement / sens opposé)
        confirmed = [
            t for t in self.tracks.values() if t.life == TrackLifeState.CONFIRMED
        ]
        if len(dets) >= 2 and confirmed:
            parent = max(confirmed, key=lambda t: t.age_scans)
            for i in list(unassigned):
                if i in assigned_det:
                    continue
                if self._loose_detach_gate(parent, dets[i]):
                    self._propose_detach(parent, dets[i], batch)
                    assigned_det.add(i)
            unassigned = [i for i in range(len(dets)) if i not in assigned_det]

        if ambiguous_tracks:
            active = [self.tracks[t] for t in self.tracks if self.tracks[t].life != TrackLifeState.REJECTED]
            hyps = generate_local_hypotheses(
                active,
                ambiguous_tracks,
                unassigned,
                self.cfg,
                self._new_hyp_id,
            )
            n_hyps = len(hyps)
            n_pruned = sum(1 for h in hyps if h.pruned)
            best = hyps[0] if hyps else None
            if best and best.kind != HypothesisKind.SINGLE_CONTINUED:
                self._handle_structural_hypothesis(best, batch, unassigned, ambiguous_tracks)

        # Init new tracks from unassigned (after hypothesis handling)
        for i in unassigned:
            if i in assigned_det:
                continue
            rec = dets[i]
            self._init_buffer.append((batch.scan_id, rec.timestamp, rec.stonesoup_vector))
        self._try_init_tracks(batch)

        # Clutter penalty on unassigned
        gate_vol = (4 / 3) * np.pi * (5000**3)
        clutter_score = clutter_log_score(self.cfg, len(unassigned), gate_vol)
        for tr in self.tracks.values():
            if tr.life != TrackLifeState.REJECTED:
                tr.score += clutter_score / max(1, len(self.tracks))

        self._scan_count += 1
        dt = time.perf_counter() - t0
        return ScanMetrics(dt, len(dets), len(self.tracks), n_hyps, n_pruned)

    def _loose_detach_gate(self, parent: Track, det_record) -> bool:
        """Gate conservateur pour une 2e détection (séparation / sens opposé)."""
        pred = self._predict_track(parent, det_record.timestamp)
        if pred is None:
            return False
        xyz = detection_cartesian(
            det_record.range_m, det_record.azimuth_rad, det_record.elevation_rad, self.radar_pos
        )
        sv = pred.state_vector.ravel()
        dx = xyz[0] - sv[0]
        dy = xyz[1] - sv[2]
        dz = xyz[2] - sv[4]
        dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        max_dist = max(8000.0, 0.15 * float(np.linalg.norm([sv[0], sv[2], sv[4]])))
        return dist <= max_dist

    def _try_init_tracks(self, batch: ScanBatch) -> None:
        if len(self._init_buffer) < 2:
            return
        if any(t.life == TrackLifeState.CONFIRMED for t in self.tracks.values()):
            return
        recent = self._init_buffer[-3:]
        scan_ids = {s for s, _, _ in recent}
        if len(scan_ids) < 2:
            return
        tid = self._new_track_id()
        dets = [(ts, vec) for _, ts, vec in recent]
        tr = init_track_from_detections(tid, dets, self.cfg, self.measurement_model)
        if tr:
            self.tracks[tid] = tr
            self._init_buffer.clear()

    def _propose_detach(self, parent: Track, det_record, batch: ScanBatch) -> None:
        """Branche fille + accumulation de score (confirmation retardée)."""
        existing_child = next(
            (
                t
                for t in self.tracks.values()
                if t.parent_track_id == parent.track_id and t.life != TrackLifeState.REJECTED
            ),
            None,
        )
        if existing_child is not None:
            child = existing_child
            child_id = child.track_id
        else:
            child_id = self._new_track_id()
            child = init_track_from_detections(
                child_id,
                [(batch.timestamp, det_record.stonesoup_vector)],
                self.cfg,
                self.measurement_model,
            )
            if child is None:
                child = Track(track_id=child_id)
            child.parent_track_id = parent.track_id
            child.life = TrackLifeState.CANDIDATE
            self.tracks[child_id] = child
        det = detection_from_record(det_record, self.measurement_model)
        if child.state is not None:
            self._update_track(child, det)
        elif existing_child is None:
            self._update_track(child, det)

        ev_key = parent.track_id
        pending = self._pending_separations.get(ev_key, {"count": 0, "score": 0.0})
        # Ne confirmer que sur des scans où la séparation est observable (≥2 mesures)
        if batch.detections and len(batch.detections) >= 2:
            pending["count"] += 1
        cor = common_origin_score(
            parent.position_history,
            child.position_history,
            parent.cov_history,
            child.cov_history,
            self.cfg,
        )
        pending["score"] += parent.score * 0.01 + cor.score + 2.0
        self._pending_separations[ev_key] = pending

        pending["child_id"] = child_id
        if pending["count"] >= self.cfg.hypothesis.separation_confirmation_scans:
            if pending["score"] >= self.cfg.hypothesis.separation_score_threshold:
                self._emit_separation_event(
                    parent.track_id,
                    pending["child_id"],
                    batch,
                    pending["score"],
                    cor,
                    HypothesisKind.DETACH_WITH_CONTINUATION,
                )
                del self._pending_separations[ev_key]

    def _emit_separation_event(
        self,
        parent_id: str,
        child_id: str,
        batch: ScanBatch,
        score: float,
        cor,
        kind: HypothesisKind,
    ) -> None:
        if any(e.origin_track_or_group_id == parent_id and child_id in e.descendant_track_ids for e in self.events):
            return
        etype = (
            EventType.PROBABLE_DETACH
            if kind == HypothesisKind.DETACH_WITH_CONTINUATION
            else EventType.UNRESOLVED_GROUP_RESOLVED
        )
        if not cor.compatible and kind == HypothesisKind.DETACH_WITH_CONTINUATION:
            etype = EventType.AMBIGUOUS
        self.events.append(
            SeparationEvent(
                event_id=new_event_id(),
                origin_track_or_group_id=parent_id,
                descendant_track_ids=[parent_id, child_id],
                event_type=etype.value,
                remaining_ambiguities=[] if cor.compatible else ["origine_commune_incertaine"],
                estimated_separation_interval=(None, batch.timestamp.isoformat()),
                first_observable_resolution_time=batch.timestamp.isoformat(),
                confirmation_time=batch.timestamp.isoformat(),
                score=score,
                rationale={
                    "hypothesis_kind": kind.name,
                    "common_origin": cor.details,
                    "note": "origine commune probable, mécanisme indéterminé"
                    if cor.compatible
                    else "mécanisme indéterminé",
                },
            )
        )

    def _handle_structural_hypothesis(
        self,
        hyp,
        batch: ScanBatch,
        unassigned: list[int],
        ambiguous: set[str],
    ) -> None:
        if not unassigned:
            return
        dets = batch.detections
        i = unassigned[0]
        det = detection_from_record(dets[i], self.measurement_model)

        if hyp.kind == HypothesisKind.INDEPENDENT_BIRTH:
            tid = self._new_track_id()
            tr = init_track_from_detections(tid, [(batch.timestamp, dets[i].stonesoup_vector)], self.cfg, self.measurement_model)
            if tr is None:
                tr = Track(track_id=tid)
            tr.life = TrackLifeState.CANDIDATE
            self.tracks[tid] = tr
            return

        parent_id = next(iter(ambiguous), None)
        if parent_id is None and self.tracks:
            parent_id = max(self.tracks.keys(), key=lambda k: self.tracks[k].age_scans)

        if hyp.kind in (HypothesisKind.UNRESOLVED_TO_MULTI, HypothesisKind.DETACH_WITH_CONTINUATION):
            child_id = self._new_track_id()
            child = init_track_from_detections(
                child_id,
                [(batch.timestamp, dets[i].stonesoup_vector)],
                self.cfg,
                self.measurement_model,
            )
            if child is None:
                child = Track(track_id=child_id)
            child.parent_track_id = parent_id
            child.life = TrackLifeState.CANDIDATE
            self.tracks[child_id] = child

            if parent_id and parent_id in self.tracks:
                self._update_track(self.tracks[parent_id], det)

            ev_key = f"{parent_id}:{child_id}"
            pending = self._pending_separations.get(ev_key, {"count": 0, "score": 0.0})
            pending["count"] += 1
            cor = common_origin_score(
                self.tracks[parent_id].position_history if parent_id else [],
                child.position_history,
                self.tracks[parent_id].cov_history if parent_id else [],
                child.cov_history,
                self.cfg,
            )
            pending["score"] += hyp.score + cor.score
            self._pending_separations[ev_key] = pending

            if pending["count"] >= self.cfg.hypothesis.separation_confirmation_scans:
                if pending["score"] >= self.cfg.hypothesis.separation_score_threshold:
                    self._emit_separation_event(
                        parent_id or "unknown",
                        child_id,
                        batch,
                        pending["score"],
                        cor,
                        hyp.kind,
                    )
                    del self._pending_separations[ev_key]

    def run(self, scans: list[ScanBatch]) -> TrackerOutput:
        metrics = []
        for scan in scans:
            metrics.append(self.process_scan(scan))
        active = [t for t in self.tracks.values() if t.life != TrackLifeState.REJECTED]
        return TrackerOutput(tracks=active, events=self.events, scan_metrics=metrics)
