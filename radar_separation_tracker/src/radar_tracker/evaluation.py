"""Évaluation vérité terrain vs événements estimés."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from radar_tracker.events import SeparationEvent
from radar_tracker.simulator.truth import GroundTruthEvent, ScenarioTruth


@dataclass
class EventMatchMetrics:
    precision: float
    recall: float
    false_events_per_hour: float
    mean_confirmation_delay_s: float
    lineage_errors: int


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s)


def match_events(
    estimated: list[SeparationEvent],
    truth: ScenarioTruth,
    time_tolerance_s: float = 15.0,
) -> EventMatchMetrics:
    """Règle : un événement estimé matche le GT si type compatible et |t_est - t_gt| <= tolérance."""
    gt = truth.events
    if not gt and not estimated:
        return EventMatchMetrics(1.0, 1.0, 0.0, 0.0, 0)
    matched_gt = set()
    matched_est = set()
    delays = []
    lineage_errors = 0

    type_map = {
        "unresolved_disperses": {"unresolved_group_resolved", "ambiguous"},
        "probable_detach": {"probable_detach", "ambiguous"},
    }

    for ei, ev in enumerate(estimated):
        t_est = _parse_iso(ev.confirmation_time)
        for gi, g in enumerate(gt):
            if gi in matched_gt:
                continue
            allowed = type_map.get(g.event_type, {g.event_type})
            if ev.event_type not in allowed and ev.event_type != "ambiguous":
                continue
            if t_est and abs((t_est - g.time).total_seconds()) <= time_tolerance_s:
                matched_gt.add(gi)
                matched_est.add(ei)
                delays.append(abs((t_est - g.time).total_seconds()))
                break

    tp = len(matched_est)
    fp = len(estimated) - tp
    fn = len(gt) - len(matched_gt)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    duration_h = max(
        (truth.start_time + timedelta(seconds=120) - truth.start_time).total_seconds() / 3600,
        1e-6,
    )
    false_per_hour = fp / duration_h
    mean_delay = sum(delays) / len(delays) if delays else 0.0
    return EventMatchMetrics(precision, recall, false_per_hour, mean_delay, lineage_errors)
