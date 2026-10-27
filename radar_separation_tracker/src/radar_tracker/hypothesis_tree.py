"""Arbre d'hypothèses local à fenêtre glissante (2 branches max v1)."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from radar_tracker.config import AppConfig
from radar_tracker.track_state import Track

logger = logging.getLogger(__name__)


class HypothesisKind(Enum):
    SINGLE_CONTINUED = auto()
    INDEPENDENT_BIRTH = auto()
    UNRESOLVED_TO_MULTI = auto()
    DETACH_WITH_CONTINUATION = auto()


@dataclass
class AssociationHypothesis:
    hypothesis_id: str
    kind: HypothesisKind
    tracks: list[Track]
    score: float = 0.0
    scan_log: list[str] = field(default_factory=list)
    pruned: bool = False


@dataclass
class HypothesisTree:
    root_id: str
    hypotheses: dict[str, AssociationHypothesis] = field(default_factory=dict)
    scan_history: list[str] = field(default_factory=list)


def _clone_tracks(tracks: list[Track]) -> list[Track]:
    # Références seulement — les hypothèses servent au score, pas à cloner les états Stone Soup.
    return list(tracks)


def generate_local_hypotheses(
    base_tracks: list[Track],
    ambiguous_track_ids: set[str],
    unassigned_detection_indices: list[int],
    cfg: AppConfig,
    next_ids,
) -> list[AssociationHypothesis]:
    """Génère hypothèses pour un groupe local ambigu (≤2 branches)."""
    hyps: list[AssociationHypothesis] = []
    base_score = sum(t.score for t in base_tracks)

    hyps.append(
        AssociationHypothesis(
            hypothesis_id=next_ids(),
            kind=HypothesisKind.SINGLE_CONTINUED,
            tracks=_clone_tracks(base_tracks),
            score=base_score,
        )
    )

    if unassigned_detection_indices:
        hyps.append(
            AssociationHypothesis(
                hypothesis_id=next_ids(),
                kind=HypothesisKind.INDEPENDENT_BIRTH,
                tracks=_clone_tracks(base_tracks),
                score=base_score - cfg.hypothesis.complexity_penalty,
            )
        )

    if len(ambiguous_track_ids) >= 1 and unassigned_detection_indices:
        hyps.append(
            AssociationHypothesis(
                hypothesis_id=next_ids(),
                kind=HypothesisKind.UNRESOLVED_TO_MULTI,
                tracks=_clone_tracks(base_tracks),
                score=base_score - 2 * cfg.hypothesis.complexity_penalty,
            )
        )
        hyps.append(
            AssociationHypothesis(
                hypothesis_id=next_ids(),
                kind=HypothesisKind.DETACH_WITH_CONTINUATION,
                tracks=_clone_tracks(base_tracks),
                score=base_score - 2 * cfg.hypothesis.complexity_penalty,
            )
        )

    max_h = cfg.hypothesis.max_hypotheses
    hyps.sort(key=lambda h: h.score, reverse=True)
    if len(hyps) > max_h:
        for h in hyps[max_h:]:
            h.pruned = True
            if cfg.performance.log_pruned_hypotheses:
                logger.info("hypothesis pruned id=%s kind=%s", h.hypothesis_id, h.kind)
        hyps = hyps[:max_h]
    return hyps


def trim_window(tree: HypothesisTree, depth: int) -> None:
    if len(tree.scan_history) > depth:
        tree.scan_history = tree.scan_history[-depth:]
