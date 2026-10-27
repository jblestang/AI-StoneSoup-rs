"""Événements de séparation et export."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(str, Enum):
    UNRESOLVED_GROUP_RESOLVED = "unresolved_group_resolved"
    PROBABLE_DETACH = "probable_detach"
    REJECTED_CROSSING = "rejected_crossing"
    REJECTED_INDEPENDENT_BIRTH = "rejected_independent_birth"
    AMBIGUOUS = "ambiguous"


class TrackLifeState(str, Enum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass
class SeparationEvent:
    event_id: str
    origin_track_or_group_id: str
    descendant_track_ids: list[str]
    event_type: str
    remaining_ambiguities: list[str]
    estimated_separation_interval: tuple[str | None, str | None]
    first_observable_resolution_time: str | None
    confirmation_time: str | None
    score: float
    rationale: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_event_id() -> str:
    return str(uuid.uuid4())


def export_events_json(events: list[SeparationEvent], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in events], f, indent=2)
