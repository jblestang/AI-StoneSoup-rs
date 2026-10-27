#!/usr/bin/env python3
"""Évalue événements estimés vs vérité terrain (jeu de test séparé du réglage)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar_tracker.evaluation import match_events
from radar_tracker.events import SeparationEvent
from radar_tracker.simulator.truth import GroundTruthEvent, ScenarioTruth, generate_scenario_truth


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--events", default=str(ROOT / "output" / "events.json"))
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=99)
    args = p.parse_args()
    truth = generate_scenario_truth(args.scenario, args.duration, 1.0, seed=args.seed)
    raw = json.loads(Path(args.events).read_text()) if Path(args.events).exists() else []
    estimated = [
        SeparationEvent(
            event_id=e["event_id"],
            origin_track_or_group_id=e["origin_track_or_group_id"],
            descendant_track_ids=e["descendant_track_ids"],
            event_type=e["event_type"],
            remaining_ambiguities=e.get("remaining_ambiguities", []),
            estimated_separation_interval=tuple(e.get("estimated_separation_interval", (None, None))),
            first_observable_resolution_time=e.get("first_observable_resolution_time"),
            confirmation_time=e.get("confirmation_time"),
            score=e.get("score", 0.0),
            rationale=e.get("rationale", {}),
        )
        for e in raw
    ]
    m = match_events(estimated, truth)
    print(json.dumps(m.__dict__, indent=2))


if __name__ == "__main__":
    main()
