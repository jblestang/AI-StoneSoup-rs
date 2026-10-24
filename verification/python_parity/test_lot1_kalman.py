"""Differential check: Rust scenario fixtures match exported Stone Soup track."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "verification" / "fixtures" / "lot1_kalman.json"


def test_fixture_track_internal_consistency() -> None:
    data = json.loads(FIX.read_text())
    assert data["upstream_sha"] == "a4336b920a799cfe0a77ecb05867c5deeb371c7a"
    assert len(data["measurements"]) == len(data["track"]) == len(data["truth"])


def test_rust_parity_via_cargo_test() -> None:
    proc = subprocess.run(
        ["cargo", "test", "-p", "stonesoup-scenarios", "matches_python_reference_track", "--locked"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
