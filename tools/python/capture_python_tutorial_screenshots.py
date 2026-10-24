#!/usr/bin/env python3
"""Capture Stone Soup Rust viewer-style PNGs for catalogue tutorials 01–19."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("OUT", ROOT / "screenshot" / "python"))
TUTORIALS = tuple(f"{i:02d}" for i in range(1, 20))


def dump_bin() -> Path:
    path = ROOT / "target" / "release" / "dump_viewer_trail"
    if path.is_file():
        return path
    subprocess.check_call(
        [
            "cargo",
            "build",
            "-p",
            "stonesoup-scenarios",
            "--bin",
            "dump_viewer_trail",
            "--release",
            "--locked",
        ],
        cwd=ROOT,
    )
    return path


def dump_trail(code: str) -> dict:
    proc = subprocess.run(
        [str(dump_bin()), code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"dump_viewer_trail {code} failed (exit {proc.returncode})")
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def main() -> int:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "matplotlib", "-q"]
        )

    plot_script = ROOT / "tools/python/plot_viewer_trail.py"
    OUT.mkdir(parents=True, exist_ok=True)

    for code in TUTORIALS:
        payload = dump_trail(code)
        tmp = OUT / f".trail_{code}.json"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        dest = OUT / f"tutorial_{code}.png"
        subprocess.check_call(
            [sys.executable, str(plot_script), str(tmp), str(dest)],
        )
        tmp.unlink(missing_ok=True)
        print("wrote", dest.resolve())

    # Keep step-wise resampling plots (Stone Soup resampler sequence).
    subprocess.check_call(
        [sys.executable, str(ROOT / "tools/python/plot_lot5_resampling_reference.py")],
        cwd=ROOT,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
