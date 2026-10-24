#!/usr/bin/env python3
"""Export systematic resampler + ESS reference (Stone Soup v1.9.1, NumPy seed 1991)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

UPSTREAM_SHA = "a4336b920a799cfe0a77ecb05867c5deeb371c7a"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "verification" / "fixtures" / "resampler_systematic.json"


def main() -> int:
    import numpy as np
    from stonesoup.resampler.particle import SystematicResampler
    from stonesoup.types.particle import Particle
    from stonesoup.types.state import ParticleState

    np.random.seed(1991)
    n = 8
    weights = np.array([0.05, 0.1, 0.05, 0.2, 0.15, 0.1, 0.25, 0.1])
    log_w = np.log(weights)
    particles = [
        Particle(
            state_vector=np.array(
                [[float(i)], [float(i) * 0.5], [float(i) * 2.0], [float(i) * -0.1]]
            ),
            weight=float(w),
        )
        for i, w in enumerate(weights)
    ]
    state = ParticleState(None, particle_list=particles)
    ess_before = 1.0 / np.sum(np.exp(2 * log_w))
    resampled = SystematicResampler().resample(state, nparts=n)
    ess_after = 1.0 / np.sum(np.exp(2 * resampled.log_weight))
    payload = {
        "upstream_sha": UPSTREAM_SHA,
        "numpy_seed": 1991,
        "n_particles": n,
        "state_dim": 4,
        "log_weights_in": log_w.tolist(),
        "particles_in": state.state_vector.T.tolist(),
        "ess_before": float(ess_before),
        "log_weights_out": resampled.log_weight.tolist(),
        "particles_out": resampled.state_vector.T.tolist(),
        "ess_after": float(ess_after),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
