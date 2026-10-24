# Lot 2 — Nonlinear filters implementation plan

**Branch:** `cursor/lot2-ekf-abcb`  
**Upstream:** Stone Soup v1.9.1 @ `a4336b920a799cfe0a77ecb05867c5deeb371c7a`

## Scope

| Phase | Tutorial | Deliverable |
|-------|----------|-------------|
| **2a (started)** | `02_ExtendedKalmanFilterTutorial.py` | EKF + bearing–range, fixtures, `parity_lot2` |
| 2b | `03_UnscentedKalmanFilterTutorial.py` | UKF + parity |
| 2c | `04_ParticleFilter.py` | Particle filter + cap + parity |
| 2d | Integration | Viewer/headless catalogue, matrix `validated` rows |

## Phase 2a (this PR)

- `stonesoup-core`: `mod_bearing`, `CartesianToBearingRange`, `ExtendedKalmanUpdater`
- Fixtures: `verification/fixtures/lot2_ekf.json`, `lot2_fixtures.rs`
- Tests: `parity_lot2`, `ekf_tutorial02_step0/1`
- Export: `tools/python/export_lot2_fixtures.py`

## Notes

- Analytic bearing–range Jacobian (Stone Soup finite-diff breaks at bearing π).
- Innovation uses Euclidean `mod_bearing` on the bearing component (NumPy-compatible `%`).
