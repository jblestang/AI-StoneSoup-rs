# Next steps

**Current lot:** 1 delivered (usable vertical slice). **Next:** Lot 2 — EKF, UKF, particle filter.

## Immediate technical debt

1. Extend `#[no_panic]` coverage to Kalman predict/update (refactor `Mat::mul` or isolate certified kernel).
2. Negative mutation test wired in CI once (1) passes.
3. Browser smoke test (load WASM, step scenario, reset) — not automated in this session.
4. `cargo run -p xtask -- verify`: add optional Trunk step when `NO_COLOR` unset.

## Lot 2 outline

- Nonlinear measurement models (range/bearing), circular angles.
- EKF / UKF / particle filter in `stonesoup-core` with bounded particle cap.
- Port tutorials `02_ExtendedKalmanFilterTutorial.py`, `03_UnscentedKalmanFilterTutorial.py`, `04_ParticleFilter.py`.
- Extend Python fixture exporter + parity tests.

## Inventory

- Remaining components: see `docs/PORTING_MATRIX.csv` (`todo` / `blocked`).
- Full catalogue port is **not** complete — 1/73 scenarios validated at pinned SHA.
