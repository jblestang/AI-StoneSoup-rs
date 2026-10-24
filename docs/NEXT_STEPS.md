# Next steps

**Lot 1:** complete (Kalman tutorial vertical slice, viewer, WASM, parity, no-panic harness).

**Lot 2:** in progress — Tutorial **02 EKF** (bearing–range, `mod_bearing`, analytic Jacobian, parity vs Stone Soup v1.9.1). **Next within Lot 2:** UKF (03), particle filter (04), viewer/headless catalogue entries.

## Lot 2 outline

- Nonlinear measurement models (range/bearing), circular angles.
- EKF / UKF / particle filter in `stonesoup-core` with bounded particle cap.
- Port tutorials `02_ExtendedKalmanFilterTutorial.py`, `03_UnscentedKalmanFilterTutorial.py`, `04_ParticleFilter.py`.
- Extend Python fixture exporter + parity tests.

## Optional polish

- Browser smoke test (load WASM, step scenario, reset) in CI when headless WASM runner is available.
- `xtask verify`: optional Trunk step when `NO_COLOR` unset.

## Inventory

- Remaining components: see `docs/PORTING_MATRIX.csv` (`todo` / `blocked`).
- Full catalogue port is **not** complete — 1/73 scenarios validated at pinned SHA.
