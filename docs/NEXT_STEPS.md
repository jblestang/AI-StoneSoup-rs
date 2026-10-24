# Next steps

**Lot 1:** complete (Kalman tutorial vertical slice, viewer, WASM, parity, no-panic harness).

**Lot 2:** complete — Tutorials **02 EKF**, **03 UKF**, and **04 particle filter** (bearing–range, parity vs Stone Soup v1.9.1 @ pinned SHA). Headless CLI: `lot2_ekf`, `lot2_ukf`, `lot2_pf`.

**Lot 3a:** Tutorial **05 clutter + nearest neighbour** — `Lot3ClutterScenario`, Mahalanobis gate, parity `parity_lot3`, headless `lot3_clutter`, viewer shows true vs clutter detections.

**Next (Lot 3b+):** tutorials **06–08** (multi-target, PDA, JPDA).

## Optional polish

- Browser smoke test (load WASM, step scenario, reset) in CI when headless WASM runner is available.
- Particle filter: ESS/systematic resampler parity with Stone Soup (reference PF track omits resampling for strict Rust parity).
- `xtask verify`: optional Trunk step when `NO_COLOR` unset.

## Inventory

- Remaining components: see `docs/PORTING_MATRIX.csv` (`todo` / `blocked`).
- Validated tutorials at pinned SHA: 01, 02, 03, 04, 05.
