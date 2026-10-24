# Next steps

**Lot 1:** complete (Kalman tutorial vertical slice, viewer, WASM, parity, no-panic harness).

**Lot 2:** complete — Tutorials **02 EKF**, **03 UKF**, and **04 particle filter** (bearing–range, parity vs Stone Soup v1.9.1 @ pinned SHA). Headless CLI: `lot2_ekf`, `lot2_ukf`, `lot2_pf`.

**Lot 3:** complete — **05** NN/clutter, **06** global NN (two targets), **07** PDA, **08** JPDA (parity vs pinned Stone Soup). Headless: `lot3_clutter`, `lot3_gnn`, `lot3_pda`, `lot3_jpda`.

**Next:** Lot 4+ per `PORTING_MATRIX.csv` (initiators, trackers, metrics, …).

## Optional polish

- Browser smoke test (load WASM, step scenario, reset) in CI when headless WASM runner is available.
- Particle filter: ESS/systematic resampler parity with Stone Soup (reference PF track omits resampling for strict Rust parity).
- `xtask verify`: optional Trunk step when `NO_COLOR` unset.

## Inventory

- Remaining components: see `docs/PORTING_MATRIX.csv` (`todo` / `blocked`).
- Validated tutorials at pinned SHA: 01–08.
