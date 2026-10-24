# Next steps

**Lot 1:** complete (Kalman tutorial vertical slice, viewer, WASM, parity, no-panic harness).

**Lot 2:** complete — Tutorials **02 EKF**, **03 UKF**, and **04 particle filter** (bearing–range, parity vs Stone Soup v1.9.1 @ pinned SHA). Headless CLI: `lot2_ekf`, `lot2_ukf`, `lot2_pf`.

**Lot 3:** complete — **05** NN/clutter, **06** global NN (two targets), **07** PDA, **08** JPDA (parity vs pinned Stone Soup). Headless: `lot3_clutter`, `lot3_gnn`, `lot3_pda`, `lot3_jpda`.

**Next:** Lot 4+ per `PORTING_MATRIX.csv` (initiators, trackers, metrics, …).

## Optional polish

Done on Lot 3 branch:

- **WASM smoke:** `verification/scripts/wasm_viewer_smoke.sh` (Playwright: `?steps=3`, Reset, `?finish=1`); runs from `xtask verify` when `NO_COLOR` is unset (after `trunk build --release`).
- **Resampler:** `stonesoup_sim::systematic_resample` / `effective_sample_size` with fixture parity (`verification/fixtures/resampler_systematic.json`). Tutorial 04 PF scenario still omits resampling for strict track parity.
- **Viewer:** Lot 3 tutorials 06–08 show per-step clutter/true detections and RMSE in the WASM viewer (`?steps=N` for partial automation).

## Inventory

- Remaining components: see `docs/PORTING_MATRIX.csv` (`todo` / `blocked`).
- Validated tutorials at pinned SHA: 01–08.
