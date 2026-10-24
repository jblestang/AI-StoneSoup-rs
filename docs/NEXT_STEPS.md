# Next steps

**Lot 1:** complete (Kalman tutorial vertical slice, viewer, WASM, parity, no-panic harness).

**Lot 2:** complete — Tutorials **02 EKF**, **03 UKF**, and **04 particle filter** (bearing–range, parity vs Stone Soup v1.9.1 @ pinned SHA). Headless CLI: `lot2_ekf`, `lot2_ukf`, `lot2_pf`.

**Lot 3:** complete — **05** NN/clutter, **06** global NN (two targets), **07** PDA, **08** JPDA (parity vs pinned Stone Soup). Headless: `lot3_clutter`, `lot3_gnn`, `lot3_pda`, `lot3_jpda`.

**Lot 4:** complete — **09** initiators & deleters (`MultiMeasurementInitiator`, GNN, covariance deleter), **10** simulation stack (`MultiTargetTracker` loop). Headless: `lot4_initiators`, `lot4_simulation`. Parity: `parity_lot4_t09`, `parity_lot4_t10`.

**Lot 5:** complete — **Resampling** tutorial (`Multinomial`, `Systematic`, `Stratified`, `Residual`, `ESS` wrappers in `stonesoup-sim`). Headless: `lot5_resampling`. WASM viewer: `?tutorial=11` (aliases `resampling`, `sample`). Parity: `parity_lot5_resampling` (per-step through all 6 resampler demos + primitive fixtures).

**Next:** Lot 6+ per `PORTING_MATRIX.csv` (information filter, GM-PHD, metrics, …).

## Optional polish

Done on Lot 3 branch:

- **WASM smoke:** `verification/scripts/wasm_viewer_smoke.sh` (Playwright: `?steps=3`, Reset, `?finish=1`); runs from `xtask verify` on local machines with `trunk` installed (`VERIFY_WASM=1` to force in CI).
- **Resampler:** `stonesoup_sim::systematic_resample` / `ess_resample_if_needed` with fixture parity; Tutorial 04 PF uses ESS + systematic resampling (Stone Soup tutorial 04).
- **Viewer:** Lot 3 tutorials 06–08 show per-step clutter/true detections and RMSE in the WASM viewer (`?steps=N` for partial automation).

## Inventory

- Remaining components: see `docs/PORTING_MATRIX.csv` (`todo` / `blocked`).
- Validated tutorials at pinned SHA: 01–10 + Resampling sampling tutorial.
