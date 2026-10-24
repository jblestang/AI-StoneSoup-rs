# Next steps

**Lot 1:** complete (Kalman tutorial vertical slice, viewer, WASM, parity, no-panic harness).

**Lot 2:** complete — Tutorials **02 EKF**, **03 UKF**, and **04 particle filter** (bearing–range, parity vs Stone Soup v1.9.1 @ pinned SHA). Headless CLI: `lot2_ekf`, `lot2_ukf`, `lot2_pf`.

**Lot 3:** complete — **05** NN/clutter, **06** global NN (two targets), **07** PDA, **08** JPDA (parity vs pinned Stone Soup). Headless: `lot3_clutter`, `lot3_gnn`, `lot3_pda`, `lot3_jpda`.

**Lot 4:** complete — **09** initiators & deleters (`MultiMeasurementInitiator`, GNN, covariance deleter), **10** simulation stack (`MultiTargetTracker` loop). Headless: `lot4_initiators`, `lot4_simulation`. Parity: `parity_lot4_t09`, `parity_lot4_t10`.

**Lot 5:** complete — **Resampling** tutorial (`Multinomial`, `Systematic`, `Stratified`, `Residual`, `ESS` wrappers in `stonesoup-sim`). Headless: `lot5_resampling`. WASM viewer: `?tutorial=11` (aliases `resampling`, `sample`). Parity: `parity_lot5_resampling` (per-step through all 6 resampler demos + primitive fixtures).

**Lot 6:** complete — **Information filter** tutorial (information-form predict/update, parity vs Kalman-equivalent Stone Soup track). Headless: `lot6_information`. WASM viewer: `?tutorial=12`. Parity: `parity_lot6_information` (21 steps through end).

**Lot 7:** complete — **GM-PHD** tutorial (`PHDUpdater`, mixture hypothesise/update, prune/merge reducer; seed 42 replay). Headless: `lot7_gm_phd`. WASM viewer: `?tutorial=13`. Parity: `parity_lot7_gm_phd` (reduced mixture cardinality + strong-mode positions vs Stone Soup export).

**Lot 8:** complete — **ASD filter** tutorial (`ASDFilter.py`: accumulated state densities, OOS measurement order, `max_nstep=30`). Headless: `lot8_asd`. WASM viewer: `?tutorial=14`. Parity: `parity_lot8_asd` (101 act-state updates vs Stone Soup export).

**Lot 11:** complete — **KD-tree GNN** tutorial (`DetectionKDTreeGNN2D`, scaled multi-target stack with seed-42 replay). Headless: `lot11_kdtree`. WASM viewer: `?tutorial=15` (aliases `kdtree`). Parity: `parity_lot11_kdtree` (12 steps, sorted track means vs Stone Soup export).

**Lot 10:** complete — **Single sensor management** tutorial (`01_SingleSensorManagement.py`: rotating bearing–range radar, random vs brute-force managers, EKF + GNN; measurement replay + `associate_gnn_2d_ekf`). Headless: `lot10_single_sensor`. WASM viewer: `?tutorial=16` (aliases `sensor`, `sensormanagement`). Parity: `parity_lot10_single_sensor` (random + brute, 20 steps, sorted track means vs Stone Soup export).

**Lot 12:** complete — **AKKF** tutorial (`AKKF.py`: quadratic kernel, bearing-only BOT, 30 particles). Headless: `lot12_akkf` (alias `lot9_akkf`). WASM viewer: `?tutorial=17` (aliases `akkf`, `kernel`). Parity: `parity_lot9_akkf` (30 steps, weighted mean vs Stone Soup export).

**Lot 13:** complete — **Multi-sensor management** tutorial (`02_MultiSensorManagement.py`: two rotating radars, random vs brute-force managers, per-detection EKF geometry via `associate_gnn_2d_ekf_det_models`). Headless: `lot13_multi_sensor`. WASM viewer: `?tutorial=18` (aliases `multisensor`, `multi-sensor`). Parity: `parity_lot13_multi_sensor` (random + brute, 20 steps).

**Lot 14:** complete — **Optimised sensor management** tutorial (`03_OptimisedSensorManagement.py`: SciPy `OptimizeBrute` and `OptimizeBasinHopping` managers vs brute-force baseline; replay parity). Headless: `lot14_optimised_sensor`. WASM viewer: `?tutorial=19` (aliases `optimised`, `opt-sensor`). Parity: `parity_lot14_optimised_sensor` (three manager modes, 20 steps).

**Lot 15:** complete — **Information vs network architectures** tutorial (`02_Information_and_Network_Architectures.py`: hierarchical generator, top fusion-node track replay from Stone Soup export; network/information agreement + edge-removal modes). Headless: `lot15_network_info_arch`. WASM viewer: `?tutorial=20` (aliases `architecture`, `network-arch`). Parity: `parity_lot15_network_info_arch` (fixture replay + network/information track alignment).

**Lot 18:** complete — **Simple OOSM** example (`example_simple_oosm.py`: fixed-lag vs naive arrival-order tracks exported from Stone Soup; Rust fixture replay). Headless: `lot18_simple_oosm`. WASM viewer: `?tutorial=23` (aliases `oosm`, `simple-oosm`). Parity: `parity_lot18_simple_oosm`.

**Lot 19:** complete — **Kalman filter OOSM** example (`KalmanFilterOOSMExample.py`: three EKF trackers — ignore delay, lag-adjusted, sensor-1-only; Stone Soup export + Rust fixture replay). Headless: `lot19_kalman_oosm`. WASM viewer: `?tutorial=24` (aliases `kalman-oosm`, `ekf-oosm`). Parity: `parity_lot19_kalman_oosm` (exact replay for all three tracker modes).

**Lot 20:** complete — **Particle filter OOSM** example (`PF_OOSM_example.py`: Orton–Marrs-style PF OOSM vs ignore-delayed track; Stone Soup export + Rust fixture replay). Headless: `lot20_pf_oosm`. WASM viewer: `?tutorial=25` (aliases `pf-oosm`, `particle-oosm`). Parity: `parity_lot20_pf_oosm`.

**Lot 22:** complete — **Bearings-only EKF** example (`bearing_only_example.py`: moving platform `RadarBearing`, single-target EKF tracker; Stone Soup export + Rust fixture replay). Headless: `lot22_bearing_only`. WASM viewer: `?tutorial=27` (aliases `bearing-only`, `bot`). Parity: `parity_lot22_bearing_only`.

**Lot 21:** complete — **OOSM Algorithm C** example (`example_oosm_algorithm.py`: UKF+JPDA, two targets, inverse-time pseudo-measurements vs ignore-OOSM; Stone Soup export + Rust fixture replay). Headless: `lot21_oosm_algorithm`. WASM viewer: `?tutorial=26` (aliases `algorithm-c`, `oosm-algorithm`). Parity: `parity_lot21_oosm_algorithm` (four track/mode exports).

**Lot 17:** complete — **Introduction to architectures** tutorial (`01_Introduction_to_Architectures.py`: static B→A and C→A information graph; Stone Soup topology export + viewer layout). Headless: `lot17_arch_intro`. WASM viewer: `?tutorial=22` (aliases `arch-intro`, `intro-arch`). Parity: `parity_lot17_arch_intro`.

**Lot 16:** complete — **Avoiding data incest** tutorial (`03_Avoiding_Data_Incest.py`: NH vs hierarchical information architectures, Fusion Node 1 tracks + mean covariance trace replay). Headless: `lot16_data_incest`. WASM viewer: `?tutorial=21` (aliases `data-incest`, `incest`). Parity: `parity_lot16_data_incest` (fixture replay; NH lower mean cov trace vs hierarchical).

**Lot 23:** complete — **Multi-tracker comparison** example (`Multi_Tracker_Example.py`: EKF/UKF/PF/ESIF on manoeuvring targets; Stone Soup export + fixture replay). Headless: `lot23_multi_tracker`. WASM viewer: `?tutorial=28` (aliases `multi-tracker`). Parity: `parity_lot23_multi_tracker` (four filter modes).

**Lot 24:** complete — **Ensemble Kalman filter** example (`EnsembleFilterExample.py`: EnKF bearing–range, seed 1991). Headless: `lot24_ensemble_filter`. WASM viewer: `?tutorial=29` (aliases `ensemble`, `enkf`). Parity: `parity_lot24_ensemble_filter`.

**Lot 25:** complete — **Gromov particle flow** example (`ParticleFlow.py`: bootstrap PF vs Gromov flow vs parallel-EKF flow; seed 2020). Headless: `lot25_particle_flow`. WASM viewer: `?tutorial=30` (aliases `particle-flow`, `gromov`). Parity: `parity_lot25_particle_flow` (three filter modes).

## Screenshots & viewer catalogue

- **WASM** final states (catalogue **01–30**): [screenshot/wasm/README.md](../screenshot/wasm/README.md) — `bash tools/scripts/capture_wasm_tutorial_screenshots.sh`
- **Python** viewer-style plots (same trails via `dump_viewer_trail`): [screenshot/python/README.md](../screenshot/python/README.md) — `bash tools/scripts/capture_python_tutorial_screenshots.sh`

## Optional polish

Done on Lot 3 branch:

- **WASM smoke:** `verification/scripts/wasm_viewer_smoke.sh` (Playwright: `?steps=3`, Reset, `?finish=1`); runs from `xtask verify` on local machines with `trunk` installed (`VERIFY_WASM=1` to force in CI).
- **Resampler:** `stonesoup_sim::systematic_resample` / `ess_resample_if_needed` with fixture parity; Tutorial 04 PF uses ESS + systematic resampling (Stone Soup tutorial 04).
- **Viewer:** Lot 3 tutorials 06–08 show per-step clutter/true detections and RMSE in the WASM viewer (`?steps=N` for partial automation).

## Inventory

- Remaining components: see `docs/PORTING_MATRIX.csv` (`todo` / `blocked`).
- Validated catalogue entries at pinned SHA: **01–27** (examples **23–27** include full OOSM set + bearings-only BOT) (viewer **11** = resampling; **12** = information filter; **13** = GM-PHD; **14** = ASD; **15** = KD-tree GNN; **16** = single sensor management; **17** = AKKF; **18** = multi-sensor management; **19** = optimised sensor management; **20** = network/information architectures; **21** = data incest architectures; **22** = architecture introduction graph).
- **Next feature lots:** native OOSM / architecture engines, further `PORTING_MATRIX.csv` `todo` rows.
