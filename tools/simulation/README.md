# Dual rotating radar clutter experiment (Rust)

Native port of the two-radar, 10% clutter, 250-plots-per-radar scenario.

## Run

```bash
cargo run -p stonesoup-scenarios --release --bin dual_radar_clutter_stats
```

Writes `verification/reports/dual_radar_clutter_stats.json` with plot counts, clutter fraction, truth coverage, track / false-track counts, and track durations.

## Scenario

| Radar | Rotation | Range |
|-------|----------|-------|
| A | 6 s | 400 km |
| B | 3 s | 200 km |

Co-located **90° FOV** rotating bearing–range sensors, fused **GNN + EKF** multi-target tracker (3-point confirmation, 8 s coast delete). Four truths at **80 km** on cardinal bearings. Poisson clutter (`CLUTTER_LAMBDA = 0.52`) tuned for ~**10%** clutter among plots.

Implementation: `crates/stonesoup-scenarios/src/dual_radar_clutter.rs`.

Smoke test: `cargo test -p stonesoup-scenarios --test dual_radar_smoke`.
