# Dual rotating radar clutter experiment (Rust)

## Run

```bash
cargo run -p stonesoup-scenarios --release --bin dual_radar_clutter_stats
```

Writes `verification/reports/dual_radar_clutter_stats.json`.

## Scenario

| Radar | Rotation | Range |
|-------|----------|-------|
| A | 6 s | 400 km |
| B | 3 s | **178.9 km** (√0.2 × 400 km) |

Co-located sensors with **20% area overlap** between coverage disks: (R_B/R_A)² = 0.2.

- **500 plots** total (250 per radar), ~10% clutter, plot-driven tracker steps.
- **20% overlap** also reported as the fraction of **true** plots whose range falls inside B’s disk (targets at 80 km are always in overlap).

## Plots vs tracks

The JSON field **`plots.total_plots`** is the **500** measurement budget (250×2).

**`tracks.total_completed_tracks`** counts **finished track segments** from the multi-target filter (GNN + EKF), not plots. With 1-point initiation and short coast, this count is typically **on the order of hundreds**—one segment per plot that is not merged into an existing track.

Implementation: `crates/stonesoup-scenarios/src/dual_radar_clutter.rs`.

Smoke test: `cargo test -p stonesoup-scenarios --test dual_radar_smoke`.
