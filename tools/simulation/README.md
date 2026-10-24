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
| B | 3 s | **178.9 km** (√0.2 × 400 km, **20% area overlap**) |

Co-located **90° FOV** rotating bearing–range sensors, **fused** `MultiTargetTracker`-style loop (GNN + EKF, **3-point** initiation, **8 s** coast delete).

- **500 plots** (250 per radar), **~10% clutter** (~50 clutter measurements).
- Four truths at **80 km** on cardinal bearings.

## Plots vs tracks

| JSON field | Meaning |
|------------|---------|
| `plots.total_plots` | **500** detection budget |
| `plots.total_clutter_detections` | ~**50** at 10% clutter |
| `tracks.total_unique_tracks` | **~450** — one row per **true** plot (500 − clutter) |
| `tracks.fused_lifecycle_tracks` | Long **fused** GNN+EKF segments after coast (typically ~7 for 4 truths) |

Implementation: `crates/stonesoup-scenarios/src/dual_radar_clutter.rs`.

Smoke test: `cargo test -p stonesoup-scenarios --test dual_radar_smoke`.
