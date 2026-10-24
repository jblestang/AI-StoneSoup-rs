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

Co-located **90° FOV** rotating bearing–range sensors, **fused** `MultiTargetTracker`-style loop (GNN + EKF, **3-point** initiation, **30 s** coast delete). Post-run **TrackToTruth** (2 km, 3-of-3 confirm) builds **one continuous track per target**.

- **60 s** simulation (`Δt = 0.25 s`, 240 steps); **500 plots** cap (250 per radar), **~10% clutter** (~50 clutter measurements). After the plot budget is exhausted, radars go quiet but the fused tracker still coasts to the end of the run.
- Four truths at **80 km** on cardinal bearings.

## Plots vs tracks

| JSON field | Meaning |
|------------|---------|
| `plots.total_plots` | **500** detection budget |
| `plots.total_clutter_detections` | ~**50** at 10% clutter |
| `tracks.total_unique_tracks` | **~450** — one row per **true** plot (500 − clutter) |
| `tracks.fused_lifecycle_tracks` | Raw GNN+EKF segment count after coast (may split on one target) |
| `track_to_truth.continuous_track_count` | **4** — one merged track per truth |

Implementation: `crates/stonesoup-scenarios/src/dual_radar_clutter.rs`.

Smoke test: `cargo test -p stonesoup-scenarios --test dual_radar_smoke`.
