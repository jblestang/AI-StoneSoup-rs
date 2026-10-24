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

- **60 s** simulation (`Δt = 0.25 s`, 240 steps); up to **250 target detections per rotation per radar** (counter resets each 360° sweep). **Clutter is additional** (Poisson λ=0.1 per dwell). ~10 revs on A, ~20 on B in 60 s.
- Four truths at **80 km** on cardinal bearings.

## Plots vs tracks

| JSON field | Meaning |
|------------|---------|
| `plots.total_plots` | Sum over all rotations in 60 s (≫ 500) |
| `plots.radar_*_max_targets_in_one_rotation` | ≤ **250** true-target cap per rotation (clutter not counted) |
| `tracks.total_unique_tracks` | One row per **true** plot (detection bookkeeping) |
| `tracks.fused_lifecycle_tracks` | Raw GNN+EKF segment count after coast (may split on one target) |
| `track_to_truth.continuous_track_count` | **4** — one merged track per truth |

Implementation: `crates/stonesoup-scenarios/src/dual_radar_clutter.rs`.

Smoke test: `cargo test -p stonesoup-scenarios --test dual_radar_smoke`.
