//! Dual rotating radar network — plot and track statistics (JSON report).

use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::ExitCode;

use stonesoup_scenarios::{
    max_simulation_steps, DualRadarClutterSim, CLUTTER_FRACTION_TARGET, CLUTTER_LAMBDA, DT_S,
    FOV_DEG, NUM_TRUTHS, OVERLAP_AREA_FRACTION, TARGETS_PER_ROTATION, PERIOD_A_S, PERIOD_B_S,
    RANGE_A_M, RANGE_B_M, SEED, SIM_DURATION_S, ScenarioError,
};

fn main() -> ExitCode {
    match run() {
        Ok(path) => {
            eprintln!("wrote {path}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("dual_radar_clutter_stats: {e:?}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<String, ScenarioError> {
    let mut sim = DualRadarClutterSim::new()?;
    let stats = sim.run()?;
    let total = stats.total_true + stats.total_clutter;
    let overlap_fraction_of_true =
        stats.true_plots_in_overlap as f64 / stats.total_true.max(1) as f64;
    let mut dur_true = Vec::new();
    let mut dur_false = Vec::new();
    for t in &stats.fused_completed {
        let d = stats.duration_s(t);
        if t.truth_id.is_some() {
            dur_true.push(d);
        } else {
            dur_false.push(d);
        }
    }
    let mean = |v: &[f64]| {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    let median = |v: &mut [f64]| {
        if v.is_empty() {
            0.0
        } else {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            v[v.len() / 2]
        }
    };
    let mut dur_true_sorted = dur_true.clone();
    let mut dur_false_sorted = dur_false.clone();
    let payload = serde_json::json!({
        "parameters": {
            "implementation": "rust",
            "targets_per_rotation_per_radar": TARGETS_PER_ROTATION,
            "clutter_fraction_target": CLUTTER_FRACTION_TARGET,
            "clutter_lambda_poisson": CLUTTER_LAMBDA,
            "overlap_area_fraction_co_located_disks": OVERLAP_AREA_FRACTION,
            "range_a_km": RANGE_A_M / 1e3,
            "range_b_km": RANGE_B_M / 1e3,
            "period_a_s": PERIOD_A_S,
            "period_b_s": PERIOD_B_S,
            "fov_deg": FOV_DEG,
            "num_truths": NUM_TRUTHS,
            "truth_layout": format!(
                "{NUM_TRUTHS} targets uniform random range 50–392 km bearing"
            ),
            "seed": SEED,
            "simulation_duration_s": SIM_DURATION_S,
            "time_step_s": DT_S,
        },
        "plots": {
            "total_plots": total,
            "radar_a_plots_total": stats.radar_a_plots,
            "radar_b_plots_total": stats.radar_b_plots,
            "radar_a_rotations": stats.radar_a_rotations,
            "radar_b_rotations": stats.radar_b_rotations,
            "radar_a_max_targets_in_one_rotation": stats.radar_a_max_targets_in_rotation,
            "radar_b_max_targets_in_one_rotation": stats.radar_b_max_targets_in_rotation,
            "total_true_detections": stats.total_true,
            "total_clutter_detections": stats.total_clutter,
            "true_plots_in_overlap_disk": stats.true_plots_in_overlap,
            "true_plots_in_overlap_fraction": overlap_fraction_of_true,
            "clutter_fraction_actual": if total == 0 { 0.0 } else { stats.total_clutter as f64 / total as f64 },
            "simulation_steps": stats.simulation_steps,
            "simulation_time_s": stats.simulation_steps as f64 * DT_S,
            "max_simulation_steps": max_simulation_steps(),
        },
        "truths": {
            "simulated": NUM_TRUTHS,
            "with_any_plot": stats.truths_with_any_plot,
            "with_labeled_fused_track": stats.truths_with_labeled_fused_track,
            "with_continuous_track_to_truth": stats.continuous_truth_track_count,
        },
        "tracks": {
            "description": "total_unique_tracks = one fused-network row per true plot. fused_lifecycle_tracks = long-coast JPDA+EKF segments (sequential per-sensor).",
            "total_unique_tracks": stats.total_unique_tracks(),
            "true_plot_tracks": stats.true_plot_tracks.len(),
            "fused_lifecycle_tracks": stats.fused_completed.len(),
            "fused_truth_associated_tracks": stats.fused_truth_tracks(),
            "fused_false_tracks": stats.fused_false_tracks(),
            "false_tracks": stats.fused_false_tracks(),
            "mean_duration_s_true": mean(&dur_true),
            "mean_duration_s_false": mean(&dur_false),
            "median_duration_s_true": median(&mut dur_true_sorted),
            "median_duration_s_false": median(&mut dur_false_sorted),
            "durations_s_true": dur_true,
            "durations_s_false": dur_false,
        },
        "track_to_truth": {
            "continuous_track_count": stats.continuous_truth_tracks.len(),
            "continuous_tracks": stats.continuous_truth_tracks.iter().map(|c| {
                serde_json::json!({
                    "truth_id": c.truth_id,
                    "wall_duration_s": (c.wall_end.0 - c.wall_start.0) as f64 / 1e9,
                    "covered_duration_s": c.covered.as_seconds(),
                    "linked_filter_track_ids": c.linked_filter_track_ids,
                })
            }).collect::<Vec<_>>(),
            "association_count": stats.track_to_truth.len(),
        },
    });
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../verification/reports");
    fs::create_dir_all(&root).map_err(scenario_io)?;
    let path = root.join("dual_radar_clutter_stats.json");
    let text = serde_json::to_string_pretty(&payload).map_err(|_| scenario_io(io::Error::other("json")))?;
    fs::write(&path, text).map_err(scenario_io)?;
    Ok(path.display().to_string())
}

fn scenario_io(_e: io::Error) -> ScenarioError {
    ScenarioError::Core(stonesoup_core::Error::InvalidParameter)
}
