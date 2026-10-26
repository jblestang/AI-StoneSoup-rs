//! Dual rotating radar network — plot and track statistics (JSON report).

use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::ExitCode;

use stonesoup_scenarios::{
    max_simulation_steps, DualRadarClutterSim, CLUTTER_FRACTION_TARGET, CLUTTER_LAMBDA, DT_S,
    FOV_DEG, NUM_TRUTHS, OVERLAP_AREA_FRACTION, RADAR_BEARING_SIGMA_DEG, RADAR_RANGE_SIGMA_M,
    TARGETS_PER_ROTATION, PERIOD_A_S, PERIOD_B_S, RANGE_A_M, RANGE_B_M, SEED, SIM_DURATION_S,
    TRUTH_MAX_MANEUVER_G, ScenarioError,
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
            "active_filter": "IMM-EKF (3 CV sub-models: q≈40 / 4k / 12k) + JPDA bearing–range",
            "initiator_min_points": stonesoup_scenarios::INITIATOR_MIN_POINTS,
            "initiator_holding_coast_steps": stonesoup_scenarios::HOLDING_TIME_STEPS_DELETER,
            "targets_per_rotation_per_radar": TARGETS_PER_ROTATION,
            "clutter_fraction_target": CLUTTER_FRACTION_TARGET,
            "clutter_lambda_poisson": CLUTTER_LAMBDA,
            "overlap_area_fraction_co_located_disks": OVERLAP_AREA_FRACTION,
            "range_a_km": RANGE_A_M / 1e3,
            "range_b_km": RANGE_B_M / 1e3,
            "period_a_s": PERIOD_A_S,
            "period_b_s": PERIOD_B_S,
            "fov_deg": FOV_DEG,
            "radar_bearing_sigma_deg": RADAR_BEARING_SIGMA_DEG,
            "radar_range_sigma_m": RADAR_RANGE_SIGMA_M,
            "num_truths": NUM_TRUTHS,
            "truth_layout": format!(
                "{NUM_TRUTHS} targets uniform range 50–392 km; speed mix 10% Mach2 … 20% 250kn; CV/CA/CT manoeuvres up to {TRUTH_MAX_MANEUVER_G} g"
            ),
            "truth_max_maneuver_g": stonesoup_scenarios::TRUTH_MAX_MANEUVER_G,
            "truth_motion_models": ["CV", "CA", "CT"],
            "truth_speed_mix": {
                "mach_2_fraction": 0.10,
                "mach_1_fraction": 0.20,
                "800_knots_fraction": 0.30,
                "400_knots_fraction": 0.20,
                "250_knots_fraction": 0.20,
                "mach_2_mps": stonesoup_scenarios::TRUTH_SPEED_MACH2_MPS,
                "mach_1_mps": stonesoup_scenarios::TRUTH_SPEED_MACH1_MPS,
                "800_knots_mps": stonesoup_scenarios::TRUTH_SPEED_800_KNOTS_MPS,
                "400_knots_mps": stonesoup_scenarios::TRUTH_SPEED_400_KNOTS_MPS,
                "250_knots_mps": stonesoup_scenarios::TRUTH_SPEED_250_KNOTS_MPS,
            },
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
            "plot_visibility": {
                "radar_a_only": stats.truth_plot_sensor_counts.radar_a_only,
                "radar_b_only": stats.truth_plot_sensor_counts.radar_b_only,
                "both_radars": stats.truth_plot_sensor_counts.both_radars,
                "no_plots": stats.truth_plot_sensor_counts.no_plots,
            },
            "continuous_track_by_plot_visibility": {
                "radar_a_only": stats.continuous_track_sensor_counts.radar_a_only,
                "radar_b_only": stats.continuous_track_sensor_counts.radar_b_only,
                "both_radars": stats.continuous_track_sensor_counts.both_radars,
            },
            "continuous_track_by_fused_sensor_updates": {
                "radar_a_only": stats.continuous_track_fused_sensor_counts.radar_a_only,
                "radar_b_only": stats.continuous_track_fused_sensor_counts.radar_b_only,
                "both_radars": stats.continuous_track_fused_sensor_counts.both_radars,
            },
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
        "track_quality": {
            "description": "Following error vs ground truth on TrackToTruth-associated samples. identity_switch = filter path jumped to a different truth; filter_handoff = new filter id on same truth.",
            "position_gate": {
                "kind": "bearing_range_chi2",
                "sensor_xy_m": [0.0, 0.0],
                "bearing_sigma_deg": stonesoup_scenarios::RADAR_BEARING_SIGMA_DEG,
                "range_sigma_m": stonesoup_scenarios::RADAR_RANGE_SIGMA_M,
                "chi2_limit": stonesoup_scenarios::TRACK_TO_TRUTH_CHI2_LIMIT,
                "max_range_1sigma_cross_range_m": stonesoup_scenarios::TRUTH_ASSOC_MAX_RANGE_SIGMA_M,
            },
            "accuracy_scale_m": stonesoup_scenarios::dual_radar_track_to_truth_config().accuracy_scale_m(),
            "established_mean_rmse_m": stats.track_quality.established_mean_rmse_m,
            "established_max_rmse_m": stats.track_quality.established_max_rmse_m,
            "mean_quality_score": if stats.track_quality.continuous_truth.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::json!(stats.track_quality.continuous_truth.iter().map(|q| q.quality_score).sum::<f64>()
                    / stats.track_quality.continuous_truth.len() as f64)
            },
            "paths_with_identity_switch": stats.track_quality.paths_with_identity_switch,
            "total_identity_switches": stats.track_quality.total_identity_switches,
            "mean_filter_handoffs_per_truth": stats.track_quality.mean_filter_handoffs_per_truth,
            "max_filter_handoffs_per_truth": stats.track_quality.max_filter_handoffs_per_truth,
            "switch_diagnosis": {
                "description": "Why TrackToTruth reports identity switches. Often label flicker in dense scenes, not filter physically jumping.",
                "t2t_consec_pairs_confirm": stonesoup_scenarios::TRACK_TO_TRUTH_CONSEC_PAIRS,
                "t2t_consec_misses_end": stonesoup_scenarios::TRACK_TO_TRUTH_CONSEC_MISSES,
                "t2t_switch_score_margin": stonesoup_scenarios::TRACK_TO_TRUTH_SWITCH_SCORE_MARGIN,
                "identity_switches_at_confirm_1": stats.switch_diagnosis.identity_switches_t2t,
                "identity_switches_if_confirm_3": stats.switch_diagnosis.identity_switches_confirm3,
                "per_step_nearest_truth_flips": stats.switch_diagnosis.per_step_nearest_truth_flips,
                "ambiguous_nearest_steps": stats.switch_diagnosis.ambiguous_nearest_steps,
                "total_path_samples": stats.switch_diagnosis.total_path_samples,
                "switch_causes": {
                    "ambiguous_nearest_neighbor": stats.switch_diagnosis.switch_causes.ambiguous_nearest_neighbor,
                    "coast_then_relock": stats.switch_diagnosis.switch_causes.coast_then_relock,
                    "diverged_relock": stats.switch_diagnosis.switch_causes.diverged_relock,
                    "unknown": stats.switch_diagnosis.switch_causes.unknown,
                },
                "coast_relock_study": {
                    "description": "CoastThenRelock = at new TrackToTruth segment start, filter is outside the prior truth BR gate but inside the new truth gate.",
                    "event_count": stats.switch_diagnosis.coast_relock.event_count,
                    "filter_closer_to_new_truth": stats.switch_diagnosis.coast_relock.filter_closer_to_new_truth,
                    "prior_in_gate_recently_then_miss": stats.switch_diagnosis.coast_relock.prior_in_gate_recently_then_miss,
                    "assoc_label_gap_ge_1s": stats.switch_diagnosis.coast_relock.assoc_label_gap_ge_1s,
                    "mean_assoc_gap_s": stats.switch_diagnosis.coast_relock.mean_assoc_gap_s,
                    "median_dist_to_prior_truth_m": stats.switch_diagnosis.coast_relock.median_dist_to_prior_truth_m,
                    "median_dist_to_new_truth_m": stats.switch_diagnosis.coast_relock.median_dist_to_new_truth_m,
                    "prior_truth_dist_ge_2km": stats.switch_diagnosis.coast_relock.prior_truth_dist_ge_2km,
                    "ping_pong_track_count": stats.switch_diagnosis.coast_relock.ping_pong_track_count,
                    "ping_pong_event_count": stats.switch_diagnosis.coast_relock.ping_pong_event_count,
                },
                "sample_events": stats.switch_diagnosis.events.iter().map(|e| {
                    serde_json::json!({
                        "track_id": e.track_id,
                        "from_truth": e.from_truth,
                        "to_truth": e.to_truth,
                        "switch_time_s": e.switch_time_s,
                        "dist_to_from_m": e.dist_to_from_m,
                        "dist_to_to_m": e.dist_to_to_m,
                        "second_nearest_gap_m": e.second_nearest_gap_m,
                        "cause": e.cause.as_str(),
                    })
                }).collect::<Vec<_>>(),
            },
            "by_truth": stats.track_quality.continuous_truth.iter().map(|q| {
                serde_json::json!({
                    "truth_id": q.truth_id,
                    "rmse_m": q.rmse_m,
                    "max_error_m": q.max_error_m,
                    "quality_score": q.quality_score,
                    "identity_switch_count": q.identity_switch_count,
                    "filter_handoffs": q.filter_handoffs,
                    "covered_duration_s": q.covered_duration_s,
                    "wall_duration_s": q.wall_duration_s,
                })
            }).collect::<Vec<_>>(),
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
