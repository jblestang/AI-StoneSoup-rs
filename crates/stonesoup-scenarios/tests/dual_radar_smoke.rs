use stonesoup_scenarios::{
    max_simulation_steps, DualRadarClutterSim, DT_S, MAX_ACTIVE, NUM_TRUTHS, SIM_DURATION_S,
    TARGETS_PER_ROTATION,
};

#[test]
fn dual_radar_runs_full_duration_with_per_rotation_target_cap() {
    let mut sim = DualRadarClutterSim::new().expect("new");
    let stats = sim.run().expect("run");
    assert_eq!(stats.simulation_steps, max_simulation_steps());
    assert!((stats.simulation_steps as f64 * DT_S - SIM_DURATION_S).abs() < 1e-6);
    assert!(
        stats.radar_a_max_targets_in_rotation <= TARGETS_PER_ROTATION,
        "A exceeded per-rotation target cap: {}",
        stats.radar_a_max_targets_in_rotation
    );
    assert!(
        stats.radar_b_max_targets_in_rotation <= TARGETS_PER_ROTATION,
        "B exceeded per-rotation target cap: {}",
        stats.radar_b_max_targets_in_rotation
    );
    assert!(
        (9..=11).contains(&stats.radar_a_rotations),
        "A rotations {}",
        stats.radar_a_rotations
    );
    assert!(
        (19..=21).contains(&stats.radar_b_rotations),
        "B rotations {}",
        stats.radar_b_rotations
    );
    let total = stats.total_true + stats.total_clutter;
    let clutter_frac = stats.total_clutter as f64 / total as f64;
    assert!(
        clutter_frac <= 0.15,
        "clutter fraction {clutter_frac}"
    );
    assert_eq!(
        stats.true_plot_tracks.len(),
        stats.total_true,
        "expect one track row per true plot"
    );
    assert_eq!(stats.total_unique_tracks(), stats.total_true);
    assert!(
        stats.fused_completed.len() <= MAX_ACTIVE,
        "fused lifecycle count {}",
        stats.fused_completed.len()
    );
    assert_eq!(NUM_TRUTHS, 1000);
    assert!(
        stats.truths_with_any_plot >= 990,
        "each truth should be plotted at least once over 60 s: {}",
        stats.truths_with_any_plot
    );
    assert!(
        stats.continuous_truth_track_count <= NUM_TRUTHS,
        "continuous {}",
        stats.continuous_truth_track_count
    );
}
