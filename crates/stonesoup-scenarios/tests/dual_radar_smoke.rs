use stonesoup_scenarios::{
    max_simulation_steps, DualRadarClutterSim, DT_S, PLOTS_PER_ROTATION, SIM_DURATION_S,
};

#[test]
fn dual_radar_runs_full_duration_with_per_rotation_plot_cap() {
    let mut sim = DualRadarClutterSim::new().expect("new");
    let stats = sim.run().expect("run");
    assert_eq!(stats.simulation_steps, max_simulation_steps());
    assert!((stats.simulation_steps as f64 * DT_S - SIM_DURATION_S).abs() < 1e-6);
    assert!(
        stats.radar_a_plots > PLOTS_PER_ROTATION,
        "expect more than one rotation of plots on A: {}",
        stats.radar_a_plots
    );
    assert!(
        stats.radar_b_plots > PLOTS_PER_ROTATION,
        "expect more than one rotation of plots on B: {}",
        stats.radar_b_plots
    );
    assert!(
        stats.radar_a_max_plots_in_rotation <= PLOTS_PER_ROTATION,
        "A exceeded per-rotation cap: {}",
        stats.radar_a_max_plots_in_rotation
    );
    assert!(
        stats.radar_b_max_plots_in_rotation <= PLOTS_PER_ROTATION,
        "B exceeded per-rotation cap: {}",
        stats.radar_b_max_plots_in_rotation
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
        (0.07..=0.13).contains(&clutter_frac),
        "clutter fraction {clutter_frac}"
    );
    assert_eq!(
        stats.true_plot_tracks.len(),
        stats.total_true,
        "expect one track row per true plot"
    );
    assert_eq!(stats.total_unique_tracks(), stats.total_true);
    assert!(
        stats.fused_completed.len() < 128,
        "fused lifecycle count {}",
        stats.fused_completed.len()
    );
    assert_eq!(
        stats.continuous_truth_tracks.len(),
        4,
        "TrackToTruth should yield one continuous track per truth"
    );
}
