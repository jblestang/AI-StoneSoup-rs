use stonesoup_scenarios::DualRadarClutterSim;

#[test]
fn dual_radar_runs_to_plot_budget() {
    let mut sim = DualRadarClutterSim::new().expect("new");
    let stats = sim.run().expect("run");
    assert_eq!(stats.radar_a_plots, stonesoup_scenarios::PLOTS_PER_RADAR);
    assert_eq!(stats.radar_b_plots, stonesoup_scenarios::PLOTS_PER_RADAR);
    assert_eq!(
        stats.radar_a_plots + stats.radar_b_plots,
        stonesoup_scenarios::PLOTS_PER_RADAR * 2
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
        stats.fused_completed.len() < 64,
        "fused lifecycle count {}",
        stats.fused_completed.len()
    );
}
