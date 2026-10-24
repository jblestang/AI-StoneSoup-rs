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
    assert!(stats.completed.len() >= stonesoup_scenarios::PLOTS_PER_RADAR);
}
