use stonesoup_scenarios::{Lot17ArchIntroScenario, LOT17_INTRO_EDGE_COUNT, MAX_LOT17_STEPS};

#[test]
fn scenario_steps_to_finished_layout() {
    assert_eq!(LOT17_INTRO_EDGE_COUNT, 2);
    let mut sc = Lot17ArchIntroScenario::tutorial_defaults().expect("defaults");
    assert!(sc.step().expect("step"));
    let snap = sc.viewer_snapshot().expect("snap");
    assert_eq!(snap.step, 1);
    assert!((snap.truth_xy.0 - 0.0).abs() < 1e-9);
    assert!((snap.estimate_xy.0 - (-35.0)).abs() < 1e-9);
    assert_eq!(snap.truth_b_xy, Some((35.0, 0.0)));
    assert_eq!(snap.rmse, 2.0);
    assert!(!sc.step().expect("step"));
    assert_eq!(sc.viewer_snapshot().expect("snap").step, MAX_LOT17_STEPS);
}
