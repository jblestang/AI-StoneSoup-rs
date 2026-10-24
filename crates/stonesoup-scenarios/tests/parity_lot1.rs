use stonesoup_scenarios::{Lot1KalmanScenario, MAX_LOT1_STEPS};

#[test]
fn matches_python_reference_track() {
    let mut sc = Lot1KalmanScenario::tutorial_defaults().expect("defaults");
    sc.run_to_end().expect("run");
    const ABS_TOL: f64 = 1e-6;
    for i in 0..MAX_LOT1_STEPS {
        let est = sc.reference_track_at(i).expect("ref");
        let mut s = Lot1KalmanScenario::tutorial_defaults().expect("defaults");
        while s.step_index() < i + 1 {
            s.step().expect("step");
        }
        for j in 0..4 {
            let got = s.estimate_state().mean.get(j).expect("mean");
            let d = (got - est[j]).abs();
            assert!(d <= ABS_TOL, "step {i} dim {j}: got {got} ref {}", est[j]);
        }
    }
}
