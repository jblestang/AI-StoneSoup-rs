use stonesoup_scenarios::{Lot3PdaScenario, MAX_LOT3_PDA_STEPS};

#[test]
fn matches_python_reference_track() {
    const ABS_TOL: f64 = 1e-5;
    let mut sc = Lot3PdaScenario::tutorial_defaults().expect("defaults");
    sc.run_to_end().expect("run");
    for i in 0..MAX_LOT3_PDA_STEPS {
        let est = sc.reference_track_at(i).expect("ref");
        let mut s = Lot3PdaScenario::tutorial_defaults().expect("defaults");
        while s.step_index() < i + 1 {
            s.step().expect("step");
        }
        for j in 0..4 {
            let got = s.estimate_state().mean.get(j).expect("mean");
            assert!(
                (got - est[j]).abs() <= ABS_TOL,
                "step {i} dim {j}: got {got} ref {}",
                est[j]
            );
        }
    }
}
