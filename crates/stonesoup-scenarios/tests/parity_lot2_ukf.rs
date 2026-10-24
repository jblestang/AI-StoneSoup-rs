use stonesoup_scenarios::Lot2UkfScenario;

const TOL: f64 = 2e-6;

#[test]
fn matches_python_reference_track() {
    let mut scenario = Lot2UkfScenario::tutorial_defaults().expect("defaults");
    for i in 0..stonesoup_scenarios::MAX_LOT2_STEPS {
        scenario.step().expect("step");
        let ref_state = scenario.reference_track_at(i).expect("ref");
        let got = scenario.estimate_state().mean.0;
        for (a, b) in got.iter().zip(ref_state.iter()) {
            assert!(
                (a - b).abs() < TOL,
                "step {i}: got {got:?} ref {ref_state:?}"
            );
        }
    }
}
