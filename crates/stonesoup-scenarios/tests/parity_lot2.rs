use stonesoup_scenarios::Lot2EkfScenario;

/// Stone Soup EKF uses numeric Jacobians; analytic bearing–range H differs slightly on step 0.
const TOL: f64 = 2e-6;

#[test]
fn matches_python_reference_track() {
    let mut scenario = Lot2EkfScenario::tutorial_defaults().expect("defaults");
    for i in 0..stonesoup_scenarios::MAX_LOT2_STEPS {
        scenario.step().expect("step");
        let ref_state = scenario.reference_track_at(i).expect("ref");
        let got = scenario.estimate_state().mean.0;
        for (j, (a, b)) in got.iter().zip(ref_state.iter()).enumerate() {
            assert!(
                (a - b).abs() < TOL,
                "step {i} dim {j}: got {got:?} ref {ref_state:?}"
            );
        }
    }
}
