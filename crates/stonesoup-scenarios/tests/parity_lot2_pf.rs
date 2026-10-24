use stonesoup_scenarios::Lot2ParticleScenario;

/// Particle filter stochastic parity (fixed seed, 1000 particles).
const TOL: f64 = 1e-5;

#[test]
fn matches_python_reference_track() {
    let mut scenario = Lot2ParticleScenario::tutorial_defaults().expect("defaults");
    for i in 0..stonesoup_scenarios::MAX_LOT2_STEPS {
        scenario.step().expect("step");
        let ref_state = scenario.reference_mean_at(i).expect("ref");
        let got = scenario.mean_state();
        for (j, (&a, &b)) in got.iter().zip(ref_state.iter()).enumerate() {
            assert!(
                (a - b).abs() < TOL,
                "step {i} dim {j}: got {got:?} ref {ref_state:?}"
            );
        }
    }
}
