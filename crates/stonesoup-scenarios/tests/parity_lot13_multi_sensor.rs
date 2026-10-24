use stonesoup_scenarios::{
    Lot13MultiSensorScenario, Lot13SensorManagerMode, MAX_LOT13_STEPS, MAX_LOT13_TRACKS,
};

fn assert_track_parity(mode: Lot13SensorManagerMode, abs_tol: f64) {
    let mut sc = Lot13MultiSensorScenario::tutorial_defaults(mode).expect("defaults");
    for step in 0..MAX_LOT13_STEPS {
        let more = sc.step().expect("step");
        assert_eq!(
            more,
            step + 1 < MAX_LOT13_STEPS,
            "mode {:?} early finish at {step}",
            mode
        );
        let ref_means =
            Lot13MultiSensorScenario::reference_track_means_at(step, mode).expect("ref");
        let mut got_means = [[0.0; 4]; MAX_LOT13_TRACKS];
        let n_got = sc.sorted_track_means(&mut got_means);
        assert_eq!(
            n_got,
            ref_means.len(),
            "mode {:?} step {step}: track count mismatch",
            mode
        );
        for (t, ref_row) in ref_means.iter().enumerate() {
            for j in 0..4 {
                let got = got_means[t][j];
                assert!(
                    (got - ref_row[j]).abs() <= abs_tol,
                    "mode {:?} step {step} track {t} dim {j}: got {got} ref {}",
                    mode,
                    ref_row[j]
                );
            }
        }
    }
}

#[test]
fn random_manager_matches_python_track_means() {
    assert_track_parity(Lot13SensorManagerMode::Random, 5e-3);
}

#[test]
fn brute_force_manager_matches_python_track_means() {
    assert_track_parity(Lot13SensorManagerMode::BruteForce, 2e-3);
}
