use stonesoup_scenarios::{
    Lot10SensorManagerMode, Lot10SingleSensorScenario, MAX_LOT10_STEPS, MAX_LOT10_TRACKS,
};

fn assert_track_parity(mode: Lot10SensorManagerMode) {
    const ABS_TOL: f64 = 2e-3;
    let mut sc = Lot10SingleSensorScenario::tutorial_defaults(mode).expect("defaults");
    for step in 0..MAX_LOT10_STEPS {
        let more = sc.step().expect("step");
        assert_eq!(
            more,
            step + 1 < MAX_LOT10_STEPS,
            "mode {:?} early finish at {step}",
            mode
        );
        let ref_means =
            Lot10SingleSensorScenario::reference_track_means_at(step, mode).expect("ref");
        let mut got_means = [[0.0; 4]; MAX_LOT10_TRACKS];
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
                    (got - ref_row[j]).abs() <= ABS_TOL,
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
    assert_track_parity(Lot10SensorManagerMode::Random);
}

#[test]
fn brute_force_manager_matches_python_track_means() {
    assert_track_parity(Lot10SensorManagerMode::BruteForce);
}
