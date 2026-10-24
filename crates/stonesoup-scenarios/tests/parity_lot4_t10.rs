use stonesoup_scenarios::{Lot4Tutorial10Scenario, MAX_LOT4_T10_STEPS};

#[test]
fn matches_python_snapshot_track_means() {
    const ABS_TOL: f64 = 1e-4;
    let mut sc = Lot4Tutorial10Scenario::tutorial_defaults().expect("defaults");
    for step in 0..MAX_LOT4_T10_STEPS {
        let more = sc.step().expect("step");
        assert_eq!(more, step + 1 < MAX_LOT4_T10_STEPS, "early finish at {step}");
        let ref_means = Lot4Tutorial10Scenario::reference_track_means_at(step).expect("ref");
        let mut got_means = [[0.0; 4]; 16];
        let n_got = sc.snapshot_track_means(&mut got_means);
        assert_eq!(n_got, ref_means.len(), "step {step}: track count mismatch");
        for (t, ref_row) in ref_means.iter().enumerate() {
            for j in 0..4 {
                let got = got_means[t][j];
                assert!(
                    (got - ref_row[j]).abs() <= ABS_TOL,
                    "step {step} track {t} dim {j}: got {got} ref {}",
                    ref_row[j]
                );
            }
        }
    }
}
