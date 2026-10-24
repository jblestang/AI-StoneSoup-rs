use stonesoup_scenarios::{Lot3JpdaScenario, MAX_LOT3_JPDA_STEPS};

#[test]
fn matches_python_reference_tracks() {
    const ABS_TOL: f64 = 1e-5;
    for track in 0..2 {
        let mut sc = Lot3JpdaScenario::tutorial_defaults().expect("defaults");
        sc.run_to_end().expect("run");
        for i in 0..MAX_LOT3_JPDA_STEPS {
            let est = sc.reference_track_at(track, i).expect("ref");
            let mut s = Lot3JpdaScenario::tutorial_defaults().expect("defaults");
            while s.step_index() < i + 1 {
                s.step().expect("step");
            }
            let state = s.track_state(track).expect("track");
            for j in 0..4 {
                let got = state.mean.get(j).expect("mean");
                assert!(
                    (got - est[j]).abs() <= ABS_TOL,
                    "track {track} step {i} dim {j}: got {got} ref {}",
                    est[j]
                );
            }
        }
    }
}
