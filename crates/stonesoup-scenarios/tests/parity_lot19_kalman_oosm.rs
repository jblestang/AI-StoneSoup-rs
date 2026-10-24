use stonesoup_scenarios::{
    Lot19KalmanOosmMode, Lot19KalmanOosmScenario, MAX_LOT19_LOOPS, MAX_LOT19_STEPS,
};

fn assert_replay(mode: Lot19KalmanOosmMode) {
    let mut sc = Lot19KalmanOosmScenario::tutorial_defaults(mode).expect("defaults");
    for step in 0..MAX_LOT19_STEPS {
        let more = sc.step().expect("step");
        assert_eq!(
            more,
            step + 1 < MAX_LOT19_STEPS,
            "mode {:?} early finish at {step}",
            mode
        );
        if step == 0 {
            continue;
        }
        let idx = step - 1;
        let reference = Lot19KalmanOosmScenario::reference_mean_at(idx, mode).expect("ref");
        let got = sc.estimate_mean();
        assert_eq!(got, reference, "mode {:?} loop {idx}", mode);
    }
    assert_eq!(sc.loop_index(), MAX_LOT19_LOOPS);
}

#[test]
fn tracker1_replay_matches_export() {
    assert_replay(Lot19KalmanOosmMode::Tracker1);
}

#[test]
fn tracker2_replay_matches_export() {
    assert_replay(Lot19KalmanOosmMode::Tracker2);
}

#[test]
fn tracker3_replay_matches_export() {
    assert_replay(Lot19KalmanOosmMode::Tracker3);
}
