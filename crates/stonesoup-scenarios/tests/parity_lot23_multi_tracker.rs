use stonesoup_scenarios::{
    Lot23MultiTrackerMode, Lot23MultiTrackerScenario, MAX_LOT23_STEPS, MAX_LOT23_TRACKS,
};

fn assert_replay(mode: Lot23MultiTrackerMode) {
    let mut sc = Lot23MultiTrackerScenario::tutorial_defaults(mode).expect("defaults");
    for step in 0..MAX_LOT23_STEPS {
        let more = sc.step().expect("step");
        assert_eq!(
            more,
            step + 1 < MAX_LOT23_STEPS,
            "mode {:?} early finish at {step}",
            mode
        );
        let ref_means =
            Lot23MultiTrackerScenario::reference_track_means_at(step, mode).expect("ref");
        let mut got_means = [[0.0; 4]; MAX_LOT23_TRACKS];
        let n_got = sc.sorted_track_means(&mut got_means);
        assert_eq!(
            n_got,
            ref_means.len(),
            "mode {:?} step {step}: track count mismatch",
            mode
        );
        for (t, ref_row) in ref_means.iter().enumerate() {
            assert_eq!(
                got_means[t], *ref_row,
                "mode {:?} step {step} track {t}",
                mode
            );
        }
    }
}

#[test]
fn ekf_replay_matches_export() {
    assert_replay(Lot23MultiTrackerMode::Ekf);
}

#[test]
fn ukf_replay_matches_export() {
    assert_replay(Lot23MultiTrackerMode::Ukf);
}

#[test]
fn pf_replay_matches_export() {
    assert_replay(Lot23MultiTrackerMode::Pf);
}

#[test]
fn esif_replay_matches_export() {
    assert_replay(Lot23MultiTrackerMode::Esif);
}
