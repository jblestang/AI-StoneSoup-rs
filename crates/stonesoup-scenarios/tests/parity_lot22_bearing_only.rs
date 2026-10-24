use stonesoup_scenarios::{Lot22BearingOnlyScenario, MAX_LOT22_STEPS};

#[test]
fn track_replay_matches_export() {
    let mut sc = Lot22BearingOnlyScenario::tutorial_defaults().expect("defaults");
    for step in 0..MAX_LOT22_STEPS {
        let more = sc.step().expect("step");
        assert_eq!(more, step + 1 < MAX_LOT22_STEPS, "early finish at {step}");
        if step == 0 {
            continue;
        }
        let idx = step - 1;
        let reference = Lot22BearingOnlyScenario::reference_mean_at(idx).expect("ref");
        assert_eq!(sc.estimate_mean(), reference, "step {idx}");
    }
}
