use stonesoup_scenarios::{Lot24EnsembleFilterScenario, MAX_LOT24_STEPS};

#[test]
fn track_replay_matches_export() {
    let mut sc = Lot24EnsembleFilterScenario::tutorial_defaults().expect("defaults");
    for step in 0..MAX_LOT24_STEPS {
        let more = sc.step().expect("step");
        assert_eq!(more, step + 1 < MAX_LOT24_STEPS, "early finish at {step}");
        if step == 0 {
            continue;
        }
        let idx = step - 1;
        let reference = Lot24EnsembleFilterScenario::reference_mean_at(idx).expect("ref");
        assert_eq!(sc.estimate_mean(), reference, "step {idx}");
    }
}
