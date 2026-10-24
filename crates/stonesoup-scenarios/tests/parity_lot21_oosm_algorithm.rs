use stonesoup_scenarios::{Lot21OosmAlgorithmMode, Lot21OosmAlgorithmScenario, MAX_LOT21_STEPS};

fn assert_replay(mode: Lot21OosmAlgorithmMode) {
    let mut sc = Lot21OosmAlgorithmScenario::tutorial_defaults(mode).expect("defaults");
    for step in 0..MAX_LOT21_STEPS {
        let more = sc.step().expect("step");
        assert_eq!(
            more,
            step + 1 < MAX_LOT21_STEPS,
            "mode {:?} early finish at {step}",
            mode
        );
        if step == 0 {
            continue;
        }
        let scan = step - 1;
        let reference = Lot21OosmAlgorithmScenario::reference_mean_at(scan, mode).expect("ref");
        let got = sc.estimate_mean();
        assert_eq!(got, reference, "mode {:?} scan {scan}", mode);
    }
    assert_eq!(sc.scan_index(), MAX_LOT21_STEPS - 1);
}

#[test]
fn oosm_target0_replay_matches_export() {
    assert_replay(Lot21OosmAlgorithmMode::OosmTarget0);
}

#[test]
fn oosm_target1_replay_matches_export() {
    assert_replay(Lot21OosmAlgorithmMode::OosmTarget1);
}

#[test]
fn ignore_target0_replay_matches_export() {
    assert_replay(Lot21OosmAlgorithmMode::IgnoreTarget0);
}

#[test]
fn ignore_target1_replay_matches_export() {
    assert_replay(Lot21OosmAlgorithmMode::IgnoreTarget1);
}
