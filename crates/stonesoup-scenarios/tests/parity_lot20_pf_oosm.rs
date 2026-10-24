use stonesoup_scenarios::{Lot20PfOosmMode, Lot20PfOosmScenario, MAX_LOT20_SCANS, MAX_LOT20_STEPS};

fn assert_replay(mode: Lot20PfOosmMode) {
    let mut sc = Lot20PfOosmScenario::tutorial_defaults(mode).expect("defaults");
    for step in 0..MAX_LOT20_STEPS {
        let more = sc.step().expect("step");
        assert_eq!(
            more,
            step + 1 < MAX_LOT20_STEPS,
            "mode {:?} early finish at {step}",
            mode
        );
        if step == 0 {
            continue;
        }
        let arrival = step - 1;
        let reference = Lot20PfOosmScenario::reference_mean_at(arrival, mode).expect("ref");
        let got = sc.estimate_mean();
        assert_eq!(got, reference, "mode {:?} arrival {arrival}", mode);
    }
    assert_eq!(sc.arrival_index(), MAX_LOT20_SCANS);
}

#[test]
fn oosm_pf_replay_matches_export() {
    assert_replay(Lot20PfOosmMode::Oosm);
}

#[test]
fn ignore_delayed_pf_replay_matches_export() {
    assert_replay(Lot20PfOosmMode::IgnoreDelayed);
}

#[test]
fn oosm_pf_closer_to_truth_than_ignore_at_final_arrival() {
    let mut oosm = Lot20PfOosmScenario::tutorial_defaults(Lot20PfOosmMode::Oosm).expect("o");
    let mut ignore =
        Lot20PfOosmScenario::tutorial_defaults(Lot20PfOosmMode::IgnoreDelayed).expect("i");
    oosm.run_to_end().expect("oosm");
    ignore.run_to_end().expect("ignore");
    let truth_idx = MAX_LOT20_SCANS - 1;
    let truth = Lot20PfOosmScenario::truth_at(truth_idx).expect("truth");
    let err_oosm = {
        let m = oosm.estimate_mean();
        let dx = m[0] - truth[0];
        let dy = m[2] - truth[2];
        dx * dx + dy * dy
    };
    let err_ignore = {
        let m = ignore.estimate_mean();
        let dx = m[0] - truth[0];
        let dy = m[2] - truth[2];
        dx * dx + dy * dy
    };
    assert!(
        err_oosm <= err_ignore,
        "OOSM PF should be no worse than ignore-OOSM at final step: oosm={err_oosm} ignore={err_ignore}"
    );
}
