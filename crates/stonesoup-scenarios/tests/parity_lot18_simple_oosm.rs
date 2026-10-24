use stonesoup_scenarios::{
    Lot18OosmMode, Lot18SimpleOosmScenario, MAX_LOT18_SCANS, MAX_LOT18_STEPS,
};

fn assert_mode_parity(mode: Lot18OosmMode, abs_tol: f64) {
    let mut sc = Lot18SimpleOosmScenario::tutorial_defaults(mode).expect("defaults");
    for step in 0..MAX_LOT18_STEPS {
        let more = sc.step().expect("step");
        assert_eq!(
            more,
            step + 1 < MAX_LOT18_STEPS,
            "mode {:?} early finish at {step}",
            mode
        );
        if step == 0 {
            continue;
        }
        let arrival = step - 1;
        let reference = Lot18SimpleOosmScenario::reference_mean_at(arrival, mode).expect("ref");
        let got = sc.estimate_mean();
        for j in 0..4 {
            assert!(
                (got[j] - reference[j]).abs() <= abs_tol,
                "mode {:?} arrival {arrival} dim {j}: got {} ref {}",
                mode,
                got[j],
                reference[j]
            );
        }
    }
    assert_eq!(sc.arrival_index(), MAX_LOT18_SCANS);
}

#[test]
fn naive_arrival_order_matches_python() {
    assert_mode_parity(Lot18OosmMode::Naive, 2e-3);
}

#[test]
fn fixed_lag_oosm_matches_python() {
    assert_mode_parity(Lot18OosmMode::FixedLag, 2e-3);
}

#[test]
fn fixed_lag_closer_to_truth_than_naive_at_final_scan() {
    let mut naive = Lot18SimpleOosmScenario::tutorial_defaults(Lot18OosmMode::Naive).expect("n");
    let mut fixed = Lot18SimpleOosmScenario::tutorial_defaults(Lot18OosmMode::FixedLag).expect("f");
    naive.run_to_end().expect("naive");
    fixed.run_to_end().expect("fixed");
    let last_truth = Lot18SimpleOosmScenario::truth_at(MAX_LOT18_SCANS - 1).expect("truth");
    let err_naive = {
        let m = naive.estimate_mean();
        libm::hypot(m[0] - last_truth[0], m[2] - last_truth[2])
    };
    let err_fixed = {
        let m = fixed.estimate_mean();
        libm::hypot(m[0] - last_truth[0], m[2] - last_truth[2])
    };
    assert!(
        err_fixed <= err_naive + 1e-6,
        "fixed-lag should beat naive: fixed={err_fixed} naive={err_naive}"
    );
}
