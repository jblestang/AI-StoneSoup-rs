//! AKKF tutorial parity vs Stone Soup (`lot9_akkf_fixtures.rs`).

use stonesoup_scenarios::{Lot9AkkfScenario, MAX_LOT9_STEPS};

#[test]
fn akkf_weighted_mean_matches_stonesoup() {
    const ABS_TOL: f64 = 4e-3;
    let mut sc = Lot9AkkfScenario::tutorial_defaults().expect("defaults");
    for step in 0..MAX_LOT9_STEPS {
        let outcome = sc.step().expect("step");
        let ref_row = Lot9AkkfScenario::reference_mean_at(step).expect("ref");
        let snap = sc.snapshot().expect("snap");
        let est = snap.estimate_xy;
        assert!(
            (est.0 - ref_row[0]).abs() <= ABS_TOL,
            "step {step} x: got {} ref {}",
            est.0,
            ref_row[0]
        );
        assert!(
            (est.1 - ref_row[2]).abs() <= ABS_TOL,
            "step {step} y: got {} ref {}",
            est.1,
            ref_row[2]
        );
        if step + 1 < MAX_LOT9_STEPS {
            assert!(matches!(
                outcome,
                stonesoup_scenarios::Lot9StepOutcome::Stepped
            ));
        }
    }
    assert!(sc.snapshot().expect("snap").finished);
}
