//! ASD filter tutorial parity vs Stone Soup (`lot8_asd_fixtures.rs`).

use stonesoup_scenarios::{Lot8AsdScenario, MAX_LOT8_STEPS};

#[test]
fn asd_filter_act_position_matches_stonesoup() {
    const ABS_TOL: f64 = 1e-3;
    let mut sc = Lot8AsdScenario::tutorial_defaults().expect("defaults");
    for step in 0..MAX_LOT8_STEPS {
        let outcome = sc.step().expect("step");
        assert!(
            matches!(
                outcome,
                stonesoup_scenarios::Lot8StepOutcome::Stepped
                    | stonesoup_scenarios::Lot8StepOutcome::Finished
            ),
            "step {step}"
        );
        let ref_row = Lot8AsdScenario::reference_act_mean_at(step).expect("ref");
        let est = sc.act_mean();
        assert!(
            (est[0] - ref_row[0]).abs() <= ABS_TOL,
            "step {step} x: got {} ref {}",
            est[0],
            ref_row[0]
        );
        assert!(
            (est[2] - ref_row[2]).abs() <= ABS_TOL,
            "step {step} y: got {} ref {}",
            est[2],
            ref_row[2]
        );
        if step + 1 < MAX_LOT8_STEPS {
            assert!(matches!(
                outcome,
                stonesoup_scenarios::Lot8StepOutcome::Stepped
            ));
        }
    }
    assert!(sc.snapshot().expect("snap").finished);
}
