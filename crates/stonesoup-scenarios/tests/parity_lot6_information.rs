//! Information filter tutorial parity vs Stone Soup (`lot6_if_fixtures.rs`).

use stonesoup_scenarios::{Lot6InformationScenario, MAX_LOT6_STEPS};

#[test]
fn information_filter_matches_stonesoup_through_end() {
    const ABS_TOL: f64 = 1e-4;
    let mut sc = Lot6InformationScenario::tutorial_defaults().expect("defaults");
    for step in 0..MAX_LOT6_STEPS {
        let outcome = sc.step().expect("step");
        assert!(
            matches!(
                outcome,
                stonesoup_scenarios::Lot6StepOutcome::Stepped
                    | stonesoup_scenarios::Lot6StepOutcome::Finished
            ),
            "step {step}"
        );
        let ref_row = Lot6InformationScenario::reference_track_at(step).expect("ref");
        let est = sc.estimate_state().mean.0;
        for j in 0..4 {
            assert!(
                (est[j] - ref_row[j]).abs() <= ABS_TOL,
                "step {step} dim {j}: got {} ref {}",
                est[j],
                ref_row[j]
            );
        }
        if step + 1 < MAX_LOT6_STEPS {
            assert!(matches!(
                outcome,
                stonesoup_scenarios::Lot6StepOutcome::Stepped
            ));
        }
    }
    assert!(sc.snapshot().expect("snap").finished);
}
