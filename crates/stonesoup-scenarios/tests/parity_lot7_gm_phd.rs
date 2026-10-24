//! GM-PHD tutorial parity (seed 42): step-0 mixture size and strong-mode positions.

use stonesoup_scenarios::{Lot7GmPhdScenario, LOT7_REDUCED, MAX_LOT7_STEPS};

const XY_TOL: f64 = 1.5;
const TOP_WEIGHT_MIN: f64 = 0.2;

fn mean_xy(m: &[f64; 4]) -> (f64, f64) {
    (m[0], m[2])
}

#[test]
fn gm_phd_reduced_mixture_matches_stonesoup_through_end() {
    let mut sc = Lot7GmPhdScenario::tutorial_defaults().expect("defaults");
    for step in 0..MAX_LOT7_STEPS {
        let _ = sc.step().expect("step");
        let (comps, n) = sc.snapshot_components();
        assert!(n > 0, "step {step}: empty mixture");
        if step == 0 {
            let expected = &LOT7_REDUCED[0];
            assert_eq!(
                n, expected.n,
                "step 0: component count mismatch (got {n}, expected {})",
                expected.n
            );
            for exp in expected.components.iter().take(expected.n) {
                if exp.weight < TOP_WEIGHT_MIN {
                    continue;
                }
                let (ex, ey) = mean_xy(&exp.mean);
                let mut ok = false;
                for got in comps.iter().take(n) {
                    if got.weight < TOP_WEIGHT_MIN {
                        continue;
                    }
                    let (gx, gy) = mean_xy(&got.state.mean.0);
                    let d = ((gx - ex).powi(2) + (gy - ey).powi(2)).sqrt();
                    if d <= XY_TOL {
                        ok = true;
                        break;
                    }
                }
                assert!(
                    ok,
                    "step 0: no strong component near expected xy=({ex:.2}, {ey:.2})"
                );
            }
        }
    }
}
