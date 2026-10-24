//! GM-PHD tutorial parity (seed 42): reduced mixture vs Stone Soup export.

use stonesoup_scenarios::{Lot7GmPhdScenario, LOT7_REDUCED, MAX_LOT7_STEPS};

const WTOL: f64 = 1e-5;
const MTOL: f64 = 1e-4;

fn sort_key(weight: f64, mean: &[f64; 4]) -> (i64, i64, i64) {
    (
        -(weight * 1e12) as i64,
        (mean[0] * 1e6) as i64,
        (mean[2] * 1e6) as i64,
    )
}

#[test]
fn gm_phd_reduced_mixture_matches_stonesoup_through_end() {
    let mut sc = Lot7GmPhdScenario::tutorial_defaults().expect("defaults");
    for step in 0..MAX_LOT7_STEPS {
        let _ = sc.step().expect("step");
        let (comps, n) = sc.snapshot_components();
        let expected = &LOT7_REDUCED[step];
        assert_eq!(
            n, expected.n,
            "step {step}: component count (got {n}, expected {})",
            expected.n
        );
        let mut got_sorted: Vec<_> = comps.iter().take(n).collect();
        let mut exp_sorted: Vec<_> = expected.components.iter().take(expected.n).collect();
        got_sorted.sort_by_key(|c| sort_key(c.weight, &c.state.mean.0));
        exp_sorted.sort_by_key(|c| sort_key(c.weight, &c.mean));
        for (got, exp) in got_sorted.iter().zip(exp_sorted.iter()) {
            assert!(
                (got.weight - exp.weight).abs() <= WTOL,
                "step {step}: weight delta {} (got {}, exp {})",
                (got.weight - exp.weight).abs(),
                got.weight,
                exp.weight
            );
            for i in 0..4 {
                assert!(
                    (got.state.mean.0[i] - exp.mean[i]).abs() <= MTOL,
                    "step {step}: mean[{i}] delta",
                );
            }
        }
    }
}
