//! Pre-reduction mixture size matches Stone Soup export each step.

use std::fs;
use std::path::PathBuf;

use stonesoup_scenarios::{Lot7GmPhdScenario, MAX_LOT7_STEPS};

#[test]
fn pre_reduce_counts_match_stonesoup_export() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("verification/fixtures/lot7_gm_phd.json");
    let payload: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(path).expect("json")).expect("parse");
    let expected: Vec<usize> = payload["pre_reduce_by_step"]
        .as_array()
        .expect("pre_reduce_by_step")
        .iter()
        .map(|v| v.as_u64().expect("n") as usize)
        .collect();

    let mut sc = Lot7GmPhdScenario::tutorial_defaults().expect("defaults");
    for step in 0..MAX_LOT7_STEPS.min(expected.len()) {
        sc.step().expect("step");
        assert_eq!(
            sc.last_pre_reduce_n, expected[step],
            "step {step} pre-reduce component count"
        );
    }
}
