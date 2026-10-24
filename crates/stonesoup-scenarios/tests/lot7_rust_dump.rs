//! Emits Rust GM-PHD reduced mixtures as JSON (for Python comparison). Run with:
//! `cargo test -p stonesoup-scenarios --test lot7_rust_dump -- --nocapture`

use stonesoup_scenarios::{Lot7GmPhdScenario, MAX_LOT7_STEPS};

#[test]
fn dump_reduced_json() {
    let mut sc = Lot7GmPhdScenario::tutorial_defaults().expect("defaults");
    let mut steps = Vec::new();
    for step in 0..MAX_LOT7_STEPS {
        let _ = sc.step().expect("step");
        let (comps, n) = sc.snapshot_components();
        let mut components = Vec::new();
        for c in comps.iter().take(n) {
            components.push(serde_json::json!({
                "weight": c.weight,
                "mean": c.state.mean.0,
                "tag": format!("{:?}", c.tag),
            }));
        }
        steps.push(serde_json::json!({
            "step": step,
            "n": n,
            "components": components,
        }));
    }
    println!("{}", serde_json::to_string(&steps).expect("json"));
}
