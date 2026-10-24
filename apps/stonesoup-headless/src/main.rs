//! Run Lot 1 scenario without GUI; prints final RMSE on stdout.

use std::process::ExitCode;

use stonesoup_scenarios::scenario::{AnyScenario, Scenario};
use stonesoup_scenarios::{ScenarioId, ScenarioRegistry};

fn main() -> ExitCode {
    let id = std::env::args()
        .nth(1)
        .and_then(|s| {
            if s == "lot1" {
                Some(ScenarioId::Lot1KalmanTutorial)
            } else {
                None
            }
        })
        .unwrap_or(ScenarioId::Lot1KalmanTutorial);

    let mut scenario = match AnyScenario::spawn(id) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("scenario error: {e:?}");
            return ExitCode::from(2);
        }
    };

    if scenario.run_to_end().is_err() {
        eprintln!("run failed");
        return ExitCode::from(3);
    }

    let snap = match scenario.snapshot() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("snapshot: {e:?}");
            return ExitCode::from(4);
        }
    };

    println!(
        "scenario={:?} steps={} rmse={:.6} finished={}",
        id, snap.step, snap.rmse, snap.finished
    );

    let _ = ScenarioRegistry::all();
    ExitCode::SUCCESS
}
