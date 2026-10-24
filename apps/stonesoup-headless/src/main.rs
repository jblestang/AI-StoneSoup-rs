//! Run scenario tutorials without GUI.

use std::process::ExitCode;

use stonesoup_scenarios::scenario::{AnyScenario, Scenario};
use stonesoup_scenarios::{
    Lot2EkfScenario, Lot2ParticleScenario, Lot2UkfScenario, ScenarioId, ScenarioRegistry,
};

fn parse_scenario(arg: Option<String>) -> ScenarioKind {
    match arg.as_deref() {
        Some("lot1") => ScenarioKind::Lot1,
        Some("lot2_ekf") => ScenarioKind::Lot2Ekf,
        Some("lot2_ukf") => ScenarioKind::Lot2Ukf,
        Some("lot2_pf") => ScenarioKind::Lot2Particle,
        _ => ScenarioKind::Lot1,
    }
}

enum ScenarioKind {
    Lot1,
    Lot2Ekf,
    Lot2Ukf,
    Lot2Particle,
}

fn main() -> ExitCode {
    let kind = parse_scenario(std::env::args().nth(1));

    let _ = ScenarioRegistry::all();

    match kind {
        ScenarioKind::Lot1 => {
            let id = ScenarioId::Lot1KalmanTutorial;
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
                "scenario=lot1 steps={} rmse={:.6} finished={}",
                snap.step, snap.rmse, snap.finished
            );
        }
        ScenarioKind::Lot2Ekf => {
            let mut scenario = match Lot2EkfScenario::tutorial_defaults() {
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
                "scenario=lot2_ekf steps={} rmse={:.6} finished={}",
                snap.step, snap.rmse, snap.finished
            );
        }
        ScenarioKind::Lot2Ukf => {
            let mut scenario = match Lot2UkfScenario::tutorial_defaults() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("scenario error: {e:?}");
                    return ExitCode::from(2);
                }
            };
            for _ in 0..stonesoup_scenarios::MAX_LOT2_STEPS {
                let _ = scenario.step();
            }
            let state = scenario.estimate_state().mean.0;
            println!(
                "scenario=lot2_ukf finished=true estimate_xy=({:.3}, {:.3})",
                state[0], state[2]
            );
        }
        ScenarioKind::Lot2Particle => {
            let mut scenario = match Lot2ParticleScenario::tutorial_defaults() {
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
            let mean = scenario.mean_state();
            println!(
                "scenario=lot2_pf finished=true estimate_xy=({:.3}, {:.3})",
                mean[0], mean[2]
            );
        }
    }

    ExitCode::SUCCESS
}
