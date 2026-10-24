//! Run scenario tutorials without GUI.

use std::process::ExitCode;

use stonesoup_scenarios::scenario::{AnyScenario, Scenario};
use stonesoup_scenarios::{
    Lot11KdtreeScenario, Lot2EkfScenario, Lot2ParticleScenario, Lot2UkfScenario, Lot3JpdaScenario,
    Lot3MultiTargetScenario, Lot3PdaScenario, Lot4Tutorial09Scenario, Lot4Tutorial10Scenario,
    Lot5ResamplingScenario, Lot6InformationScenario, Lot7GmPhdScenario, Lot8AsdScenario,
    ScenarioId, ScenarioRegistry, MAX_LOT11_KDTREE_STEPS, MAX_LOT3_JPDA_STEPS, MAX_LOT3_MT_STEPS,
    MAX_LOT3_PDA_STEPS, MAX_LOT4_T09_STEPS, MAX_LOT4_T10_STEPS, MAX_LOT5_RESAMPLING_STEPS,
    MAX_LOT6_STEPS, MAX_LOT7_STEPS, MAX_LOT8_STEPS,
};

fn parse_scenario(arg: Option<String>) -> ScenarioKind {
    match arg.as_deref() {
        Some("lot1") => ScenarioKind::Lot1,
        Some("lot2_ekf") => ScenarioKind::Lot2Ekf,
        Some("lot2_ukf") => ScenarioKind::Lot2Ukf,
        Some("lot2_pf") => ScenarioKind::Lot2Particle,
        Some("lot3_clutter") => ScenarioKind::Lot3Clutter,
        Some("lot3_gnn") => ScenarioKind::Lot3Gnn,
        Some("lot3_pda") => ScenarioKind::Lot3Pda,
        Some("lot3_jpda") => ScenarioKind::Lot3Jpda,
        Some("lot4_initiators") => ScenarioKind::Lot4Initiators,
        Some("lot4_simulation") => ScenarioKind::Lot4Simulation,
        Some("lot5_resampling") => ScenarioKind::Lot5Resampling,
        Some("lot6_information") => ScenarioKind::Lot6Information,
        Some("lot7_gm_phd") => ScenarioKind::Lot7Gmphd,
        Some("lot8_asd") => ScenarioKind::Lot8Asd,
        Some("lot11_kdtree") => ScenarioKind::Lot11Kdtree,
        _ => ScenarioKind::Lot1,
    }
}

enum ScenarioKind {
    Lot1,
    Lot2Ekf,
    Lot2Ukf,
    Lot2Particle,
    Lot3Clutter,
    Lot3Gnn,
    Lot3Pda,
    Lot3Jpda,
    Lot4Initiators,
    Lot4Simulation,
    Lot5Resampling,
    Lot6Information,
    Lot7Gmphd,
    Lot8Asd,
    Lot11Kdtree,
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
            let snap = match scenario.viewer_snapshot() {
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
        ScenarioKind::Lot3Clutter => {
            let id = ScenarioId::Lot3ClutterTutorial;
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
            let snap = match scenario.viewer_snapshot() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("snapshot: {e:?}");
                    return ExitCode::from(4);
                }
            };
            println!(
                "scenario=lot3_clutter steps={} rmse={:.6} finished={}",
                snap.step, snap.rmse, snap.finished
            );
        }
        ScenarioKind::Lot3Gnn => {
            let mut scenario = match Lot3MultiTargetScenario::tutorial_defaults() {
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
            let a = scenario.track_state(0).expect("track a");
            println!(
                "scenario=lot3_gnn steps={} finished=true track_a_xy=({:.3}, {:.3})",
                MAX_LOT3_MT_STEPS, a.mean.0[0], a.mean.0[2]
            );
        }
        ScenarioKind::Lot3Pda => {
            let mut scenario = match Lot3PdaScenario::tutorial_defaults() {
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
            let s = scenario.estimate_state();
            println!(
                "scenario=lot3_pda steps={} finished=true estimate_xy=({:.3}, {:.3})",
                MAX_LOT3_PDA_STEPS, s.mean.0[0], s.mean.0[2]
            );
        }
        ScenarioKind::Lot3Jpda => {
            let mut scenario = match Lot3JpdaScenario::tutorial_defaults() {
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
            let a = scenario.track_state(0).expect("track a");
            println!(
                "scenario=lot3_jpda steps={} finished=true track_a_xy=({:.3}, {:.3})",
                MAX_LOT3_JPDA_STEPS, a.mean.0[0], a.mean.0[2]
            );
        }
        ScenarioKind::Lot4Initiators => {
            let mut scenario = match Lot4Tutorial09Scenario::tutorial_defaults() {
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
            let mut means = [[0.0; 4]; 16];
            let n = scenario.snapshot_track_means(&mut means);
            let xy = if n > 0 {
                (means[0][0], means[0][2])
            } else {
                (0.0, 0.0)
            };
            println!(
                "scenario=lot4_initiators steps={} finished=true tracks={} primary_xy=({:.3}, {:.3})",
                MAX_LOT4_T09_STEPS, n, xy.0, xy.1
            );
        }
        ScenarioKind::Lot4Simulation => {
            let mut scenario = match Lot4Tutorial10Scenario::tutorial_defaults() {
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
            let mut means = [[0.0; 4]; 16];
            let n = scenario.snapshot_track_means(&mut means);
            let xy = if n > 0 {
                (means[0][0], means[0][2])
            } else {
                (0.0, 0.0)
            };
            println!(
                "scenario=lot4_simulation steps={} finished=true tracks={} primary_xy=({:.3}, {:.3})",
                MAX_LOT4_T10_STEPS, n, xy.0, xy.1
            );
        }
        ScenarioKind::Lot11Kdtree => {
            let mut scenario = match Lot11KdtreeScenario::tutorial_defaults() {
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
            let mut means = [[0.0; 4]; 16];
            let n = scenario.snapshot_track_means(&mut means);
            let xy = if n > 0 {
                (means[0][0], means[0][2])
            } else {
                (0.0, 0.0)
            };
            println!(
                "scenario=lot11_kdtree steps={} finished=true tracks={} primary_xy=({:.3}, {:.3})",
                MAX_LOT11_KDTREE_STEPS, n, xy.0, xy.1
            );
        }
        ScenarioKind::Lot7Gmphd => {
            let mut scenario = match Lot7GmPhdScenario::tutorial_defaults() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("scenario error: {e:?}");
                    return ExitCode::from(2);
                }
            };
            while matches!(
                scenario.step(),
                Ok(stonesoup_scenarios::Lot7StepOutcome::Stepped)
            ) {}
            let (comps, n) = scenario.snapshot_components();
            let (ex, ey) = if n > 0 {
                (comps[0].state.mean.0[0], comps[0].state.mean.0[2])
            } else {
                (0.0, 0.0)
            };
            println!(
                "scenario=lot7_gm_phd steps={} finished=true components={} top_weight_xy=({:.3}, {:.3})",
                MAX_LOT7_STEPS, n, ex, ey
            );
        }
        ScenarioKind::Lot8Asd => {
            let mut scenario = match Lot8AsdScenario::tutorial_defaults() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("scenario error: {e:?}");
                    return ExitCode::from(2);
                }
            };
            while matches!(
                scenario.step(),
                Ok(stonesoup_scenarios::Lot8StepOutcome::Stepped)
            ) {}
            let snap = scenario.snapshot().expect("snapshot");
            println!(
                "scenario=lot8_asd steps={} finished={} rmse={:.6} estimate_xy=({:.3}, {:.3})",
                MAX_LOT8_STEPS, snap.finished, snap.rmse, snap.estimate_xy.0, snap.estimate_xy.1
            );
        }
        ScenarioKind::Lot6Information => {
            let mut scenario = match Lot6InformationScenario::tutorial_defaults() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("scenario error: {e:?}");
                    return ExitCode::from(2);
                }
            };
            while matches!(
                scenario.step(),
                Ok(stonesoup_scenarios::Lot6StepOutcome::Stepped)
            ) {}
            let snap = scenario.snapshot().expect("snapshot");
            println!(
                "scenario=lot6_information steps={} finished={} rmse={:.6} estimate_xy=({:.3}, {:.3})",
                MAX_LOT6_STEPS,
                snap.finished,
                snap.rmse,
                snap.estimate_xy.0,
                snap.estimate_xy.1
            );
        }
        ScenarioKind::Lot5Resampling => {
            let mut scenario = match Lot5ResamplingScenario::tutorial_defaults() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("scenario error: {e:?}");
                    return ExitCode::from(2);
                }
            };
            while scenario.step().expect("step") {}
            println!(
                "scenario=lot5_resampling steps={} finished=true nparts={} primary={:.3}",
                MAX_LOT5_RESAMPLING_STEPS,
                scenario.nparts(),
                scenario.primary_particle_value()
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
