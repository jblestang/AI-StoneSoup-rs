//! Run scenario tutorials without GUI.

use std::process::ExitCode;

use stonesoup_scenarios::scenario::{AnyScenario, Scenario};
use stonesoup_scenarios::{
    Lot10SensorManagerMode, Lot10SingleSensorScenario, Lot11KdtreeScenario,
    Lot13MultiSensorScenario, Lot13SensorManagerMode, Lot14OptimisedSensorMode,
    Lot14OptimisedSensorScenario, Lot15ArchitectureMode, Lot15NetworkInfoArchScenario,
    Lot16DataIncestMode, Lot16DataIncestScenario, Lot17ArchIntroScenario, Lot2EkfScenario,
    Lot2ParticleScenario, Lot2UkfScenario, Lot3JpdaScenario, Lot3MultiTargetScenario,
    Lot3PdaScenario, Lot4Tutorial09Scenario, Lot4Tutorial10Scenario, Lot5ResamplingScenario,
    Lot6InformationScenario, Lot7GmPhdScenario, Lot8AsdScenario, Lot9AkkfScenario, ScenarioId,
    ScenarioRegistry, MAX_LOT10_STEPS, MAX_LOT11_KDTREE_STEPS, MAX_LOT13_STEPS, MAX_LOT14_STEPS,
    MAX_LOT15_STEPS, MAX_LOT16_STEPS, MAX_LOT17_STEPS, MAX_LOT3_JPDA_STEPS, MAX_LOT3_MT_STEPS,
    MAX_LOT3_PDA_STEPS, MAX_LOT4_T09_STEPS, MAX_LOT4_T10_STEPS, MAX_LOT5_RESAMPLING_STEPS,
    MAX_LOT6_STEPS, MAX_LOT7_STEPS, MAX_LOT8_STEPS, MAX_LOT9_STEPS,
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
        Some("lot10_single_sensor") => ScenarioKind::Lot10SingleSensor,
        Some("lot12_akkf") | Some("lot9_akkf") => ScenarioKind::Lot12Akkf,
        Some("lot13_multi_sensor") => ScenarioKind::Lot13MultiSensor,
        Some("lot14_optimised_sensor") => ScenarioKind::Lot14OptimisedSensor,
        Some("lot15_network_info_arch") | Some("lot15_architecture") => {
            ScenarioKind::Lot15NetworkInfoArch
        }
        Some("lot16_data_incest") | Some("lot16_architecture") => ScenarioKind::Lot16DataIncest,
        Some("lot17_arch_intro") | Some("lot17_architecture") => ScenarioKind::Lot17ArchIntro,
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
    Lot10SingleSensor,
    Lot11Kdtree,
    Lot12Akkf,
    Lot13MultiSensor,
    Lot14OptimisedSensor,
    Lot15NetworkInfoArch,
    Lot16DataIncest,
    Lot17ArchIntro,
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
        ScenarioKind::Lot10SingleSensor => {
            let mut scenario = match Lot10SingleSensorScenario::tutorial_defaults(
                Lot10SensorManagerMode::BruteForce,
            ) {
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
            let mut means = [[0.0; 4]; 3];
            let n = scenario.sorted_track_means(&mut means);
            let xy = if n > 0 {
                (means[0][0], means[0][2])
            } else {
                (0.0, 0.0)
            };
            println!(
                "scenario=lot10_single_sensor steps={} finished=true tracks={} primary_xy=({:.3}, {:.3})",
                MAX_LOT10_STEPS, n, xy.0, xy.1
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
        ScenarioKind::Lot13MultiSensor => {
            let mut scenario = match Lot13MultiSensorScenario::tutorial_defaults(
                Lot13SensorManagerMode::BruteForce,
            ) {
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
            let mut means = [[0.0; 4]; 3];
            let n = scenario.sorted_track_means(&mut means);
            let xy = if n > 0 {
                (means[0][0], means[0][2])
            } else {
                (0.0, 0.0)
            };
            println!(
                "scenario=lot13_multi_sensor steps={} finished=true tracks={} primary_xy=({:.3}, {:.3})",
                MAX_LOT13_STEPS, n, xy.0, xy.1
            );
        }
        ScenarioKind::Lot14OptimisedSensor => {
            let mut scenario = match Lot14OptimisedSensorScenario::tutorial_defaults(
                Lot14OptimisedSensorMode::OptimizeBrute,
            ) {
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
            let mut means = [[0.0; 4]; 3];
            let n = scenario.sorted_track_means(&mut means);
            let xy = if n > 0 {
                (means[0][0], means[0][2])
            } else {
                (0.0, 0.0)
            };
            println!(
                "scenario=lot14_optimised_sensor steps={} finished=true tracks={} primary_xy=({:.3}, {:.3})",
                MAX_LOT14_STEPS, n, xy.0, xy.1
            );
        }
        ScenarioKind::Lot15NetworkInfoArch => {
            let mut scenario = match Lot15NetworkInfoArchScenario::tutorial_defaults(
                Lot15ArchitectureMode::Network,
            ) {
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
            let mut means = [[0.0; 4]; 3];
            let n = scenario.sorted_track_means(&mut means);
            let xy = if n > 0 {
                (means[0][0], means[0][2])
            } else {
                (0.0, 0.0)
            };
            println!(
                "scenario=lot15_network_info_arch steps={} finished=true tracks={} primary_xy=({:.3}, {:.3})",
                MAX_LOT15_STEPS, n, xy.0, xy.1
            );
        }
        ScenarioKind::Lot16DataIncest => {
            let mut scenario = match Lot16DataIncestScenario::tutorial_defaults(
                Lot16DataIncestMode::NonHierarchical,
            ) {
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
                "scenario=lot16_data_incest steps={} finished=true mean_cov_trace={:.6}",
                MAX_LOT16_STEPS, snap.rmse
            );
        }
        ScenarioKind::Lot17ArchIntro => {
            let mut scenario = match Lot17ArchIntroScenario::tutorial_defaults() {
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
                "scenario=lot17_arch_intro steps={} finished=true edges={:.0}",
                MAX_LOT17_STEPS, snap.rmse
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
        ScenarioKind::Lot12Akkf => {
            let mut scenario = match Lot9AkkfScenario::tutorial_defaults() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("scenario error: {e:?}");
                    return ExitCode::from(2);
                }
            };
            while matches!(
                scenario.step(),
                Ok(stonesoup_scenarios::Lot9StepOutcome::Stepped)
            ) {}
            let snap = scenario.snapshot().expect("snapshot");
            println!(
                "scenario=lot12_akkf steps={} finished={} rmse={:.6} estimate_xy=({:.3}, {:.3})",
                MAX_LOT9_STEPS, snap.finished, snap.rmse, snap.estimate_xy.0, snap.estimate_xy.1
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
