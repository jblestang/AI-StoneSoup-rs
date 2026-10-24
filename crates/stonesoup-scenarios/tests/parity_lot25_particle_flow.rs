use stonesoup_scenarios::{Lot25ParticleFlowMode, Lot25ParticleFlowScenario, MAX_LOT25_STEPS};

fn assert_replay(mode: Lot25ParticleFlowMode) {
    let mut sc = Lot25ParticleFlowScenario::tutorial_defaults(mode).expect("defaults");
    for step in 0..MAX_LOT25_STEPS {
        let more = sc.step().expect("step");
        assert_eq!(
            more,
            step + 1 < MAX_LOT25_STEPS,
            "mode {:?} early finish at {step}",
            mode
        );
        if step == 0 {
            continue;
        }
        let idx = step - 1;
        let reference = Lot25ParticleFlowScenario::reference_mean_at(idx, mode).expect("ref");
        assert_eq!(sc.estimate_mean(), reference, "mode {:?} step {idx}", mode);
    }
}

#[test]
fn bootstrap_replay_matches_export() {
    assert_replay(Lot25ParticleFlowMode::Bootstrap);
}

#[test]
fn gromov_flow_replay_matches_export() {
    assert_replay(Lot25ParticleFlowMode::GromovFlow);
}

#[test]
fn parallel_ekf_flow_replay_matches_export() {
    assert_replay(Lot25ParticleFlowMode::ParallelEkf);
}
