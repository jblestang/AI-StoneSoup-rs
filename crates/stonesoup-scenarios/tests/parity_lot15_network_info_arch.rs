use stonesoup_scenarios::{
    Lot15ArchitectureMode, Lot15NetworkInfoArchScenario, MAX_LOT15_STEPS, MAX_LOT15_TRACKS,
};

fn assert_fixture_replay(mode: Lot15ArchitectureMode) {
    let mut sc = Lot15NetworkInfoArchScenario::tutorial_defaults(mode).expect("defaults");
    for step in 0..MAX_LOT15_STEPS {
        let more = sc.step().expect("step");
        assert_eq!(
            more,
            step + 1 < MAX_LOT15_STEPS,
            "mode {:?} early finish at {step}",
            mode
        );
        let ref_means =
            Lot15NetworkInfoArchScenario::reference_track_means_at(step, mode).expect("ref");
        let mut got_means = [[0.0; 4]; MAX_LOT15_TRACKS];
        let n_got = sc.sorted_track_means(&mut got_means);
        assert_eq!(
            n_got,
            ref_means.len(),
            "mode {:?} step {step}: track count mismatch",
            mode
        );
        for (t, ref_row) in ref_means.iter().enumerate() {
            for j in 0..4 {
                assert_eq!(
                    got_means[t][j], ref_row[j],
                    "mode {:?} step {step} track {t} dim {j}",
                    mode
                );
            }
        }
    }
}

#[test]
fn network_fixture_replay_steps() {
    assert_fixture_replay(Lot15ArchitectureMode::Network);
}

#[test]
fn information_fixture_replay_steps() {
    assert_fixture_replay(Lot15ArchitectureMode::Information);
}

#[test]
fn network_and_information_top_node_tracks_align() {
    const ABS_TOL: f64 = 5e-2;
    for step in 1..MAX_LOT15_STEPS {
        let net = Lot15NetworkInfoArchScenario::reference_track_means_at(
            step,
            Lot15ArchitectureMode::Network,
        )
        .expect("net");
        let info = Lot15NetworkInfoArchScenario::reference_track_means_at(
            step,
            Lot15ArchitectureMode::Information,
        )
        .expect("info");
        assert_eq!(net.len(), info.len(), "track count mismatch at step {step}");
        for (t, (a, b)) in net.iter().zip(info.iter()).enumerate() {
            for j in 0..4 {
                assert!(
                    (a[j] - b[j]).abs() <= ABS_TOL,
                    "step {step} track {t} dim {j}: network {} vs information {}",
                    a[j],
                    b[j]
                );
            }
        }
    }
}

#[test]
fn edge_removed_network_rm_fixture_replay() {
    assert_fixture_replay(Lot15ArchitectureMode::NetworkEdgeRemoved);
}

#[test]
fn edge_removed_information_rm_fixture_replay() {
    assert_fixture_replay(Lot15ArchitectureMode::InformationEdgeRemoved);
}
