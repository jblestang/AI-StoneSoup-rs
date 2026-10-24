use stonesoup_scenarios::{
    Lot16DataIncestMode, Lot16DataIncestScenario, MAX_LOT16_STEPS, MAX_LOT16_TRACKS,
};

fn assert_fixture_replay(mode: Lot16DataIncestMode) {
    let mut sc = Lot16DataIncestScenario::tutorial_defaults(mode).expect("defaults");
    for step in 0..MAX_LOT16_STEPS {
        let more = sc.step().expect("step");
        assert_eq!(
            more,
            step + 1 < MAX_LOT16_STEPS,
            "mode {:?} early finish at {step}",
            mode
        );
        let ref_means = Lot16DataIncestScenario::reference_track_means_at(step, mode).expect("ref");
        let mut got_means = [[0.0; 4]; MAX_LOT16_TRACKS];
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
fn non_hierarchical_fixture_replay() {
    assert_fixture_replay(Lot16DataIncestMode::NonHierarchical);
}

#[test]
fn hierarchical_fixture_replay() {
    assert_fixture_replay(Lot16DataIncestMode::Hierarchical);
}

#[test]
fn non_hierarchical_fusion_node_has_lower_mean_cov_trace() {
    const ABS_TOL: f64 = 1e-9;
    let mut nh_not_lower = 0usize;
    let mut compared = 0usize;
    for step in 1..MAX_LOT16_STEPS {
        let nh = Lot16DataIncestScenario::reference_mean_cov_trace_at(
            step,
            Lot16DataIncestMode::NonHierarchical,
        )
        .expect("nh");
        let h = Lot16DataIncestScenario::reference_mean_cov_trace_at(
            step,
            Lot16DataIncestMode::Hierarchical,
        )
        .expect("h");
        if nh.is_nan() || h.is_nan() {
            continue;
        }
        compared += 1;
        if nh > h + ABS_TOL {
            nh_not_lower += 1;
        }
    }
    assert!(
        compared > 40,
        "expected many comparable covariance-trace steps"
    );
    assert!(
        nh_not_lower <= 2,
        "NH mean cov trace should be <= H at almost all steps (got {nh_not_lower} violations)"
    );
}
