//! Resampling tutorial parity vs Stone Soup (`lot5_resampling_fixtures.rs`).

use stonesoup_scenarios::lot5_resampling_fixtures as fx;
use stonesoup_sim::{
    ess_resample_if_needed_with, multinomial_resample, replay_resampling_tutorial_preamble,
    residual_resample, stratified_resample, systematic_resample, EssInnerResampler, NumpyRng,
    ResidualStage2Method, LOT2_NUMPY_SEED, MAX_RESAMPLE_PARTICLES,
};

fn init_equal_particles(
    parts: &mut [[f64; 4]; MAX_RESAMPLE_PARTICLES],
    lw: &mut [f64; MAX_RESAMPLE_PARTICLES],
) {
    for i in 0..5 {
        parts[i][0] = i as f64;
        lw[i] = libm::log(0.2);
    }
}

fn init_weighted(
    parts: &mut [[f64; 4]; MAX_RESAMPLE_PARTICLES],
    lw: &mut [f64; MAX_RESAMPLE_PARTICLES],
) {
    let w = [0.1, 0.5, 0.15, 0.05, 0.2];
    for i in 0..5 {
        parts[i][0] = i as f64;
        lw[i] = libm::log(w[i]);
    }
}

fn assert_state<const NP: usize, const D: usize>(
    label: &str,
    parts: &[[f64; 4]; MAX_RESAMPLE_PARTICLES],
    lw: &[f64; MAX_RESAMPLE_PARTICLES],
    ref_state: &[[f64; D]; NP],
    ref_log_w: &[f64; NP],
) {
    for i in 0..NP {
        for d in 0..D {
            assert!(
                (parts[i][d] - ref_state[i][d]).abs() < 1e-12,
                "{label} particle[{i}][{d}] got {} ref {}",
                parts[i][d],
                ref_state[i][d]
            );
        }
        assert!(
            (lw[i] - ref_log_w[i]).abs() < 1e-12,
            "{label} log_w[{i}] got {} ref {}",
            lw[i],
            ref_log_w[i]
        );
    }
}

#[test]
fn resampling_tutorial_matches_stonesoup() {
    let mut parts = [[0.0f64; 4]; MAX_RESAMPLE_PARTICLES];
    let mut lw = [0.0f64; MAX_RESAMPLE_PARTICLES];

    let mut rng = NumpyRng::seed(LOT2_NUMPY_SEED);
    replay_resampling_tutorial_preamble(&mut rng, 5);
    init_equal_particles(&mut parts, &mut lw);
    multinomial_resample(&mut parts, &mut lw, 5, 5, &mut rng).unwrap();
    assert_state(
        "multinomial_equal_n5",
        &parts,
        &lw,
        &fx::LOT5_0_STATE,
        &fx::LOT5_0_LOG_W,
    );

    init_equal_particles(&mut parts, &mut lw);
    multinomial_resample(&mut parts, &mut lw, 5, 10, &mut rng).unwrap();
    assert_state(
        "multinomial_upsample_n10",
        &parts,
        &lw,
        &fx::LOT5_1_STATE,
        &fx::LOT5_1_LOG_W,
    );

    rng = NumpyRng::seed(LOT2_NUMPY_SEED);
    replay_resampling_tutorial_preamble(&mut rng, 5);
    init_equal_particles(&mut parts, &mut lw);
    systematic_resample(&mut parts, &mut lw, 5, 5, &mut rng).unwrap();
    assert_state(
        "systematic_equal_n5",
        &parts,
        &lw,
        &fx::LOT5_2_STATE,
        &fx::LOT5_2_LOG_W,
    );

    rng = NumpyRng::seed(LOT2_NUMPY_SEED);
    replay_resampling_tutorial_preamble(&mut rng, 5);
    init_equal_particles(&mut parts, &mut lw);
    stratified_resample(&mut parts, &mut lw, 5, 5, &mut rng).unwrap();
    assert_state(
        "stratified_equal_n5",
        &parts,
        &lw,
        &fx::LOT5_3_STATE,
        &fx::LOT5_3_LOG_W,
    );

    rng = NumpyRng::seed(LOT2_NUMPY_SEED);
    replay_resampling_tutorial_preamble(&mut rng, 5);
    init_weighted(&mut parts, &mut lw);
    residual_resample(
        &mut parts,
        &mut lw,
        5,
        ResidualStage2Method::Systematic,
        &mut rng,
    )
    .unwrap();
    assert_state(
        "residual_systematic_weighted_n5",
        &parts,
        &lw,
        &fx::LOT5_4_STATE,
        &fx::LOT5_4_LOG_W,
    );

    rng = NumpyRng::seed(LOT2_NUMPY_SEED);
    replay_resampling_tutorial_preamble(&mut rng, 5);
    init_equal_particles(&mut parts, &mut lw);
    let _ = ess_resample_if_needed_with(
        &mut parts,
        &mut lw,
        5,
        2.5,
        &mut rng,
        EssInnerResampler::ResidualSystematic,
    )
    .unwrap();
    assert_state(
        "ess_residual_equal_n5",
        &parts,
        &lw,
        &fx::LOT5_5_STATE,
        &fx::LOT5_5_LOG_W,
    );
}
