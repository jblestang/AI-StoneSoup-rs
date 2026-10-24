//! Resampling tutorial parity vs Stone Soup (`lot5_resampling_fixtures.rs`).

use stonesoup_scenarios::lot5_resampling_fixtures as fx;
use stonesoup_scenarios::{lot5_case_name, Lot5ResamplingScenario, MAX_LOT5_RESAMPLING_STEPS};
use stonesoup_sim::{
    ess_resample_if_needed_with, multinomial_resample, replay_resampling_tutorial_preamble,
    residual_resample, stratified_resample, systematic_resample, EssInnerResampler, NumpyRng,
    ResidualStage2Method, LOT2_NUMPY_SEED, MAX_RESAMPLE_PARTICLES,
};

const ABS_TOL: f64 = 1e-12;

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
                (parts[i][d] - ref_state[i][d]).abs() <= ABS_TOL,
                "{label} particle[{i}][{d}] got {} ref {}",
                parts[i][d],
                ref_state[i][d]
            );
        }
        assert!(
            (lw[i] - ref_log_w[i]).abs() <= ABS_TOL,
            "{label} log_w[{i}] got {} ref {}",
            lw[i],
            ref_log_w[i]
        );
    }
}

fn assert_scenario_matches_case(sc: &Lot5ResamplingScenario, case: usize) {
    let label = lot5_case_name(case as u8);
    match case {
        0 => assert_scenario_against(
            sc,
            label,
            fx::LOT5_0_NPARTS,
            &fx::LOT5_0_STATE,
            &fx::LOT5_0_LOG_W,
        ),
        1 => assert_scenario_against(
            sc,
            label,
            fx::LOT5_1_NPARTS,
            &fx::LOT5_1_STATE,
            &fx::LOT5_1_LOG_W,
        ),
        2 => assert_scenario_against(
            sc,
            label,
            fx::LOT5_2_NPARTS,
            &fx::LOT5_2_STATE,
            &fx::LOT5_2_LOG_W,
        ),
        3 => assert_scenario_against(
            sc,
            label,
            fx::LOT5_3_NPARTS,
            &fx::LOT5_3_STATE,
            &fx::LOT5_3_LOG_W,
        ),
        4 => assert_scenario_against(
            sc,
            label,
            fx::LOT5_4_NPARTS,
            &fx::LOT5_4_STATE,
            &fx::LOT5_4_LOG_W,
        ),
        5 => assert_scenario_against(
            sc,
            label,
            fx::LOT5_5_NPARTS,
            &fx::LOT5_5_STATE,
            &fx::LOT5_5_LOG_W,
        ),
        _ => panic!("invalid case {case}"),
    }
}

fn assert_scenario_against<const NP: usize, const D: usize>(
    sc: &Lot5ResamplingScenario,
    label: &str,
    nparts: usize,
    ref_state: &[[f64; D]; NP],
    ref_log_w: &[f64; NP],
) {
    assert_eq!(sc.nparts(), nparts, "{label}: nparts mismatch");
    for i in 0..nparts {
        let got = sc.resample_state(i);
        assert!(
            (got - ref_state[i][0]).abs() <= ABS_TOL,
            "{label} particle[{i}] got {got} ref {}",
            ref_state[i][0]
        );
        let lw = sc.resample_log_weight(i);
        assert!(
            (lw - ref_log_w[i]).abs() <= ABS_TOL,
            "{label} log_w[{i}] got {lw} ref {}",
            ref_log_w[i]
        );
    }
}

#[test]
fn scenario_steps_match_stonesoup_through_end() {
    let mut sc = Lot5ResamplingScenario::tutorial_defaults().expect("defaults");
    assert_eq!(sc.completed_steps(), 0);
    assert!(!sc.is_finished());

    for case in 0..MAX_LOT5_RESAMPLING_STEPS {
        let continues = sc.step().expect("step");
        assert_eq!(
            sc.completed_steps(),
            case + 1,
            "step counter after case {case}"
        );
        assert_eq!(
            continues,
            case + 1 < MAX_LOT5_RESAMPLING_STEPS,
            "continue flag at case {case}"
        );
        assert_scenario_matches_case(&sc, case);
    }

    assert!(sc.is_finished());
    assert!(!sc.step().expect("step after finished"));
}

#[test]
fn resampler_primitives_match_stonesoup_fixtures() {
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
