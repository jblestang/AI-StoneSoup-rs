use stonesoup_sim::{
    advance_lot9_preamble, initialise_lot9_filter_rng, multivariate_normal4_batch,
    prior_covariance, NumpyRng, LOT9_NUMPY_SEED,
};

#[test]
fn lot9_prior_first_particle_matches_stonesoup() {
    let (_, particles) = initialise_lot9_filter_rng().expect("init");
    const REF: [f64; 4] = [
        -0.30309776034737074,
        -0.006322902426008042,
        0.5439647891316347,
        -0.06120928423880468,
    ];
    for (a, b) in particles[0].iter().zip(REF.iter()) {
        assert!((a - b).abs() < 1e-6, "got {a} ref {b}");
    }
}

#[test]
fn lot9_mvn_batch_matches_numpy_reference() {
    let cov = prior_covariance().expect("cov");
    let mean = [-0.3, 0.001, 0.7, -0.055];
    let mut rng = NumpyRng::seed(LOT9_NUMPY_SEED);
    advance_lot9_preamble(&mut rng).expect("preamble");
    let mut rng = NumpyRng::seed(LOT9_NUMPY_SEED);
    let samples = multivariate_normal4_batch(&mut rng, mean, &cov, 3).expect("batch");
    const REF0: [f64; 4] = [
        -0.30309776034737074,
        -0.006322902426008042,
        0.5439647891316347,
        -0.06120928423880468,
    ];
    for (a, b) in samples[0].iter().zip(REF0.iter()) {
        assert!((a - b).abs() < 1e-6, "sample0 dim: got {a} ref {b}");
    }
}
