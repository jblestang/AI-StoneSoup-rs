use stonesoup_sim::{sample_lot2_pf_prior_particle, NumpyRng, LOT2_NUMPY_SEED};

#[test]
fn first_prior_particle_matches_pf_only_export() {
    let mut rng = NumpyRng::seed(LOT2_NUMPY_SEED);
    let p0 = sample_lot2_pf_prior_particle(&mut rng, [0.0, 1.0, 0.0, 1.0]).expect("p0");
    assert!((p0[0] - (-1.155463241555381)).abs() < 1e-6, "{p0:?}");
    assert!((p0[3] - 0.9086432436552871).abs() < 1e-6);
}
