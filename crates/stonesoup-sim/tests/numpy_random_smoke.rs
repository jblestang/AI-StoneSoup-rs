#[test]
fn numpy_seed_first_random_matches_reference() {
    let mut rng = stonesoup_sim::NumpyRng::seed(1991);
    let v = rng.rk_double();
    assert!((v - 0.520_869_569_348_265_8).abs() < 1e-15);
}
