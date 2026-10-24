use stonesoup_sim::{generate_lot1_tutorial, LOT1_NUMPY_SEED};

#[test]
fn truth_step1_matches_fixture_order_of_magnitude() {
    let data = generate_lot1_tutorial(LOT1_NUMPY_SEED).expect("gen");
    // Lot 1 embedded fixture (Tutorial 01).
    assert!(
        (data.truth[1][0] - 1.177223465328352).abs() < 1e-6,
        "{:?}",
        data.truth[1]
    );
}
