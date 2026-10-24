use stonesoup_sim::{generate_lot2_tutorial, LOT2_NUMPY_SEED, LOT2_NUM_STEPS};

#[test]
fn truth_and_measurements_match_ekf_fixtures() {
    let data = generate_lot2_tutorial(LOT2_NUMPY_SEED).expect("gen");
    assert_eq!(data.truth.len(), LOT2_NUM_STEPS);
    // Spot-check against embedded EKF JSON values (Tutorial 02 export).
    assert!((data.truth[1][0] - 1.177223465328352).abs() < 1e-9);
}
