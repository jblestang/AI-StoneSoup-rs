use stonesoup_sim::{mvn_factor_a, prior_covariance};

#[test]
fn prior_mvn_a_matches_numpy() {
    let cov = prior_covariance().expect("cov");
    let a = mvn_factor_a(&cov).expect("a");
    const NUMPY_A: [[f64; 4]; 4] = [
        [0.0, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.005],
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.01, 0.0],
    ];
    for r in 0..4 {
        for c in 0..4 {
            assert!(
                (a[r][c] - NUMPY_A[r][c]).abs() < 1e-6,
                "A[{r},{c}] got {} numpy {}",
                a[r][c],
                NUMPY_A[r][c]
            );
        }
    }
}
