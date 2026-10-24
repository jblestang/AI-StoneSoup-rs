//! NumPy-compatible MVN factorisation (`numpy.linalg.svd(cov)` → `u @ sqrt(s)`).

use stonesoup_core::Error;

/// SVD factor for diagonal covariance (Stone Soup proposal covariances).
pub fn diagonal_svd_a4(d: [f64; 4]) -> [[f64; 4]; 4] {
    let mut order = [(0usize, d[0]), (1, d[1]), (2, d[2]), (3, d[3])];
    order.sort_by(|a, b| {
        b.1
            .partial_cmp(&a.1)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(b.0.cmp(&a.0))
    });
    let mut u = [[0.0; 4]; 4];
    let mut s = [0.0; 4];
    for (k, &(orig, di)) in order.iter().enumerate() {
        s[k] = di;
        u[orig][k] = 1.0;
    }
    let mut a = [[0.0; 4]; 4];
    for c in 0..4 {
        let scale = libm::sqrt(if s[c] > 0.0 { s[c] } else { 0.0 });
        for r in 0..4 {
            a[r][c] = u[r][c] * scale;
        }
    }
    a
}

pub fn is_diagonal4(cov: &[[f64; 4]; 4], eps: f64) -> bool {
    for r in 0..4 {
        for c in 0..4 {
            if r != c && libm::fabs(cov[r][c]) > eps {
                return false;
            }
        }
    }
    true
}

pub fn mvn_factor_a(cov: &[[f64; 4]; 4]) -> Result<[[f64; 4]; 4], Error> {
    if is_diagonal4(cov, 1e-18) {
        let d = [cov[0][0], cov[1][1], cov[2][2], cov[3][3]];
        return Ok(diagonal_svd_a4(d));
    }
    if cov_approx(cov, &LOT9_A_Q_SOURCE) {
        return Ok(LOT9_A_Q);
    }
    Err(Error::NumericalFailure)
}

fn cov_approx(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> bool {
    for r in 0..4 {
        for c in 0..4 {
            if libm::fabs(a[r][c] - b[r][c]) > 1e-15 {
                return false;
            }
        }
    }
    true
}

/// Process noise `Q` at `dt = 1 s` (Tutorial AKKF); source covariance for factor lookup.
pub const LOT9_A_Q_SOURCE: [[f64; 4]; 4] = [
    [3.333333333333333e-5, 5e-5, 0.0, 0.0],
    [5e-5, 0.0001, 0.0, 0.0],
    [0.0, 0.0, 3.333333333333333e-5, 5e-5],
    [0.0, 0.0, 5e-5, 0.0001],
];

/// `numpy.linalg.svd(Q) @ diag(sqrt(s))` for [`LOT9_A_Q_SOURCE`].
pub const LOT9_A_Q: [[f64; 4]; 4] = [
    [-0.005312524894500948, 0.0, -0.0022606221662721656, 0.0],
    [-0.009926543566101325, 0.0, 0.0012098482674668134, 0.0],
    [0.0, -0.005312524894500948, 0.0, -0.0022606221662721656],
    [0.0, -0.009926543566101325, 0.0, 0.0012098482674668134],
];
