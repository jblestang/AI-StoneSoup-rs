#![allow(dead_code, clippy::indexing_slicing)] // Reserved for future numeric Jacobians (UKF / finite-diff models).

use crate::error::Error;
use crate::linalg::{require_finite, Mat};

/// Finite-difference step size matching Stone Soup `stonesoup.functions.jacobian`.
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn fd_delta(x: f64) -> f64 {
    let spacing = if x == 0.0 {
        f64::MIN_POSITIVE
    } else {
        libm::fabs(x) * f64::EPSILON
    };
    let mut d = 1e8 * spacing;
    if d < 1e-8 {
        d = 1e-8;
    }
    d
}

/// Jacobian of `f: [f64; N] -> [f64; M]` via forward differences (Stone Soup default).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn jacobian4x2<F>(x: [f64; 4], mut f: F) -> Result<Mat<2, 4>, Error>
where
    F: FnMut([f64; 4]) -> Result<[f64; 2], Error>,
{
    for &v in &x {
        require_finite(v)?;
    }
    let f0 = f(x)?;
    require_finite(f0[0])?;
    require_finite(f0[1])?;

    let mut jac = Mat::<2, 4>::zeros();
    for i in 0..4 {
        let delta = fd_delta(x[i]);
        let mut xp = x;
        xp[i] += delta;
        let fp = f(xp)?;
        let col0 = (fp[0] - f0[0]) / delta;
        let col1 = (fp[1] - f0[1]) / delta;
        require_finite(col0)?;
        require_finite(col1)?;
        jac.set(0, i, col0)?;
        jac.set(1, i, col1)?;
    }
    Ok(jac)
}
