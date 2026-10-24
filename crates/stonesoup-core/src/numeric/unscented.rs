#![allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

use crate::error::Error;
use crate::linalg::{cholesky4, require_finite, Mat, VecN};
use crate::state::GaussianState;

pub const SIGMA_COUNT: usize = 9;
pub const STATE_DIM: usize = 4;

#[derive(Debug, Clone, Copy)]
pub struct SigmaWeights {
    pub mean: [f64; SIGMA_COUNT],
    pub covar: [f64; SIGMA_COUNT],
}

#[derive(Debug, Clone, Copy)]
pub struct UkfParams {
    pub alpha: f64,
    pub beta: f64,
    pub kappa: f64,
}

impl Default for UkfParams {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 2.0,
            kappa: -1.0, // 3 - STATE_DIM
        }
    }
}

impl UkfParams {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn weights(&self) -> Result<(f64, SigmaWeights), Error> {
        let n = STATE_DIM as f64;
        let alpha2 = self.alpha * self.alpha;
        let lamda = alpha2 * (n + self.kappa) - n;
        let c = n + lamda;
        if c <= 0.0 {
            return Err(Error::NumericalFailure);
        }
        let mut mean = [0.0; SIGMA_COUNT];
        let mut covar = [0.0; SIGMA_COUNT];
        mean[0] = lamda / c;
        covar[0] = lamda / c + (1.0 - alpha2 + self.beta);
        for i in 1..SIGMA_COUNT {
            mean[i] = 0.5 / c;
            covar[i] = mean[i];
        }
        Ok((c, SigmaWeights { mean, covar }))
    }
}

/// Generate sigma points from Gaussian mean/cov (Stone Soup `gauss2sigma`).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn gauss2sigma(
    state: &GaussianState,
    params: UkfParams,
) -> Result<([VecN<4>; SIGMA_COUNT], SigmaWeights), Error> {
    let (c, weights) = params.weights()?;
    let scale = libm::sqrt(c);
    let l = cholesky4(state.cov)?;
    let mean = state.mean.0;
    let mut points = [VecN([0.0; 4]); SIGMA_COUNT];
    points[0] = VecN(mean);
    for i in 0..STATE_DIM {
        let mut col = [0.0; 4];
        for r in 0..STATE_DIM {
            col[r] = l.get(r, i)? * scale;
        }
        let mut plus = mean;
        let mut minus = mean;
        for r in 0..STATE_DIM {
            plus[r] += col[r];
            minus[r] -= col[r];
        }
        points[1 + i] = VecN(plus);
        points[1 + STATE_DIM + i] = VecN(minus);
    }
    Ok((points, weights))
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn weighted_mean(points: &[VecN<4>], w: &[f64; SIGMA_COUNT]) -> Result<VecN<4>, Error> {
    let mut m = [0.0; 4];
    for j in 0..SIGMA_COUNT {
        for i in 0..4 {
            m[i] += w[j] * points[j].0[i];
        }
    }
    for v in m {
        require_finite(v)?;
    }
    Ok(VecN(m))
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn sigma2gauss4(
    points: &[VecN<4>],
    w: &SigmaWeights,
    noise: Option<Mat<4, 4>>,
) -> Result<(VecN<4>, Mat<4, 4>), Error> {
    let mean = weighted_mean(points, &w.mean)?;
    let mut cov = Mat::<4, 4>::zeros();
    for j in 0..SIGMA_COUNT {
        for a in 0..4 {
            for b in 0..4 {
                let da = points[j].0[a] - mean.0[a];
                let db = points[j].0[b] - mean.0[b];
                let cur = cov.get(a, b)? + w.covar[j] * da * db;
                cov.set(a, b, cur)?;
            }
        }
    }
    if let Some(q) = noise {
        for a in 0..4 {
            for b in 0..4 {
                cov.set(a, b, cov.get(a, b)? + q.get(a, b)?)?;
            }
        }
    }
    Ok((mean, cov))
}
