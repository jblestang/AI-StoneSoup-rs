//! Six-dimensional IMM state `[x, vx, ax, y, vy, ay]` with marginal 4D `[x, vx, y, vy]` for JPDA.

use crate::error::Error;
use crate::linalg::{require_finite, Mat, VecN};
use crate::state::GaussianState;
use crate::time::Timestamp;

pub const IMM_STATE_DIM: usize = 6;

/// Ground-truth–aligned CA state; CT/CV use the same layout (ax/ay may be unused in CT).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianState6 {
    pub mean: [f64; IMM_STATE_DIM],
    pub cov: [[f64; IMM_STATE_DIM]; IMM_STATE_DIM],
    pub timestamp: Timestamp,
}

impl GaussianState6 {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn new(
        mean: [f64; IMM_STATE_DIM],
        cov: [[f64; IMM_STATE_DIM]; IMM_STATE_DIM],
        timestamp: Timestamp,
    ) -> Result<Self, Error> {
        for &v in &mean {
            require_finite(v)?;
        }
        for row in &cov {
            for &v in row {
                require_finite(v)?;
            }
        }
        Ok(Self {
            mean,
            cov,
            timestamp,
        })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn from_position_velocity_4d(
        state: &GaussianState,
        accel_var: f64,
    ) -> Result<Self, Error> {
        let m = state.mean.0;
        let mut mean = [0.0; IMM_STATE_DIM];
        mean[0] = m[0];
        mean[1] = m[1];
        mean[3] = m[2];
        mean[4] = m[3];
        let mut cov = [[0.0; IMM_STATE_DIM]; IMM_STATE_DIM];
        let idx4 = [0usize, 1, 2, 3];
        let idx6 = [0usize, 1, 3, 4];
        for (a, &i6) in idx6.iter().enumerate() {
            for (b, &j6) in idx6.iter().enumerate() {
                cov[i6][j6] = state.cov.0[idx4[a]][idx4[b]];
            }
        }
        cov[2][2] = accel_var;
        cov[5][5] = accel_var;
        Self::new(mean, cov, state.timestamp)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn marginalize_to_4d(&self) -> Result<GaussianState, Error> {
        let idx6 = [0usize, 1, 3, 4];
        let mut mean = [0.0; 4];
        for (k, &i) in idx6.iter().enumerate() {
            mean[k] = self.mean[i];
        }
        let mut cov = Mat::<4, 4>::zeros();
        for (a, &i6) in idx6.iter().enumerate() {
            for (b, &j6) in idx6.iter().enumerate() {
                cov.0[a][b] = self.cov[i6][j6];
            }
        }
        GaussianState::new(VecN(mean), cov, self.timestamp)
    }
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn mat6_vec6(m: &[[f64; IMM_STATE_DIM]; IMM_STATE_DIM], v: &[f64; IMM_STATE_DIM]) -> [f64; IMM_STATE_DIM] {
    let mut out = [0.0; IMM_STATE_DIM];
    for r in 0..IMM_STATE_DIM {
        let mut s = 0.0;
        for c in 0..IMM_STATE_DIM {
            s += m[r][c] * v[c];
        }
        out[r] = s;
    }
    out
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn mat6_mul(
    a: &[[f64; IMM_STATE_DIM]; IMM_STATE_DIM],
    b: &[[f64; IMM_STATE_DIM]; IMM_STATE_DIM],
) -> [[f64; IMM_STATE_DIM]; IMM_STATE_DIM] {
    let mut out = [[0.0; IMM_STATE_DIM]; IMM_STATE_DIM];
    for r in 0..IMM_STATE_DIM {
        for c in 0..IMM_STATE_DIM {
            let mut s = 0.0;
            for k in 0..IMM_STATE_DIM {
                s += a[r][k] * b[k][c];
            }
            out[r][c] = s;
        }
    }
    out
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn mat6_transpose(m: &[[f64; IMM_STATE_DIM]; IMM_STATE_DIM]) -> [[f64; IMM_STATE_DIM]; IMM_STATE_DIM] {
    let mut out = [[0.0; IMM_STATE_DIM]; IMM_STATE_DIM];
    for r in 0..IMM_STATE_DIM {
        for c in 0..IMM_STATE_DIM {
            out[r][c] = m[c][r];
        }
    }
    out
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn mat6_add(
    a: &[[f64; IMM_STATE_DIM]; IMM_STATE_DIM],
    b: &[[f64; IMM_STATE_DIM]; IMM_STATE_DIM],
) -> [[f64; IMM_STATE_DIM]; IMM_STATE_DIM] {
    let mut out = [[0.0; IMM_STATE_DIM]; IMM_STATE_DIM];
    for r in 0..IMM_STATE_DIM {
        for c in 0..IMM_STATE_DIM {
            out[r][c] = a[r][c] + b[r][c];
        }
    }
    out
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn predict_linear6(
    prior: &GaussianState6,
    f: &[[f64; IMM_STATE_DIM]; IMM_STATE_DIM],
    q: &[[f64; IMM_STATE_DIM]; IMM_STATE_DIM],
    timestamp: Timestamp,
) -> Result<GaussianState6, Error> {
    if prior.timestamp == timestamp {
        return Ok(*prior);
    }
    let mean = mat6_vec6(f, &prior.mean);
    let fp = mat6_mul(f, &prior.cov);
    let ft = mat6_transpose(f);
    let fpft = mat6_mul(&fp, &ft);
    let cov = mat6_add(&fpft, q);
    GaussianState6::new(mean, cov, timestamp)
}

/// Gaussian mixture reduction for 6D IMM mixing.
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn gm_reduce_6(
    means: &[[f64; IMM_STATE_DIM]],
    covs: &[[[f64; IMM_STATE_DIM]; IMM_STATE_DIM]],
    weights: &[f64],
) -> Result<( [f64; IMM_STATE_DIM], [[f64; IMM_STATE_DIM]; IMM_STATE_DIM]), Error> {
    let n = means.len();
    if n == 0 || n != covs.len() || n != weights.len() {
        return Err(Error::InvalidDimension);
    }
    let mut wsum = 0.0;
    for &w in weights {
        wsum += w;
    }
    if wsum <= 0.0 {
        return Err(Error::NumericalFailure);
    }
    let mut mean = [0.0; IMM_STATE_DIM];
    for (m, &w) in means.iter().zip(weights) {
        let ww = w / wsum;
        for d in 0..IMM_STATE_DIM {
            mean[d] += m[d] * ww;
        }
    }
    let mut cov = [[0.0; IMM_STATE_DIM]; IMM_STATE_DIM];
    for (m, (c, &w)) in means.iter().zip(covs.iter().zip(weights)) {
        let ww = w / wsum;
        for r in 0..IMM_STATE_DIM {
            for col in 0..IMM_STATE_DIM {
                cov[r][col] += ww * c[r][col];
            }
        }
        for r in 0..IMM_STATE_DIM {
            for col in 0..IMM_STATE_DIM {
                let d_r = m[r] - mean[r];
                let d_c = m[col] - mean[col];
                cov[r][col] += ww * d_r * d_c;
            }
        }
    }
    Ok((mean, cov))
}
