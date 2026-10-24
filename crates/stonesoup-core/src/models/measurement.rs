use crate::error::Error;
use crate::linalg::{mat_vec_mul, Mat, VecN};
use crate::state::GaussianState;

/// Linear Gaussian measurement z = H x + v, v ~ N(0, R), mapping positions (0, 2) from 4D state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearGaussianMeasurement {
    pub noise_covar: Mat<2, 2>,
}

impl LinearGaussianMeasurement {
    pub fn position_2d(noise_covar: Mat<2, 2>) -> Result<Self, Error> {
        for i in 0..2 {
            for j in 0..2 {
                crate::linalg::require_finite(noise_covar.get(i, j)?)?;
            }
        }
        Ok(Self { noise_covar })
    }

    pub fn matrix_h(&self) -> Result<Mat<2, 4>, Error> {
        let mut h = Mat::<2, 4>::zeros();
        h.set(0, 0, 1.0)?;
        h.set(1, 2, 1.0)?;
        Ok(h)
    }

    pub fn predict_measurement(&self, state: &GaussianState) -> Result<VecN<2>, Error> {
        mat_vec_mul(self.matrix_h()?, state.mean)
    }

    pub fn r(&self) -> Mat<2, 2> {
        self.noise_covar
    }
}
