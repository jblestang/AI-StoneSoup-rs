use crate::error::Error;
use crate::linalg::{mat_vec_mul, Mat, VecN};
use crate::state::GaussianState;

/// Linear Gaussian measurement z = H x + v, v ~ N(0, R), mapping positions (0, 2) from 4D state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearGaussianMeasurement {
    pub noise_covar: Mat<2, 2>,
}

impl LinearGaussianMeasurement {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn position_2d(noise_covar: Mat<2, 2>) -> Result<Self, Error> {
        crate::linalg::require_finite(noise_covar.0[0][0])?;
        crate::linalg::require_finite(noise_covar.0[0][1])?;
        crate::linalg::require_finite(noise_covar.0[1][0])?;
        crate::linalg::require_finite(noise_covar.0[1][1])?;
        Ok(Self { noise_covar })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn matrix_h(&self) -> Result<Mat<2, 4>, Error> {
        let mut h = Mat::<2, 4>::zeros();
        h.set(0, 0, 1.0)?;
        h.set(1, 2, 1.0)?;
        Ok(h)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn predict_measurement(&self, state: &GaussianState) -> Result<VecN<2>, Error> {
        mat_vec_mul(self.matrix_h()?, state.mean)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn r(&self) -> Mat<2, 2> {
        self.noise_covar
    }
}
