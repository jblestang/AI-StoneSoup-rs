use crate::error::Error;
use crate::linalg::{require_finite, Mat, VecN};
use crate::state::GaussianState;
use crate::types::mod_bearing;

/// Bearing–range sensor at fixed Cartesian offset (Tutorial 02 `CartesianToBearingRange`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CartesianToBearingRange {
    pub sensor_x: f64,
    pub sensor_y: f64,
    pub noise_covar: Mat<2, 2>,
}

impl CartesianToBearingRange {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn tutorial02() -> Result<Self, Error> {
        let deg5 = 5.0 * core::f64::consts::PI / 180.0;
        let bearing_var = deg5 * deg5;
        let mut r = Mat::<2, 2>::zeros();
        r.set(0, 0, bearing_var)?;
        r.set(1, 1, 1.0)?;
        Ok(Self {
            sensor_x: 50.0,
            sensor_y: 0.0,
            noise_covar: r,
        })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn h(&self, state: [f64; 4]) -> Result<[f64; 2], Error> {
        let dx = state[0] - self.sensor_x;
        let dy = state[2] - self.sensor_y;
        require_finite(dx)?;
        require_finite(dy)?;
        let bearing = mod_bearing(libm::atan2(dy, dx))?;
        let range = libm::sqrt(dx * dx + dy * dy);
        require_finite(range)?;
        Ok([bearing, range])
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn predict_measurement(&self, state: &GaussianState) -> Result<VecN<2>, Error> {
        let z = self.h(state.mean.0)?;
        Ok(VecN(z))
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn jacobian(&self, state: &GaussianState) -> Result<Mat<2, 4>, Error> {
        let s = state.mean.0;
        let dx = s[0] - self.sensor_x;
        let dy = s[2] - self.sensor_y;
        require_finite(dx)?;
        require_finite(dy)?;
        let r2 = dx * dx + dy * dy;
        if r2 <= 0.0 {
            return Err(Error::NumericalFailure);
        }
        let r = libm::sqrt(r2);
        let mut h = Mat::<2, 4>::zeros();
        // bearing = atan2(dy, dx)
        h.set(0, 0, -dy / r2)?;
        h.set(0, 2, dx / r2)?;
        // range = sqrt(dx^2 + dy^2)
        h.set(1, 0, dx / r)?;
        h.set(1, 2, dy / r)?;
        Ok(h)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn innovation(z: [f64; 2], z_pred: [f64; 2]) -> Result<[f64; 2], Error> {
        Ok([mod_bearing(z[0] - z_pred[0])?, z[1] - z_pred[1]])
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn r(&self) -> Mat<2, 2> {
        self.noise_covar
    }
}
