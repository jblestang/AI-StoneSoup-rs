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
    /// Antenna boresight in world frame (rad), Stone Soup ``dwell_centre`` + orientation.
    pub antenna_heading_rad: f64,
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
            antenna_heading_rad: 0.0,
        })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    fn rotated_xy(&self, state: [f64; 4]) -> Result<(f64, f64), Error> {
        let dx = state[0] - self.sensor_x;
        let dy = state[2] - self.sensor_y;
        require_finite(dx)?;
        require_finite(dy)?;
        let sin_a = libm::sin(self.antenna_heading_rad);
        let cos_a = libm::cos(self.antenna_heading_rad);
        let rx = cos_a * dx + sin_a * dy;
        let ry = -sin_a * dx + cos_a * dy;
        require_finite(rx)?;
        require_finite(ry)?;
        Ok((rx, ry))
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    /// Measurement from 6D IMM state `[x, vx, ax, y, vy, ay]`.
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn h6(&self, state: &[f64; 6]) -> Result<[f64; 2], Error> {
        let s4 = [state[0], state[1], state[3], state[4]];
        self.h(s4)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn jacobian6(&self, state: &[f64; 6]) -> Result<Mat<2, 6>, Error> {
        let s4 = [state[0], state[1], state[3], state[4]];
        if self.antenna_heading_rad != 0.0 {
            let model = *self;
            return crate::numeric::jacobian::jacobian6x2(*state, |x| model.h6(&x));
        }
        let dx = s4[0] - self.sensor_x;
        let dy = s4[2] - self.sensor_y;
        require_finite(dx)?;
        require_finite(dy)?;
        let r2 = dx * dx + dy * dy;
        if r2 <= 0.0 {
            return Err(Error::NumericalFailure);
        }
        let r = libm::sqrt(r2);
        let mut h = Mat::<2, 6>::zeros();
        h.set(0, 0, -dy / r2)?;
        h.set(0, 3, dx / r2)?;
        h.set(1, 0, dx / r)?;
        h.set(1, 3, dy / r)?;
        Ok(h)
    }

    pub fn h(&self, state: [f64; 4]) -> Result<[f64; 2], Error> {
        let (rx, ry) = self.rotated_xy(state)?;
        let bearing = mod_bearing(libm::atan2(ry, rx))?;
        let range = libm::sqrt(rx * rx + ry * ry);
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
        if self.antenna_heading_rad != 0.0 {
            let s = state.mean.0;
            let model = *self;
            return crate::numeric::jacobian::jacobian4x2(s, |x| model.h(x));
        }
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
        h.set(0, 0, -dy / r2)?;
        h.set(0, 2, dx / r2)?;
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
