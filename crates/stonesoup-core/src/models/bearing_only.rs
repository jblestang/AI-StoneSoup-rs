use crate::error::Error;
use crate::linalg::require_finite;
use crate::types::mod_bearing;

/// Bearing-only sensor on `(x, y)` indices (`Cartesian2DToBearing`, Tutorial AKKF).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cartesian2DToBearing {
    pub noise_var: f64,
}

impl Cartesian2DToBearing {
    pub fn tutorial_akkf() -> Result<Self, Error> {
        let std = 0.005;
        Ok(Self {
            noise_var: std * std,
        })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn h_xy(&self, x: f64, y: f64) -> Result<f64, Error> {
        require_finite(x)?;
        require_finite(y)?;
        mod_bearing(libm::atan2(y, x))
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn h_state(&self, state: [f64; 4]) -> Result<f64, Error> {
        self.h_xy(state[0], state[2])
    }
}
