use crate::error::Error;
use crate::linalg::fixed::validate_gaussian;
use crate::linalg::{Mat, VecN};
use crate::time::Timestamp;

/// Gaussian state: x ~ N(mean, cov), state order documented per scenario.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianState {
    pub mean: VecN<4>,
    pub cov: Mat<4, 4>,
    pub timestamp: Timestamp,
}

impl GaussianState {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn new(mean: VecN<4>, cov: Mat<4, 4>, timestamp: Timestamp) -> Result<Self, Error> {
        validate_gaussian(mean, cov)?;
        Ok(Self {
            mean,
            cov,
            timestamp,
        })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn new_from_array(
        mean: [f64; 4],
        cov: Mat<4, 4>,
        timestamp: Timestamp,
    ) -> Result<Self, Error> {
        Self::new(VecN(mean), cov, timestamp)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn position_xy(&self) -> Result<(f64, f64), Error> {
        Ok((self.mean.0[0], self.mean.0[2]))
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn position_cov_block(&self) -> Result<(f64, f64, f64), Error> {
        Ok((self.cov.0[0][0], self.cov.0[0][2], self.cov.0[2][2]))
    }
}
