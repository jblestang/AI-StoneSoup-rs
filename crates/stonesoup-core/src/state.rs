use crate::error::Error;
use crate::linalg::{require_finite, Mat, VecN};
use crate::time::Timestamp;

/// Gaussian state: x ~ N(mean, cov), state order documented per scenario.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianState {
    pub mean: VecN<4>,
    pub cov: Mat<4, 4>,
    pub timestamp: Timestamp,
}

impl GaussianState {
    pub fn new(mean: VecN<4>, cov: Mat<4, 4>, timestamp: Timestamp) -> Result<Self, Error> {
        for i in 0..4 {
            require_finite(mean.get(i)?)?;
        }
        for i in 0..4 {
            for j in 0..4 {
                require_finite(cov.get(i, j)?)?;
            }
        }
        Ok(Self {
            mean,
            cov,
            timestamp,
        })
    }

    pub fn new_from_array(
        mean: [f64; 4],
        cov: Mat<4, 4>,
        timestamp: Timestamp,
    ) -> Result<Self, Error> {
        Self::new(VecN(mean), cov, timestamp)
    }

    pub fn position_xy(&self) -> Result<(f64, f64), Error> {
        Ok((self.mean.get(0)?, self.mean.get(2)?))
    }

    pub fn position_cov_block(&self) -> Result<(f64, f64, f64), Error> {
        Ok((
            self.cov.get(0, 0)?,
            self.cov.get(0, 2)?,
            self.cov.get(2, 2)?,
        ))
    }
}
