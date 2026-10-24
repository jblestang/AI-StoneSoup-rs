use crate::error::Error;
use crate::linalg::require_finite;
use libm::sqrt;

/// Root-mean-square error over position (x, y) samples.
#[derive(Debug, Clone, Copy, Default)]
pub struct RmseAccumulator {
    sum_sq: f64,
    count: u32,
}

impl RmseAccumulator {
    pub const fn new() -> Self {
        Self {
            sum_sq: 0.0,
            count: 0,
        }
    }

    pub fn push(&mut self, ex: f64, ey: f64) -> Result<(), Error> {
        require_finite(ex)?;
        require_finite(ey)?;
        let d2 = ex * ex + ey * ey;
        require_finite(d2)?;
        self.sum_sq += d2;
        self.count = self.count.checked_add(1).ok_or(Error::CapacityExceeded)?;
        Ok(())
    }

    pub fn rmse(&self) -> Result<f64, Error> {
        if self.count == 0 {
            return Err(Error::InvalidParameter);
        }
        let mean = self.sum_sq / (self.count as f64);
        require_finite(mean)?;
        let r = sqrt(mean);
        require_finite(r)?;
        Ok(r)
    }

    pub fn count(&self) -> u32 {
        self.count
    }
}
