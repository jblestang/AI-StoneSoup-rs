use crate::error::Error;
use crate::linalg::{require_finite, Mat};
use crate::time::Duration;

/// Combined 2D constant-velocity model on `[x, vx, y, vy]` with independent axis noise q_x, q_y.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantVelocity2d {
    pub q_x: f64,
    pub q_y: f64,
}

impl ConstantVelocity2d {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn new(q_x: f64, q_y: f64) -> Result<Self, Error> {
        if q_x < 0.0 || q_y < 0.0 || !q_x.is_finite() || !q_y.is_finite() {
            return Err(Error::InvalidParameter);
        }
        Ok(Self { q_x, q_y })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn transition_matrix(&self, dt: Duration) -> Result<Mat<4, 4>, Error> {
        let t = dt.as_seconds();
        require_finite(t)?;
        if t < 0.0 {
            return Err(Error::InvalidTimeStep);
        }
        let mut f = Mat::<4, 4>::zeros();
        f.set(0, 0, 1.0)?;
        f.set(0, 1, t)?;
        f.set(1, 1, 1.0)?;
        f.set(2, 2, 1.0)?;
        f.set(2, 3, t)?;
        f.set(3, 3, 1.0)?;
        Ok(f)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    fn q_block_1d(&self, dt: Duration, q: f64) -> Result<Mat<2, 2>, Error> {
        let t = dt.as_seconds();
        require_finite(t)?;
        if t < 0.0 {
            return Err(Error::InvalidTimeStep);
        }
        let t2 = t * t;
        let t3 = t2 * t;
        let mut qm = Mat::<2, 2>::zeros();
        qm.set(0, 0, t3 / 3.0 * q)?;
        qm.set(0, 1, t2 / 2.0 * q)?;
        qm.set(1, 0, t2 / 2.0 * q)?;
        qm.set(1, 1, t * q)?;
        Ok(qm)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn process_noise(&self, dt: Duration) -> Result<Mat<4, 4>, Error> {
        let qx = self.q_block_1d(dt, self.q_x)?;
        let qy = self.q_block_1d(dt, self.q_y)?;
        let mut q = Mat::<4, 4>::zeros();
        q.set(0, 0, qx.0[0][0])?;
        q.set(0, 1, qx.0[0][1])?;
        q.set(1, 0, qx.0[1][0])?;
        q.set(1, 1, qx.0[1][1])?;
        q.set(2, 2, qy.0[0][0])?;
        q.set(2, 3, qy.0[0][1])?;
        q.set(3, 2, qy.0[1][0])?;
        q.set(3, 3, qy.0[1][1])?;
        Ok(q)
    }
}
