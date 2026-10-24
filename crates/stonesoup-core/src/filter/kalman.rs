use crate::error::Error;
use crate::linalg::{identity4, inv2, mat_vec_mul, require_finite, Mat};
use crate::models::measurement::LinearGaussianMeasurement;
use crate::models::transition::ConstantVelocity2d;
use crate::state::GaussianState;
use crate::time::Timestamp;
use crate::types::Detection;

/// Reusable stack workspace for Kalman steps (integrator-owned, init once per loop).
#[derive(Debug, Clone, Copy)]
pub struct KalmanWorkspace {
    pub f: Mat<4, 4>,
    pub q: Mat<4, 4>,
    pub h: Mat<2, 4>,
    pub r: Mat<2, 2>,
}

#[derive(Debug, Clone, Copy)]
pub struct KalmanPredictor {
    pub transition: ConstantVelocity2d,
}

impl KalmanPredictor {
    pub fn new(transition: ConstantVelocity2d) -> Self {
        Self { transition }
    }

    pub fn predict(
        &self,
        prior: &GaussianState,
        timestamp: Timestamp,
        ws: &mut KalmanWorkspace,
    ) -> Result<GaussianState, Error> {
        let dt = timestamp.checked_sub(prior.timestamp)?;
        if dt.0 < 0 {
            return Err(Error::InvalidTimeStep);
        }
        ws.f = self.transition.transition_matrix(dt)?;
        ws.q = self.transition.process_noise(dt)?;

        let mean = mat_vec_mul(ws.f, prior.mean)?;
        let fp = ws.f.mul(prior.cov)?;
        let fpft = fp.mul(ws.f.transpose()?)?;
        let cov = fpft.add(ws.q)?;

        GaussianState::new(mean, cov, timestamp)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KalmanUpdater {
    pub measurement: LinearGaussianMeasurement,
    pub use_joseph: bool,
}

impl KalmanUpdater {
    pub fn new(measurement: LinearGaussianMeasurement) -> Self {
        Self {
            measurement,
            use_joseph: false,
        }
    }

    pub const fn with_joseph(measurement: LinearGaussianMeasurement, use_joseph: bool) -> Self {
        Self {
            measurement,
            use_joseph,
        }
    }

    fn i_minus_kh(k: Mat<4, 2>, h: Mat<2, 4>) -> Result<Mat<4, 4>, Error> {
        let kh = k.mul(h)?;
        let i = identity4()?;
        let mut i_kh = i;
        for r in 0..4 {
            for c in 0..4 {
                let v = i_kh.get(r, c)? - kh.get(r, c)?;
                i_kh.set(r, c, v)?;
            }
        }
        Ok(i_kh)
    }

    pub fn update(
        &self,
        predicted: &GaussianState,
        detection: &Detection,
        ws: &mut KalmanWorkspace,
    ) -> Result<GaussianState, Error> {
        if detection.timestamp != predicted.timestamp {
            return Err(Error::InvalidParameter);
        }
        ws.h = self.measurement.matrix_h()?;
        ws.r = self.measurement.r();

        let z = detection.vector;
        let z_pred = mat_vec_mul(ws.h, predicted.mean)?;
        let innovation = z.sub(z_pred)?;

        let hp = ws.h.mul(predicted.cov)?;
        let hph = hp.mul(ws.h.transpose()?)?;
        let s = hph.add(ws.r)?;
        let s_inv = inv2(s)?;

        let pht = predicted.cov.mul(ws.h.transpose()?)?;
        let k = pht.mul(s_inv)?;

        let k_innov = mat_vec_mul(k, innovation)?;
        let mean = predicted.mean.add(k_innov)?;

        let cov = if self.use_joseph {
            let i_kh = Self::i_minus_kh(k, ws.h)?;
            let term1 = i_kh.mul(predicted.cov)?.mul(i_kh.transpose()?)?;
            let kr = k.mul(ws.r)?;
            let term2 = kr.mul(k.transpose()?)?;
            term1.add(term2)?
        } else {
            let i_kh = Self::i_minus_kh(k, ws.h)?;
            i_kh.mul(predicted.cov)?
        };

        for i in 0..4 {
            for j in 0..4 {
                require_finite(cov.get(i, j)?)?;
            }
        }

        GaussianState::new(mean, cov, predicted.timestamp)
    }
}
