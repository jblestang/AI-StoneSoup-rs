use crate::error::Error;
use crate::linalg::fixed::{
    add2x2, add4x4, add_vec4, i_minus_kh, inv2, mat4_vec4, mul2x4_4x1, mul2x4_4x2, mul2x4_4x4,
    mul4x2_2x1, mul4x2_2x2, mul4x2_2x4, mul4x4, mul4x4_4x2, sub_vec2, transpose2x4, transpose4,
    transpose4x2,
};
use crate::linalg::Mat;
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
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn new(transition: ConstantVelocity2d) -> Self {
        Self { transition }
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
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

        let mean = mat4_vec4(ws.f, prior.mean)?;
        let fp = mul4x4(ws.f, prior.cov)?;
        let ft = transpose4(ws.f)?;
        let fpft = mul4x4(fp, ft)?;
        let cov = add4x4(fpft, ws.q)?;

        GaussianState::new(mean, cov, timestamp)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KalmanUpdater {
    pub measurement: LinearGaussianMeasurement,
    pub use_joseph: bool,
}

impl KalmanUpdater {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
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

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
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
        let z_pred = mul2x4_4x1(ws.h, predicted.mean)?;
        let innovation = sub_vec2(z, z_pred)?;

        let hp = mul2x4_4x4(ws.h, predicted.cov)?;
        let ht = transpose2x4(ws.h)?;
        let hph = mul2x4_4x2(hp, ht)?;
        let s = add2x2(hph, ws.r)?;
        let s_inv = inv2(s)?;

        let pht = mul4x4_4x2(predicted.cov, ht)?;
        let k = mul4x2_2x2(pht, s_inv)?;

        let k_innov = mul4x2_2x1(k, innovation)?;
        let mean = add_vec4(predicted.mean, k_innov)?;

        let cov = if self.use_joseph {
            let i_kh = i_minus_kh(k, ws.h)?;
            let i_kh_t = transpose4(i_kh)?;
            let term1 = mul4x4(mul4x4(i_kh, predicted.cov)?, i_kh_t)?;
            let kr = mul4x2_2x2(k, ws.r)?;
            let kt = transpose4x2(k)?;
            let term2 = mul4x2_2x4(kr, kt)?;
            add4x4(term1, term2)?
        } else {
            let i_kh = i_minus_kh(k, ws.h)?;
            mul4x4(i_kh, predicted.cov)?
        };

        GaussianState::new(mean, cov, predicted.timestamp)
    }
}
