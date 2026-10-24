//! Information-form Kalman predict/update (Stone Soup information filter tutorial).

use crate::error::Error;
use crate::linalg::fixed::{
    add4x4, inv2, inv4, mat4_vec4, mul4x2_2x1, mul4x2_2x2, mul4x2_2x4, mul4x4, symmetrize4,
    transpose2x4, transpose4,
};
use crate::linalg::{Mat, VecN};
use crate::models::measurement::LinearGaussianMeasurement;
use crate::models::transition::ConstantVelocity2d;
use crate::state::GaussianState;
use crate::time::Timestamp;
use crate::types::Detection;

/// Information state :math:`y = Y x`, precision :math:`Y = P^{-1}`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InformationState {
    pub information: VecN<4>,
    pub precision: Mat<4, 4>,
    pub timestamp: Timestamp,
}

impl InformationState {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn new(
        information: VecN<4>,
        precision: Mat<4, 4>,
        timestamp: Timestamp,
    ) -> Result<Self, Error> {
        Ok(Self {
            information,
            precision,
            timestamp,
        })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn from_gaussian(state: &GaussianState) -> Result<Self, Error> {
        let precision = inv4(state.cov)?;
        let information = mat4_vec4(precision, state.mean)?;
        Ok(Self {
            information,
            precision,
            timestamp: state.timestamp,
        })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn to_gaussian(&self) -> Result<GaussianState, Error> {
        let cov = inv4(self.precision)?;
        let mean = mat4_vec4(cov, self.information)?;
        GaussianState::new(mean, cov, self.timestamp)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InformationKalmanWorkspace {
    pub f: Mat<4, 4>,
    pub q: Mat<4, 4>,
    pub h: Mat<2, 4>,
    pub r: Mat<2, 2>,
}

#[derive(Debug, Clone, Copy)]
pub struct InformationKalmanPredictor {
    pub transition: ConstantVelocity2d,
}

impl InformationKalmanPredictor {
    pub fn new(transition: ConstantVelocity2d) -> Self {
        Self { transition }
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn predict(
        &self,
        prior: &InformationState,
        timestamp: Timestamp,
        ws: &mut InformationKalmanWorkspace,
    ) -> Result<InformationState, Error> {
        let dt = timestamp.checked_sub(prior.timestamp)?;
        if dt.0 < 0 {
            return Err(Error::InvalidTimeStep);
        }
        ws.f = self.transition.transition_matrix(dt)?;
        ws.q = self.transition.process_noise(dt)?;
        let f_inv = inv4(ws.f)?;
        let f_inv_t = transpose4(f_inv)?;
        let y = prior.precision;
        let mk = mul4x4(mul4x4(f_inv_t, y)?, f_inv)?;
        let mq = mul4x4(mk, ws.q)?;
        let i = crate::linalg::fixed::identity4()?;
        let ck = inv4(add4x4(i, mq)?)?;
        let pred_precision = mul4x4(ck, mk)?;
        let y_tmp = mat4_vec4(f_inv_t, prior.information)?;
        let pred_information = mat4_vec4(ck, y_tmp)?;
        InformationState::new(pred_information, pred_precision, timestamp)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InformationKalmanUpdater {
    pub measurement: LinearGaussianMeasurement,
}

impl InformationKalmanUpdater {
    pub fn new(measurement: LinearGaussianMeasurement) -> Self {
        Self { measurement }
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn update(
        &self,
        predicted: &InformationState,
        detection: &Detection,
    ) -> Result<InformationState, Error> {
        if detection.timestamp != predicted.timestamp {
            return Err(Error::InvalidParameter);
        }
        let h = self.measurement.matrix_h()?;
        let r = self.measurement.r();
        let inv_r = inv2(r)?;
        let ht = transpose2x4(h)?;
        let ht_inv_r = mul4x2_2x2(ht, inv_r)?;
        let ht_inv_r_h = mul4x2_2x4(ht_inv_r, h)?;
        let mut posterior_precision = add4x4(predicted.precision, ht_inv_r_h)?;
        posterior_precision = symmetrize4(posterior_precision)?;
        let innov = mul4x2_2x1(ht_inv_r, detection.vector)?;
        let posterior_information = crate::linalg::fixed::add_vec4(predicted.information, innov)?;
        InformationState::new(
            posterior_information,
            posterior_precision,
            detection.timestamp,
        )
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn update_gaussian(
        &self,
        predicted: &InformationState,
        detection: &Detection,
        ws: &mut InformationKalmanWorkspace,
    ) -> Result<GaussianState, Error> {
        let _ = ws;
        self.update(predicted, detection)?.to_gaussian()
    }
}
