use crate::error::Error;
use crate::filter::kalman::KalmanWorkspace;
use crate::linalg::fixed::{
    add_vec4, i_minus_kh, inv2, mul2x4_4x2, mul2x4_4x4, mul4x2_2x1, mul4x2_2x2, mul4x4, mul4x4_4x2,
    transpose2x4,
};
use crate::linalg::VecN;
use crate::models::bearing_range::CartesianToBearingRange;
use crate::state::GaussianState;
use crate::types::Detection;

#[derive(Debug, Clone, Copy)]
pub struct ExtendedKalmanUpdater {
    pub measurement: CartesianToBearingRange,
}

impl ExtendedKalmanUpdater {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn new(measurement: CartesianToBearingRange) -> Self {
        Self { measurement }
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
        ws.h = self.measurement.jacobian(predicted)?;
        ws.r = self.measurement.r();

        let z_pred = self.measurement.predict_measurement(predicted)?;
        let innovation = CartesianToBearingRange::innovation(detection.vector.0, z_pred.0)?;
        let innovation_vec = [innovation[0], innovation[1]];

        let hp = mul2x4_4x4(ws.h, predicted.cov)?;
        let ht = transpose2x4(ws.h)?;
        let hph = mul2x4_4x2(hp, ht)?;
        let s = crate::linalg::fixed::add2x2(hph, ws.r)?;
        let s_inv = inv2(s)?;

        let pht = mul4x4_4x2(predicted.cov, ht)?;
        let k = mul4x2_2x2(pht, s_inv)?;

        let k_innov = mul4x2_2x1(k, VecN(innovation_vec))?;
        let mean = add_vec4(predicted.mean, k_innov)?;

        let i_kh = i_minus_kh(k, ws.h)?;
        let cov = mul4x4(i_kh, predicted.cov)?;

        GaussianState::new(mean, cov, predicted.timestamp)
    }
}
