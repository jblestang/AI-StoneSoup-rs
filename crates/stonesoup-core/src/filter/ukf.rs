#![allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

use crate::error::Error;
use crate::filter::kalman::KalmanWorkspace;
use crate::linalg::fixed::{
    add_vec4, inv2, mul4x2_2x1, mul4x2_2x2, mul4x2_2x4, sub4x4, transpose4x2,
};
use crate::linalg::{Mat, VecN};
use crate::models::bearing_range::CartesianToBearingRange;
use crate::models::transition::ConstantVelocity2d;
use crate::numeric::unscented::{gauss2sigma, sigma2gauss4, SigmaWeights, UkfParams, SIGMA_COUNT};
use crate::state::GaussianState;
use crate::time::Timestamp;
use crate::types::mod_bearing;
use crate::types::Detection;

#[derive(Debug, Clone, Copy)]
pub struct UnscentedKalmanPredictor {
    pub transition: ConstantVelocity2d,
    pub params: UkfParams,
}

impl UnscentedKalmanPredictor {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn new(transition: ConstantVelocity2d) -> Self {
        Self {
            transition,
            params: UkfParams::default(),
        }
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

        let (sigma, weights) = gauss2sigma(prior, self.params)?;
        let mut transformed = [VecN([0.0; 4]); SIGMA_COUNT];
        for j in 0..SIGMA_COUNT {
            transformed[j] = crate::linalg::fixed::mat4_vec4(ws.f, sigma[j])?;
        }
        let (mean, cov) = sigma2gauss4(&transformed, &weights, Some(ws.q))?;
        GaussianState::new(mean, cov, timestamp)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UnscentedKalmanUpdater {
    pub measurement: CartesianToBearingRange,
    pub params: UkfParams,
}

impl UnscentedKalmanUpdater {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn new(measurement: CartesianToBearingRange) -> Self {
        Self {
            measurement,
            params: UkfParams::default(),
        }
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    fn predict_measurement(
        &self,
        predicted: &GaussianState,
    ) -> Result<(VecN<2>, Mat<2, 2>, Mat<4, 2>), Error> {
        let (sigma, weights) = gauss2sigma(predicted, self.params)?;
        let mut z_points = [VecN([0.0; 2]); SIGMA_COUNT];
        for j in 0..SIGMA_COUNT {
            z_points[j] = VecN(self.measurement.h(sigma[j].0)?);
        }
        let (z_mean, z_cov) = meas_sigma2gauss(&z_points, &weights, self.measurement.r())?;
        let cross = cross_cov_state_meas(&sigma, &z_points, &weights, z_mean)?;
        Ok((z_mean, z_cov, cross))
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn update(
        &self,
        predicted: &GaussianState,
        detection: &Detection,
        _ws: &mut KalmanWorkspace,
    ) -> Result<GaussianState, Error> {
        if detection.timestamp != predicted.timestamp {
            return Err(Error::InvalidParameter);
        }
        let (z_pred, s, cross) = self.predict_measurement(predicted)?;
        let innovation = CartesianToBearingRange::innovation(detection.vector.0, z_pred.0)?;
        let innov = VecN([innovation[0], innovation[1]]);
        let s_inv = inv2(s)?;
        let k = mul4x2_2x2(cross, s_inv)?;
        let k_innov = mul4x2_2x1(k, innov)?;
        let mean = add_vec4(predicted.mean, k_innov)?;
        let ks = mul4x2_2x2(k, s)?;
        let kt = transpose4x2(k)?;
        let kskt = mul4x2_2x4(ks, kt)?;
        let cov = sub4x4(predicted.cov, kskt)?;
        GaussianState::new(mean, cov, predicted.timestamp)
    }
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn meas_sigma2gauss(
    points: &[VecN<2>; SIGMA_COUNT],
    w: &SigmaWeights,
    r: Mat<2, 2>,
) -> Result<(VecN<2>, Mat<2, 2>), Error> {
    let mut sum_cos = 0.0;
    let mut sum_sin = 0.0;
    let mut mean = [0.0; 2];
    for j in 0..SIGMA_COUNT {
        sum_cos += w.mean[j] * libm::cos(points[j].0[0]);
        sum_sin += w.mean[j] * libm::sin(points[j].0[0]);
        mean[1] += w.mean[j] * points[j].0[1];
    }
    mean[0] = mod_bearing(libm::atan2(sum_sin, sum_cos))?;
    let mut cov = Mat::<2, 2>::zeros();
    for j in 0..SIGMA_COUNT {
        for a in 0..2 {
            for b in 0..2 {
                let da = if a == 0 {
                    mod_bearing(points[j].0[a] - mean[a])?
                } else {
                    points[j].0[a] - mean[a]
                };
                let db = if b == 0 {
                    mod_bearing(points[j].0[b] - mean[b])?
                } else {
                    points[j].0[b] - mean[b]
                };
                cov.set(a, b, cov.get(a, b)? + w.covar[j] * da * db)?;
            }
        }
    }
    for a in 0..2 {
        for b in 0..2 {
            cov.set(a, b, cov.get(a, b)? + r.get(a, b)?)?;
        }
    }
    Ok((VecN(mean), cov))
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn cross_cov_state_meas(
    state: &[VecN<4>; SIGMA_COUNT],
    meas: &[VecN<2>; SIGMA_COUNT],
    w: &SigmaWeights,
    z_mean: VecN<2>,
) -> Result<Mat<4, 2>, Error> {
    let mut cross = Mat::<4, 2>::zeros();
    for j in 0..SIGMA_COUNT {
        for a in 0..4 {
            let dx = state[j].0[a] - state[0].0[a];
            for b in 0..2 {
                let dz = if b == 0 {
                    mod_bearing(meas[j].0[b] - z_mean.0[b])?
                } else {
                    meas[j].0[b] - z_mean.0[b]
                };
                cross.set(a, b, cross.get(a, b)? + w.covar[j] * dx * dz)?;
            }
        }
    }
    Ok(cross)
}
