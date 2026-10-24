#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::too_many_arguments
)]

use crate::error::Error;
use crate::filter::kalman::{
    GaussianMeasurementPrediction, KalmanPredictor, KalmanUpdater, KalmanWorkspace,
};
use crate::linalg::fixed::{inv2, sub_vec2};
use crate::linalg::{require_finite, VecN};
use crate::state::GaussianState;
use crate::time::Timestamp;
use crate::types::Detection;

use super::nearest_neighbour::MeasurementHypothesis;

/// Missed hypothesis plus up to [`super::gnn_assignment::MAX_GNN_ASSIGN_DETS`] gated detections.
pub const MAX_DISTANCE_HYPOTHESES: usize = super::gnn_assignment::MAX_GNN_ASSIGN_DETS + 1;

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn mahalanobis(
    z: VecN<2>,
    measurement_prediction: &GaussianMeasurementPrediction,
) -> Result<f64, Error> {
    let sq = squared_mahalanobis(z, measurement_prediction)?;
    Ok(libm::sqrt(sq))
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn squared_mahalanobis(
    z: VecN<2>,
    measurement_prediction: &GaussianMeasurementPrediction,
) -> Result<f64, Error> {
    let inv = inv2(measurement_prediction.cov)?;
    let dx = sub_vec2(z, measurement_prediction.mean)?;
    let d0 = dx.0[0];
    let d1 = dx.0[1];
    let s0 = inv.0[0][0] * d0 + inv.0[0][1] * d1;
    let s1 = inv.0[1][0] * d0 + inv.0[1][1] * d1;
    let sq = d0 * s0 + d1 * s1;
    require_finite(sq)?;
    if sq < 0.0 {
        return Err(Error::NumericalFailure);
    }
    Ok(sq)
}

/// Stone Soup `DistanceHypothesiser` hypotheses (missed at `gate` plus gated detections).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn distance_hypotheses(
    track_tail: &GaussianState,
    association_time: Timestamp,
    detections: &[Detection],
    gate: f64,
    predictor: &KalmanPredictor,
    updater: &KalmanUpdater,
    ws: &mut KalmanWorkspace,
    out: &mut [MeasurementHypothesis],
) -> Result<usize, Error> {
    if out.is_empty() {
        return Err(Error::CapacityExceeded);
    }
    let missed_prediction = predictor.predict(track_tail, association_time, ws)?;
    out[0] = MeasurementHypothesis {
        prediction: missed_prediction,
        detection_index: None,
        distance: gate,
    };
    let mut n = 1usize;
    for (i, det) in detections.iter().enumerate() {
        if n >= out.len() {
            return Err(Error::CapacityExceeded);
        }
        let prediction = predictor.predict(track_tail, det.timestamp, ws)?;
        let meas_pred = updater.predict_measurement(&prediction, ws)?;
        let distance = mahalanobis(det.vector, &meas_pred)?;
        if distance < gate {
            out[n] = MeasurementHypothesis {
                prediction,
                detection_index: Some(i),
                distance,
            };
            n += 1;
        }
    }
    Ok(n)
}
