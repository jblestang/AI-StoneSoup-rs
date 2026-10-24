use crate::error::Error;
use crate::filter::kalman::{
    GaussianMeasurementPrediction, KalmanPredictor, KalmanUpdater, KalmanWorkspace,
};
use crate::linalg::fixed::{inv2, sub_vec2};
use crate::linalg::{require_finite, VecN};
use crate::state::GaussianState;
use crate::time::Timestamp;
use crate::types::Detection;

/// One candidate prediction–detection pair (Stone Soup `SingleDistanceHypothesis`).
#[derive(Debug, Clone, Copy)]
pub struct MeasurementHypothesis {
    pub prediction: GaussianState,
    pub detection_index: Option<usize>,
    pub distance: f64,
}

/// Result of nearest-neighbour association for a single track.
#[derive(Debug, Clone, Copy)]
pub enum AssociationOutcome {
    /// A gated detection was selected for update.
    Matched {
        detection_index: usize,
        prediction: GaussianState,
    },
    /// No detection within the gate; append the missed-detection prediction.
    Missed { prediction: GaussianState },
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn mahalanobis(
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
    Ok(libm::sqrt(sq))
}

/// Nearest-neighbour associator for one track (Tutorial 05 / Stone Soup `NearestNeighbour`).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn associate_nearest_neighbour(
    track_tail: &GaussianState,
    association_time: Timestamp,
    detections: &[Detection],
    gate: f64,
    predictor: &KalmanPredictor,
    updater: &KalmanUpdater,
    ws: &mut KalmanWorkspace,
) -> Result<AssociationOutcome, Error> {
    let missed_prediction = predictor.predict(track_tail, association_time, ws)?;
    let mut best = MeasurementHypothesis {
        prediction: missed_prediction,
        detection_index: None,
        distance: gate,
    };

    for (i, det) in detections.iter().enumerate() {
        let prediction = predictor.predict(track_tail, det.timestamp, ws)?;
        let meas_pred = updater.predict_measurement(&prediction, ws)?;
        let distance = mahalanobis(det.vector, &meas_pred)?;
        if distance < gate && distance < best.distance {
            best = MeasurementHypothesis {
                prediction,
                detection_index: Some(i),
                distance,
            };
        }
    }

    match best.detection_index {
        Some(detection_index) => Ok(AssociationOutcome::Matched {
            detection_index,
            prediction: best.prediction,
        }),
        None => Ok(AssociationOutcome::Missed {
            prediction: best.prediction,
        }),
    }
}
