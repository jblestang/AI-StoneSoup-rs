#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::too_many_arguments
)]

use crate::error::Error;
use crate::filter::kalman::{KalmanPredictor, KalmanUpdater, KalmanWorkspace};
use crate::state::GaussianState;
use crate::time::Timestamp;
use crate::types::Detection;

use super::distance::{distance_hypotheses, MAX_DISTANCE_HYPOTHESES};

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
    let mut hyps = [MeasurementHypothesis {
        prediction: *track_tail,
        detection_index: None,
        distance: gate,
    }; MAX_DISTANCE_HYPOTHESES];
    let n = distance_hypotheses(
        track_tail,
        association_time,
        detections,
        gate,
        predictor,
        updater,
        ws,
        &mut hyps,
    )?;
    let mut best = hyps[0];
    for h in &hyps[1..n] {
        if h.distance < best.distance {
            best = *h;
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
