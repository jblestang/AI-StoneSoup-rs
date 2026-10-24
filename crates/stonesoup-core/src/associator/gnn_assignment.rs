//! GNN with 2D assignment (Stone Soup ``GNNWith2DAssignment``).

#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::too_many_arguments
)]

use crate::assign::linear_sum_assignment_min;
use crate::error::Error;
use crate::filter::kalman::{KalmanPredictor, KalmanUpdater, KalmanWorkspace};
use crate::state::GaussianState;
use crate::time::Timestamp;
use crate::types::Detection;

use super::distance::{distance_hypotheses, MAX_DISTANCE_HYPOTHESES};
use super::nearest_neighbour::MeasurementHypothesis;

pub const MAX_GNN_ASSIGN_TRACKS: usize = 16;
pub const MAX_GNN_ASSIGN_DETS: usize = 32;
const INF: f64 = 1e100;

#[derive(Debug, Clone, Copy)]
pub struct GnnAssignment {
    pub hypothesis: MeasurementHypothesis,
}

/// Associate tracks with detections via global nearest neighbour (Hungarian assignment).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn associate_gnn_2d(
    track_tails: &[GaussianState],
    association_time: Timestamp,
    detections: &[Detection],
    gate: f64,
    predictor: &KalmanPredictor,
    updater: &KalmanUpdater,
    ws: &mut KalmanWorkspace,
    out: &mut [GnnAssignment],
) -> Result<usize, Error> {
    let n_tracks = track_tails.len();
    let n_dets = detections.len();
    if n_tracks == 0 || n_tracks > MAX_GNN_ASSIGN_TRACKS || n_dets > MAX_GNN_ASSIGN_DETS {
        return Err(Error::InvalidDimension);
    }
    if out.len() < n_tracks {
        return Err(Error::CapacityExceeded);
    }

    let mut per_track = [[MeasurementHypothesis {
        prediction: track_tails[0],
        detection_index: None,
        distance: gate,
    }; MAX_DISTANCE_HYPOTHESES]; MAX_GNN_ASSIGN_TRACKS];
    let mut counts = [0usize; MAX_GNN_ASSIGN_TRACKS];
    let mut has_gated = [false; MAX_GNN_ASSIGN_TRACKS];

    for (t, tail) in track_tails.iter().enumerate() {
        counts[t] = distance_hypotheses(
            tail,
            association_time,
            detections,
            gate,
            predictor,
            updater,
            ws,
            &mut per_track[t],
        )?;
        for h in &per_track[t][1..counts[t]] {
            if h.detection_index.is_some() {
                has_gated[t] = true;
                break;
            }
        }
    }

    for t in 0..n_tracks {
        if !has_gated[t] {
            out[t] = GnnAssignment {
                hypothesis: per_track[t][0],
            };
        }
    }

    let mut detected = [0usize; MAX_GNN_ASSIGN_TRACKS];
    let mut n_det_tracks = 0usize;
    for t in 0..n_tracks {
        if has_gated[t] {
            detected[n_det_tracks] = t;
            n_det_tracks += 1;
        }
    }
    if n_det_tracks == 0 {
        return Ok(n_tracks);
    }

    let cols = n_dets + n_det_tracks;
    if cols > 48 {
        return Err(Error::CapacityExceeded);
    }

    let mut cost = [INF; MAX_GNN_ASSIGN_TRACKS * 48];
    let mut hyp = [[MeasurementHypothesis {
        prediction: track_tails[0],
        detection_index: None,
        distance: gate,
    }; 48]; MAX_GNN_ASSIGN_TRACKS];

    for (row, &t) in detected.iter().enumerate().take(n_det_tracks) {
        for h in &per_track[t][..counts[t]] {
            let col = if let Some(di) = h.detection_index {
                di
            } else {
                n_dets + row
            };
            if col < cols {
                hyp[row][col] = *h;
                cost[row * cols + col] = h.distance;
            }
        }
    }

    let mut col_for_row = [0usize; MAX_GNN_ASSIGN_TRACKS];
    linear_sum_assignment_min(&cost, n_det_tracks, cols, &mut col_for_row)?;

    for (row, &t) in detected.iter().enumerate().take(n_det_tracks) {
        let col = col_for_row[row];
        out[t] = GnnAssignment {
            hypothesis: hyp[row][col],
        };
    }

    Ok(n_tracks)
}
