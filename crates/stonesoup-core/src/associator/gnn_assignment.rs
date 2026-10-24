//! GNN with 2D assignment (Stone Soup ``GNNWith2DAssignment``).

#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::too_many_arguments
)]

use alloc::vec;
use alloc::vec::Vec;

use crate::assign::linear_sum_assignment_min;
use crate::error::Error;
use crate::filter::kalman::{KalmanPredictor, KalmanUpdater, KalmanWorkspace};
use crate::state::GaussianState;
use crate::time::Timestamp;
use crate::types::Detection;

use super::distance::{distance_hypotheses, MAX_DISTANCE_HYPOTHESES};
use super::nearest_neighbour::MeasurementHypothesis;

/// Upper bound on detections per GNN call (scenario replay buffers are sized to this).
pub const MAX_GNN_ASSIGN_DETS: usize = 48;
const INF: f64 = 1e100;

#[derive(Debug, Clone, Copy)]
pub struct GnnAssignment {
    pub hypothesis: MeasurementHypothesis,
}

fn missed_hypothesis(prediction: GaussianState, gate: f64) -> MeasurementHypothesis {
    MeasurementHypothesis {
        prediction,
        detection_index: None,
        distance: gate,
    }
}

/// Hungarian assignment from per-track distance hypotheses (shared by linear and kd-tree GNN).
pub(crate) fn assign_from_per_track_hypotheses(
    n_tracks: usize,
    n_dets: usize,
    gate: f64,
    template: GaussianState,
    per_track: &[Vec<MeasurementHypothesis>],
    out: &mut [GnnAssignment],
) -> Result<usize, Error> {
    if out.len() < n_tracks || per_track.len() < n_tracks {
        return Err(Error::InvalidDimension);
    }

    let mut has_gated = vec![false; n_tracks];
    for t in 0..n_tracks {
        for h in per_track[t].iter().skip(1) {
            if h.detection_index.is_some() {
                has_gated[t] = true;
                break;
            }
        }
    }

    for t in 0..n_tracks {
        if !has_gated[t] {
            let missed = per_track[t]
                .first()
                .copied()
                .unwrap_or_else(|| missed_hypothesis(template, gate));
            out[t] = GnnAssignment { hypothesis: missed };
        }
    }

    let mut detected = Vec::with_capacity(n_tracks);
    for t in 0..n_tracks {
        if has_gated[t] {
            detected.push(t);
        }
    }
    let n_det_tracks = detected.len();
    if n_det_tracks == 0 {
        return Ok(n_tracks);
    }

    let cols = n_dets + n_det_tracks;
    let mut cost = vec![INF; n_det_tracks * cols];
    let mut hyp = vec![vec![missed_hypothesis(template, gate); cols]; n_det_tracks];

    for (row, &t) in detected.iter().enumerate() {
        for h in &per_track[t] {
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

    let mut col_for_row = vec![0usize; n_det_tracks];
    linear_sum_assignment_min(&cost, n_det_tracks, cols, &mut col_for_row)?;

    for (row, &t) in detected.iter().enumerate() {
        let col = col_for_row[row];
        out[t] = GnnAssignment {
            hypothesis: hyp[row][col],
        };
    }

    Ok(n_tracks)
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
    if n_tracks == 0 || n_dets > MAX_GNN_ASSIGN_DETS {
        return Err(Error::InvalidDimension);
    }
    if out.len() < n_tracks {
        return Err(Error::CapacityExceeded);
    }

    let template = track_tails[0];
    let mut scratch = [missed_hypothesis(template, gate); MAX_DISTANCE_HYPOTHESES];
    let mut per_track = Vec::with_capacity(n_tracks);

    for tail in track_tails {
        let n = distance_hypotheses(
            tail,
            association_time,
            detections,
            gate,
            predictor,
            updater,
            ws,
            &mut scratch,
        )?;
        per_track.push(scratch[..n].to_vec());
    }

    assign_from_per_track_hypotheses(n_tracks, n_dets, gate, template, &per_track, out)
}
