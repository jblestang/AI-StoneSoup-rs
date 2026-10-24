//! GNN assignment with kd-tree detection filtering (``DetectionKDTreeGNN2D``).

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
use super::gnn_assignment::{
    associate_gnn_2d, GnnAssignment, MAX_GNN_ASSIGN_COLS, MAX_GNN_ASSIGN_DETS,
    MAX_GNN_ASSIGN_TRACKS,
};
use super::kdtree2::KdTree2D;
use super::nearest_neighbour::MeasurementHypothesis;

/// Stone Soup ``DetectionKDTreeGNN2D`` defaults from ``KDTree_Tutorial.py``.
pub const KDTREE_GNN_NEIGHBOURS: usize = 3;
pub const KDTREE_GNN_COV_MULTIPLIER: f64 = 3.0;

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn associate_gnn_2d_kdtree(
    track_tails: &[GaussianState],
    association_time: Timestamp,
    detections: &[Detection],
    gate: f64,
    predictor: &KalmanPredictor,
    updater: &KalmanUpdater,
    ws: &mut KalmanWorkspace,
    out: &mut [GnnAssignment],
    number_of_neighbours: usize,
    max_distance_covariance_multiplier: f64,
) -> Result<usize, Error> {
    let n_tracks = track_tails.len();
    let n_dets = detections.len();
    if n_dets == 0 || n_tracks == 0 {
        return associate_gnn_2d(
            track_tails,
            association_time,
            detections,
            gate,
            predictor,
            updater,
            ws,
            out,
        );
    }
    if n_dets > MAX_GNN_ASSIGN_DETS || n_tracks > MAX_GNN_ASSIGN_TRACKS {
        return Err(Error::InvalidDimension);
    }

    let tree = KdTree2D::build(detections)?;
    let mut filtered = [Detection {
        vector: crate::linalg::VecN([0.0, 0.0]),
        timestamp: association_time,
        sensor: crate::types::SensorId(0),
    }; MAX_GNN_ASSIGN_DETS];
    let mut index_map = [0usize; MAX_GNN_ASSIGN_DETS];
    let mut per_track = [[MeasurementHypothesis {
        prediction: track_tails[0],
        detection_index: None,
        distance: gate,
    }; MAX_DISTANCE_HYPOTHESES]; MAX_GNN_ASSIGN_TRACKS];
    let mut counts = [0usize; MAX_GNN_ASSIGN_TRACKS];

    for (t, tail) in track_tails.iter().enumerate() {
        let prediction = predictor.predict(tail, association_time, ws)?;
        let meas_pred = updater.predict_measurement(&prediction, ws)?;
        let c00 = meas_pred.cov.get(0, 0)?;
        let c11 = meas_pred.cov.get(1, 1)?;
        // Stone Soup ``DetectionKDTreeMixIn``: default ``max_distance`` is ``inf``, so only
        // ``max_distance_covariance_multiplier`` limits the kd query (Mahalanobis gating is separate).
        let mut max_dist = f64::INFINITY;
        let cov_cap = libm::fmax(c00, c11) * max_distance_covariance_multiplier;
        if cov_cap.is_finite() {
            max_dist = cov_cap;
        }
        let mx = meas_pred.mean.get(0)?;
        let my = meas_pred.mean.get(1)?;
        let mut neigh = [0usize; MAX_GNN_ASSIGN_DETS];
        let n_neigh = tree.query_knn([mx, my], number_of_neighbours, max_dist, &mut neigh)?;
        let mut n_filt = 0usize;
        for i in 0..n_neigh {
            let orig = neigh[i];
            if orig >= n_dets {
                continue;
            }
            filtered[n_filt] = detections[orig];
            index_map[n_filt] = orig;
            n_filt += 1;
        }
        counts[t] = distance_hypotheses(
            tail,
            association_time,
            &filtered[..n_filt],
            gate,
            predictor,
            updater,
            ws,
            &mut per_track[t],
        )?;
        for h in &mut per_track[t][1..counts[t]] {
            if let Some(local) = h.detection_index {
                h.detection_index = Some(index_map[local]);
            }
        }
    }

    // Reuse assignment logic from associate_gnn_2d with precomputed per_track.
    let mut has_gated = [false; MAX_GNN_ASSIGN_TRACKS];
    for t in 0..n_tracks {
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
    if cols > MAX_GNN_ASSIGN_COLS {
        return Err(Error::CapacityExceeded);
    }
    const INF: f64 = 1e100;
    let mut cost = [INF; MAX_GNN_ASSIGN_TRACKS * MAX_GNN_ASSIGN_COLS];
    let mut hyp = [[MeasurementHypothesis {
        prediction: track_tails[0],
        detection_index: None,
        distance: gate,
    }; MAX_GNN_ASSIGN_COLS]; MAX_GNN_ASSIGN_TRACKS];

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
    crate::assign::linear_sum_assignment_min(&cost, n_det_tracks, cols, &mut col_for_row)?;
    for (row, &t) in detected.iter().enumerate().take(n_det_tracks) {
        let col = col_for_row[row];
        out[t] = GnnAssignment {
            hypothesis: hyp[row][col],
        };
    }
    Ok(n_tracks)
}
