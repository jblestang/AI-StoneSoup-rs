//! GNN assignment with kd-tree detection filtering (``DetectionKDTreeGNN2D``).

#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::too_many_arguments
)]

use alloc::vec::Vec;

use crate::error::Error;
use crate::filter::kalman::{KalmanPredictor, KalmanUpdater, KalmanWorkspace};
use crate::state::GaussianState;
use crate::time::Timestamp;
use crate::types::Detection;

use super::distance::{distance_hypotheses, MAX_DISTANCE_HYPOTHESES};
use super::gnn_assignment::{
    assign_from_per_track_hypotheses, associate_gnn_2d, GnnAssignment, MAX_GNN_ASSIGN_DETS,
};
use super::kdtree2::KdTree2D;
use super::nearest_neighbour::MeasurementHypothesis;

/// Stone Soup ``DetectionKDTreeGNN2D`` defaults from ``KDTree_Tutorial.py``.
pub const KDTREE_GNN_NEIGHBOURS: usize = 3;
pub const KDTREE_GNN_COV_MULTIPLIER: f64 = 3.0;

fn missed_hypothesis(prediction: GaussianState, gate: f64) -> MeasurementHypothesis {
    MeasurementHypothesis {
        prediction,
        detection_index: None,
        distance: gate,
    }
}

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
    if n_dets > MAX_GNN_ASSIGN_DETS {
        return Err(Error::InvalidDimension);
    }
    if out.len() < n_tracks {
        return Err(Error::CapacityExceeded);
    }

    let template = track_tails[0];
    let tree = KdTree2D::build(detections)?;
    let mut filtered = [Detection {
        vector: crate::linalg::VecN([0.0, 0.0]),
        timestamp: association_time,
        sensor: crate::types::SensorId(0),
    }; MAX_GNN_ASSIGN_DETS];
    let mut index_map = [0usize; MAX_GNN_ASSIGN_DETS];
    let mut scratch = [missed_hypothesis(template, gate); MAX_DISTANCE_HYPOTHESES];
    let mut per_track = Vec::with_capacity(n_tracks);

    for tail in track_tails {
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
        let n = distance_hypotheses(
            tail,
            association_time,
            &filtered[..n_filt],
            gate,
            predictor,
            updater,
            ws,
            &mut scratch,
        )?;
        let mut hyps = scratch[..n].to_vec();
        for h in hyps.iter_mut().skip(1) {
            if let Some(local) = h.detection_index {
                h.detection_index = Some(index_map[local]);
            }
        }
        per_track.push(hyps);
    }

    assign_from_per_track_hypotheses(n_tracks, n_dets, gate, template, &per_track, out)
}
