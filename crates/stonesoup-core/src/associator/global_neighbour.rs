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
use super::nearest_neighbour::{AssociationOutcome, MeasurementHypothesis};

pub const MAX_GNN_TRACKS: usize = 2;

/// Per-track GNN assignment (Tutorial 06).
#[derive(Debug, Clone, Copy)]
pub struct GnnTrackAssignment {
    pub outcome: AssociationOutcome,
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn joint_valid(picks: &[MeasurementHypothesis; MAX_GNN_TRACKS], tracks: usize) -> bool {
    for i in 0..tracks {
        let Some(idx_i) = picks[i].detection_index else {
            continue;
        };
        for j in (i + 1)..tracks {
            if let Some(idx_j) = picks[j].detection_index {
                if idx_i == idx_j {
                    return false;
                }
            }
        }
    }
    true
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn joint_distance(picks: &[MeasurementHypothesis; MAX_GNN_TRACKS], tracks: usize) -> f64 {
    let mut s = 0.0;
    for h in &picks[..tracks] {
        s += h.distance;
    }
    s
}

/// Global nearest neighbour for up to [`MAX_GNN_TRACKS`] tracks.
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn associate_global_nearest_neighbour(
    track_tails: &[GaussianState],
    association_time: Timestamp,
    detections: &[Detection],
    gate: f64,
    predictor: &KalmanPredictor,
    updater: &KalmanUpdater,
    ws: &mut KalmanWorkspace,
    out: &mut [GnnTrackAssignment; MAX_GNN_TRACKS],
) -> Result<usize, Error> {
    let tracks = track_tails.len();
    if tracks == 0 || tracks > MAX_GNN_TRACKS {
        return Err(Error::InvalidDimension);
    }
    let mut per_track = [[MeasurementHypothesis {
        prediction: track_tails[0],
        detection_index: None,
        distance: gate,
    }; MAX_DISTANCE_HYPOTHESES]; MAX_GNN_TRACKS];
    let mut counts = [0usize; MAX_GNN_TRACKS];
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
    }

    let mut best_picks = [per_track[0][0]; MAX_GNN_TRACKS];
    let mut best_dist = f64::INFINITY;

    if tracks == 1 {
        for i0 in 0..counts[0] {
            let picks = [per_track[0][i0]; MAX_GNN_TRACKS];
            let d = joint_distance(&picks, 1);
            if d < best_dist {
                best_dist = d;
                best_picks = picks;
            }
        }
    } else {
        for i0 in 0..counts[0] {
            for i1 in 0..counts[1] {
                let picks = [per_track[0][i0], per_track[1][i1]];
                if !joint_valid(&picks, 2) {
                    continue;
                }
                let d = joint_distance(&picks, 2);
                if d < best_dist {
                    best_dist = d;
                    best_picks = picks;
                }
            }
        }
    }

    for t in 0..tracks {
        let h = best_picks[t];
        out[t] = GnnTrackAssignment {
            outcome: match h.detection_index {
                Some(detection_index) => AssociationOutcome::Matched {
                    detection_index,
                    prediction: h.prediction,
                },
                None => AssociationOutcome::Missed {
                    prediction: h.prediction,
                },
            },
        };
    }
    Ok(tracks)
}
