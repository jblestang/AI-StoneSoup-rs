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

use super::global_neighbour::MAX_GNN_TRACKS;
use super::pda::{pda_hypotheses, ProbabilityHypothesis, MAX_PDA_HYPOTHESES};

pub const MAX_JPDA_JOINTS: usize = 512;

#[derive(Debug, Clone, Copy)]
pub struct JpdaHypothesisSet {
    pub hypotheses: [ProbabilityHypothesis; MAX_PDA_HYPOTHESES],
    pub count: usize,
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn joint_valid(picks: &[ProbabilityHypothesis; MAX_GNN_TRACKS], tracks: usize) -> bool {
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

/// JPDA association (Tutorial 08): PDA hypotheses with joint probability weights.
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn jpda_associate(
    track_tails: &[GaussianState],
    association_time: Timestamp,
    detections: &[Detection],
    prob_detect: f64,
    clutter_spatial_density: f64,
    predictor: &KalmanPredictor,
    updater: &KalmanUpdater,
    ws: &mut KalmanWorkspace,
    out: &mut [JpdaHypothesisSet; MAX_GNN_TRACKS],
) -> Result<usize, Error> {
    let tracks = track_tails.len();
    if tracks == 0 || tracks > MAX_GNN_TRACKS {
        return Err(Error::InvalidDimension);
    }
    let mut per_track = [[ProbabilityHypothesis {
        prediction: track_tails[0],
        detection_index: None,
        probability: 0.0,
    }; MAX_PDA_HYPOTHESES]; MAX_GNN_TRACKS];
    let mut counts = [0usize; MAX_GNN_TRACKS];
    for (t, tail) in track_tails.iter().enumerate() {
        counts[t] = pda_hypotheses(
            tail,
            association_time,
            detections,
            prob_detect,
            clutter_spatial_density,
            predictor,
            updater,
            ws,
            &mut per_track[t],
        )?;
    }

    let mut joint_probs = [0.0f64; MAX_JPDA_JOINTS];
    let mut joint_picks = [[ProbabilityHypothesis {
        prediction: track_tails[0],
        detection_index: None,
        probability: 0.0,
    }; MAX_GNN_TRACKS]; MAX_JPDA_JOINTS];
    let mut joint_count = 0usize;

    if tracks == 1 {
        for i in 0..counts[0] {
            if joint_count >= MAX_JPDA_JOINTS {
                return Err(Error::CapacityExceeded);
            }
            joint_picks[joint_count][0] = per_track[0][i];
            joint_probs[joint_count] = per_track[0][i].probability;
            joint_count += 1;
        }
    } else if tracks == 2 {
        for i0 in 0..counts[0] {
            for i1 in 0..counts[1] {
                let picks = [per_track[0][i0], per_track[1][i1]];
                if !joint_valid(&picks, 2) {
                    continue;
                }
                if joint_count >= MAX_JPDA_JOINTS {
                    return Err(Error::CapacityExceeded);
                }
                joint_picks[joint_count] = picks;
                joint_probs[joint_count] = picks[0].probability * picks[1].probability;
                joint_count += 1;
            }
        }
    } else {
        return Err(Error::InvalidDimension);
    }

    let mut sum = 0.0;
    for j in 0..joint_count {
        sum += joint_probs[j];
    }
    if sum <= 0.0 {
        return Err(Error::NumericalFailure);
    }
    for j in 0..joint_count {
        joint_probs[j] /= sum;
    }

    for t in 0..tracks {
        let mut out_hyps = [ProbabilityHypothesis {
            prediction: track_tails[t],
            detection_index: None,
            probability: 0.0,
        }; MAX_PDA_HYPOTHESES];
        let mut out_n = 0usize;
        for h in 0..counts[t] {
            let template = per_track[t][h];
            let mut prob = 0.0;
            for j in 0..joint_count {
                if joint_picks[j][t].detection_index == template.detection_index {
                    prob += joint_probs[j];
                }
            }
            out_hyps[out_n] = ProbabilityHypothesis {
                prediction: template.prediction,
                detection_index: template.detection_index,
                probability: prob,
            };
            out_n += 1;
        }
        out[t] = JpdaHypothesisSet {
            hypotheses: out_hyps,
            count: out_n,
        };
    }
    Ok(tracks)
}
