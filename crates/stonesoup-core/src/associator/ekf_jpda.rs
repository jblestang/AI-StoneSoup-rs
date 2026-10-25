//! EKF bearing–range PDA / JPDA (conflict-group joint enumeration).

#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::too_many_arguments
)]

use alloc::vec::Vec;

use crate::error::Error;
use crate::filter::ekf::ExtendedKalmanUpdater;
use crate::filter::kalman::{GaussianMeasurementPrediction, KalmanPredictor, KalmanWorkspace};
use crate::linalg::VecN;
use crate::models::bearing_range::CartesianToBearingRange;
use crate::state::GaussianState;
use crate::time::Timestamp;
use crate::types::Detection;

use super::ekf_gnn;
use super::jpda::JpdaHypothesisSet;
use super::pda::{
    gm_reduce_single, logpdf2, normalize_probs, ProbabilityHypothesis, MAX_PDA_HYPOTHESES,
    PDA_GATE_CHI2_2D,
};

pub const MAX_JPDA_EKF_TRACKS: usize = 4096;
const MAX_JPDA_COMPONENT: usize = 8;
const MAX_JPDA_JOINTS: usize = 512;

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn joint_valid(picks: &[ProbabilityHypothesis], tracks: usize) -> bool {
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

/// PDA hypotheses for one track with per-detection EKF models.
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn ekf_pda_hypotheses(
    track_tail: &GaussianState,
    association_time: Timestamp,
    detections: &[Detection],
    prob_detect: f64,
    clutter_spatial_density: f64,
    predictor: &KalmanPredictor,
    _updater: &ExtendedKalmanUpdater,
    det_models: &[CartesianToBearingRange],
    ws: &mut KalmanWorkspace,
    out: &mut [ProbabilityHypothesis],
) -> Result<usize, Error> {
    if out.is_empty() {
        return Err(Error::CapacityExceeded);
    }
    let prob_gate = 0.95;
    let missed = predictor.predict(track_tail, association_time, ws)?;
    out[0] = ProbabilityHypothesis {
        prediction: missed,
        detection_index: None,
        probability: 1.0 - prob_detect * prob_gate,
    };
    let mut n = 1usize;
    for (i, det) in detections.iter().enumerate() {
        if n >= out.len() {
            break;
        }
        let model = det_models.get(i).ok_or(Error::InvalidDimension)?;
        let prediction = predictor.predict(track_tail, det.timestamp, ws)?;
        let meas_pred = match ekf_gnn::ekf_meas_pred(model, &prediction, ws) {
            Ok(m) => m,
            Err(Error::NumericalFailure) => continue,
            Err(e) => return Err(e),
        };
        let innovation = match CartesianToBearingRange::innovation(det.vector.0, meas_pred.mean.0)
        {
            Ok(v) => v,
            Err(_) => continue,
        };
        let sq = match super::distance::squared_mahalanobis(
            VecN(innovation),
            &GaussianMeasurementPrediction {
                mean: VecN([0.0, 0.0]),
                cov: meas_pred.cov,
            },
        ) {
            Ok(s) => s,
            Err(Error::SingularInnovation) => continue,
            Err(e) => return Err(e),
        };
        if sq <= PDA_GATE_CHI2_2D {
            let logp = match logpdf2(det.vector, meas_pred.mean, meas_pred.cov) {
                Ok(v) => v,
                Err(Error::SingularInnovation) => continue,
                Err(e) => return Err(e),
            };
            let mut prob = libm::exp(logp);
            prob *= prob_detect / clutter_spatial_density;
            out[n] = ProbabilityHypothesis {
                prediction,
                detection_index: Some(i),
                probability: prob,
            };
            n += 1;
        }
    }
    normalize_probs(&mut out[..], n)?;
    Ok(n)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn ekf_pda_merged_update(
    hyps: &[ProbabilityHypothesis],
    n: usize,
    detections: &[Detection],
    det_models: &[CartesianToBearingRange],
    updater: &mut ExtendedKalmanUpdater,
    ws: &mut KalmanWorkspace,
) -> Result<GaussianState, Error> {
    if n == 0 {
        return Err(Error::InvalidDimension);
    }
    let mut means = [VecN([0.0; 4]); MAX_PDA_HYPOTHESES];
    let mut covs = [crate::linalg::Mat::<4, 4>::zeros(); MAX_PDA_HYPOTHESES];
    let mut weights = [0.0f64; MAX_PDA_HYPOTHESES];
    let ts = hyps
        .iter()
        .take(n)
        .find_map(|h| h.detection_index)
        .and_then(|i| detections.get(i).map(|d| d.timestamp))
        .unwrap_or(hyps[0].prediction.timestamp);
    for (k, h) in hyps.iter().take(n).enumerate() {
        weights[k] = h.probability;
        let state = match h.detection_index {
            None => h.prediction,
            Some(i) => {
                let det = detections.get(i).ok_or(Error::InvalidDimension)?;
                updater.measurement = det_models.get(i).copied().ok_or(Error::InvalidDimension)?;
                match updater.update(&h.prediction, det, ws) {
                    Ok(s) => s,
                    Err(Error::SingularInnovation)
                    | Err(Error::NumericalFailure)
                    | Err(Error::InvalidParameter) => h.prediction,
                    Err(e) => return Err(e),
                }
            }
        };
        means[k] = state.mean;
        covs[k] = state.cov;
    }
    let (m, c) = gm_reduce_single(&means[..n], &covs[..n], &weights[..n])?;
    GaussianState::new(m, c, ts)
}

fn uf_find(parent: &mut [usize], x: usize) -> usize {
    if parent[x] != x {
        parent[x] = uf_find(parent, parent[x]);
    }
    parent[x]
}

fn uf_unite(parent: &mut [usize], a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        parent[rb] = ra;
    }
}

fn enumerate_component(
    per_track: &[[ProbabilityHypothesis; MAX_PDA_HYPOTHESES]],
    counts: &[usize],
    comp: &[usize],
    out: &mut [JpdaHypothesisSet],
) -> Result<(), Error> {
    let k = comp.len();
    if k == 0 {
        return Ok(());
    }
    if k == 1 {
        let t = comp[0];
        out[t] = JpdaHypothesisSet {
            hypotheses: per_track[t],
            count: counts[t],
        };
        return Ok(());
    }
    if k > MAX_JPDA_COMPONENT {
        for &t in comp {
            out[t] = JpdaHypothesisSet {
                hypotheses: per_track[t],
                count: counts[t],
            };
        }
        return Ok(());
    }

    let mut joint_probs = [0.0f64; MAX_JPDA_JOINTS];
    let mut joint_picks = [[ProbabilityHypothesis {
        prediction: per_track[comp[0]][0].prediction,
        detection_index: None,
        probability: 0.0,
    }; MAX_JPDA_COMPONENT]; MAX_JPDA_JOINTS];
    let mut joint_count = 0usize;

    fn rec(
        level: usize,
        k: usize,
        comp: &[usize],
        per_track: &[[ProbabilityHypothesis; MAX_PDA_HYPOTHESES]],
        counts: &[usize],
        picks: &mut [ProbabilityHypothesis],
        joint_probs: &mut [f64],
        joint_picks: &mut [[ProbabilityHypothesis; MAX_JPDA_COMPONENT]],
        joint_count: &mut usize,
    ) -> Result<(), Error> {
        if level == k {
            if !joint_valid(picks, k) {
                return Ok(());
            }
            if *joint_count >= MAX_JPDA_JOINTS {
                return Ok(());
            }
            let mut prod = 1.0;
            for i in 0..k {
                prod *= picks[i].probability;
            }
            joint_picks[*joint_count][..k].copy_from_slice(&picks[..k]);
            joint_probs[*joint_count] = prod;
            *joint_count += 1;
            return Ok(());
        }
        let t = comp[level];
        for i in 0..counts[t] {
            picks[level] = per_track[t][i];
            rec(
                level + 1,
                k,
                comp,
                per_track,
                counts,
                picks,
                joint_probs,
                joint_picks,
                joint_count,
            )?;
        }
        Ok(())
    }

    let mut picks = [ProbabilityHypothesis {
        prediction: per_track[comp[0]][0].prediction,
        detection_index: None,
        probability: 0.0,
    }; MAX_JPDA_COMPONENT];
    rec(
        0,
        k,
        comp,
        per_track,
        counts,
        &mut picks,
        &mut joint_probs,
        &mut joint_picks,
        &mut joint_count,
    )?;

    let mut sum = 0.0;
    for j in 0..joint_count {
        sum += joint_probs[j];
    }
    if sum <= 0.0 {
        for &t in comp {
            out[t] = JpdaHypothesisSet {
                hypotheses: per_track[t],
                count: counts[t],
            };
        }
        return Ok(());
    }
    for j in 0..joint_count {
        joint_probs[j] /= sum;
    }

    for (local_t, &global_t) in comp.iter().enumerate() {
        let mut out_hyps = [ProbabilityHypothesis {
            prediction: per_track[global_t][0].prediction,
            detection_index: None,
            probability: 0.0,
        }; MAX_PDA_HYPOTHESES];
        let mut out_n = 0usize;
        for h in 0..counts[global_t] {
            let template = per_track[global_t][h];
            let mut prob = 0.0;
            for j in 0..joint_count {
                if joint_picks[j][local_t].detection_index == template.detection_index {
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
        out[global_t] = JpdaHypothesisSet {
            hypotheses: out_hyps,
            count: out_n,
        };
    }
    Ok(())
}

/// JPDA on EKF tracks: PDA per track, joint normalization within conflict components (≤8 tracks).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn jpda_associate_ekf_det_models(
    track_tails: &[GaussianState],
    association_time: Timestamp,
    detections: &[Detection],
    prob_detect: f64,
    clutter_spatial_density: f64,
    predictor: &KalmanPredictor,
    updater: &ExtendedKalmanUpdater,
    det_models: &[CartesianToBearingRange],
    ws: &mut KalmanWorkspace,
    out: &mut [JpdaHypothesisSet],
) -> Result<usize, Error> {
    let tracks = track_tails.len();
    if tracks == 0 || tracks > MAX_JPDA_EKF_TRACKS || out.len() < tracks {
        return Err(Error::InvalidDimension);
    }
    let template = track_tails[0];
    let empty_hyp = ProbabilityHypothesis {
        prediction: template,
        detection_index: None,
        probability: 0.0,
    };
    let mut per_track = alloc::vec![[empty_hyp; MAX_PDA_HYPOTHESES]; tracks];
    let mut counts = alloc::vec![0usize; tracks];
    for (t, tail) in track_tails.iter().enumerate() {
        counts[t] = ekf_pda_hypotheses(
            tail,
            association_time,
            detections,
            prob_detect,
            clutter_spatial_density,
            predictor,
            updater,
            det_models,
            ws,
            &mut per_track[t],
        )?;
    }

    let mut parent: Vec<usize> = (0..tracks).collect();
    for t1 in 0..tracks {
        for t2 in (t1 + 1)..tracks {
            let mut share = false;
            for h1 in 0..counts[t1] {
                let Some(d1) = per_track[t1][h1].detection_index else {
                    continue;
                };
                for h2 in 0..counts[t2] {
                    if per_track[t2][h2].detection_index == Some(d1) {
                        share = true;
                        break;
                    }
                }
                if share {
                    break;
                }
            }
            if share {
                uf_unite(&mut parent, t1, t2);
            }
        }
    }

    let mut by_root: Vec<Vec<usize>> = alloc::vec![Vec::new(); tracks];
    for t in 0..tracks {
        let r = uf_find(&mut parent, t);
        by_root[r].push(t);
    }

    for comp in by_root.into_iter().filter(|c| !c.is_empty()) {
        enumerate_component(&per_track, &counts, &comp, out)?;
    }
    Ok(tracks)
}

/// Sum of marginal detection association probabilities per detection index.
pub fn jpda_detection_weights(
    sets: &[JpdaHypothesisSet],
    n_dets: usize,
) -> Vec<f64> {
    let mut w = alloc::vec![0.0; n_dets];
    for set in sets {
        for h in set.hypotheses.iter().take(set.count) {
            if let Some(di) = h.detection_index {
                if di < n_dets {
                    w[di] += h.probability;
                }
            }
        }
    }
    w
}
