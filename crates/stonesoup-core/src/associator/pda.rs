#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::too_many_arguments
)]

use crate::error::Error;
use crate::filter::kalman::{KalmanPredictor, KalmanUpdater, KalmanWorkspace};
use crate::linalg::fixed::add4x4;
use crate::linalg::{Mat, VecN};
use crate::state::GaussianState;
use crate::time::Timestamp;
use crate::types::Detection;

use super::distance::squared_mahalanobis;

/// Chi-squared gate :math:`\chi^2_{0.95,2}` (Stone Soup `PDAHypothesiser` default `prob_gate`).
pub const PDA_GATE_CHI2_2D: f64 = 5.991_464_547_107_982;

pub const MAX_PDA_HYPOTHESES: usize = 32;

#[derive(Debug, Clone, Copy)]
pub struct ProbabilityHypothesis {
    pub prediction: GaussianState,
    pub detection_index: Option<usize>,
    pub probability: f64,
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn logpdf2(z: VecN<2>, mean: VecN<2>, cov: Mat<2, 2>) -> Result<f64, Error> {
    let mp = crate::filter::kalman::GaussianMeasurementPrediction { mean, cov };
    let sq = squared_mahalanobis(z, &mp)?;
    let det = cov.0[0][0] * cov.0[1][1] - cov.0[0][1] * cov.0[1][0];
    if det <= 0.0 {
        return Err(Error::NumericalFailure);
    }
    let log_det = libm::log(det);
    const LN_2PI: f64 = 1.8378770664093453;
    Ok(-0.5 * (2.0 * LN_2PI + log_det + sq))
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn normalize_probs(hyps: &mut [ProbabilityHypothesis], n: usize) -> Result<(), Error> {
    let mut sum = 0.0;
    for h in hyps.iter().take(n) {
        sum += h.probability;
    }
    if sum <= 0.0 {
        return Err(Error::NumericalFailure);
    }
    for h in hyps.iter_mut().take(n) {
        h.probability /= sum;
    }
    Ok(())
}

/// PDA hypothesiser (Tutorial 07).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn pda_hypotheses(
    track_tail: &GaussianState,
    association_time: Timestamp,
    detections: &[Detection],
    prob_detect: f64,
    clutter_spatial_density: f64,
    predictor: &KalmanPredictor,
    updater: &KalmanUpdater,
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
            return Err(Error::CapacityExceeded);
        }
        let prediction = predictor.predict(track_tail, det.timestamp, ws)?;
        let meas_pred = updater.predict_measurement(&prediction, ws)?;
        let sq = squared_mahalanobis(det.vector, &meas_pred)?;
        if sq <= PDA_GATE_CHI2_2D {
            let logp = logpdf2(det.vector, meas_pred.mean, meas_pred.cov)?;
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

/// Gaussian mixture reduction (Stone Soup `gm_reduce_single`).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn gm_reduce_single(
    means: &[VecN<4>],
    covs: &[Mat<4, 4>],
    weights: &[f64],
) -> Result<(VecN<4>, Mat<4, 4>), Error> {
    let n = means.len();
    if n == 0 || n != covs.len() || n != weights.len() {
        return Err(Error::InvalidDimension);
    }
    let mut wsum = 0.0;
    for &w in weights {
        wsum += w;
    }
    if wsum <= 0.0 {
        return Err(Error::NumericalFailure);
    }
    let mut mean = VecN([0.0; 4]);
    for (m, &w) in means.iter().zip(weights) {
        let ww = w / wsum;
        for d in 0..4 {
            mean.0[d] += m.0[d] * ww;
        }
    }
    let mut cov = Mat::<4, 4>::zeros();
    for (m, (c, &w)) in means.iter().zip(covs.iter().zip(weights)) {
        let ww = w / wsum;
        let mut scaled = *c;
        for r in 0..4 {
            for col in 0..4 {
                scaled.0[r][col] *= ww;
            }
        }
        cov = add4x4(cov, scaled)?;
        let mut delta = VecN([0.0; 4]);
        for d in 0..4 {
            delta.0[d] = m.0[d] - mean.0[d];
        }
        // ww * delta * delta^T
        for r in 0..4 {
            for col in 0..4 {
                let v = ww * delta.0[r] * delta.0[col];
                cov.0[r][col] += v;
            }
        }
    }
    Ok((mean, cov))
}

/// PDA update: posterior states for each hypothesis, merged (Tutorial 07 loop).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn pda_merged_update(
    hyps: &[ProbabilityHypothesis],
    n: usize,
    detections: &[Detection],
    updater: &KalmanUpdater,
    ws: &mut KalmanWorkspace,
) -> Result<GaussianState, Error> {
    if n == 0 {
        return Err(Error::InvalidDimension);
    }
    let mut means = [VecN([0.0; 4]); MAX_PDA_HYPOTHESES];
    let mut covs = [Mat::<4, 4>::zeros(); MAX_PDA_HYPOTHESES];
    let mut weights = [0.0f64; MAX_PDA_HYPOTHESES];
    let ts = if let Some(i) = hyps[1].detection_index {
        detections.get(i).map(|d| d.timestamp)
    } else {
        None
    };
    let ts = ts.unwrap_or(hyps[0].prediction.timestamp);
    for (k, h) in hyps.iter().take(n).enumerate() {
        weights[k] = h.probability;
        let state = match h.detection_index {
            None => h.prediction,
            Some(i) => {
                let det = detections.get(i).ok_or(Error::InvalidDimension)?;
                updater.update(&h.prediction, det, ws)?
            }
        };
        means[k] = state.mean;
        covs[k] = state.cov;
    }
    let (m, c) = gm_reduce_single(&means[..n], &covs[..n], &weights[..n])?;
    GaussianState::new(m, c, ts)
}
