//! 3-model IMM-EKF on `[x, vx, y, vy]`: CV, CA, and a third high–process-noise CV manoeuvre slot.

use crate::associator::{gm_reduce_single, ProbabilityHypothesis, squared_mahalanobis};
use crate::error::Error;
use crate::filter::ekf::ExtendedKalmanUpdater;
use crate::filter::kalman::{KalmanPredictor, KalmanWorkspace};
use crate::linalg::VecN;
use crate::models::bearing_range::CartesianToBearingRange;
use crate::models::transition::ConstantVelocity2d;
use crate::state::GaussianState;
use crate::time::Timestamp;
use crate::types::Detection;

pub const IMM_MODEL_COUNT: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImmModelKind {
    Cv = 0,
    Ca = 1,
    Ct = 2,
}

#[derive(Debug, Clone, Copy)]
pub struct ImmEkfConfig {
    pub cv: KalmanPredictor,
    pub ca: KalmanPredictor,
    /// Third sub-model: very agile CV (manoeuvre / turn-like motion via high `q`).
    pub ct: KalmanPredictor,
    /// Row-stochastic Markov transition (imm_i → imm_j).
    pub tpm: [[f64; IMM_MODEL_COUNT]; IMM_MODEL_COUNT],
}

impl ImmEkfConfig {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn dual_radar_default() -> Result<Self, Error> {
        let cv = KalmanPredictor::new(ConstantVelocity2d::new(40.0, 40.0)?);
        let ca = KalmanPredictor::new(ConstantVelocity2d::new(4_000.0, 4_000.0)?);
        let ct = KalmanPredictor::new(ConstantVelocity2d::new(12_000.0, 12_000.0)?);
        let tpm = [
            [0.92, 0.04, 0.04],
            [0.04, 0.92, 0.04],
            [0.04, 0.04, 0.92],
        ];
        Ok(Self {
            cv,
            ca,
            ct,
            tpm,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImmEkfState {
    pub models: [GaussianState; IMM_MODEL_COUNT],
    pub mu: [f64; IMM_MODEL_COUNT],
}

impl ImmEkfState {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn from_gaussian(state: GaussianState) -> Self {
        Self {
            models: [state, state, state],
            mu: [1.0 / 3.0; IMM_MODEL_COUNT],
        }
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn combined(&self) -> Result<GaussianState, Error> {
        mix_gaussians(&self.models, &self.mu, self.models[0].timestamp)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn interaction_and_predict(
        &mut self,
        timestamp: Timestamp,
        cfg: &ImmEkfConfig,
        ws: &mut KalmanWorkspace,
    ) -> Result<(), Error> {
        let c_bar = mixing_normalizer(&self.mu, &cfg.tpm);
        let mut mixed = [self.models[0]; IMM_MODEL_COUNT];
        for j in 0..IMM_MODEL_COUNT {
            let mut w = [0.0; IMM_MODEL_COUNT];
            for i in 0..IMM_MODEL_COUNT {
                w[i] = if c_bar[j] > 1e-18 {
                    self.mu[i] * cfg.tpm[i][j] / c_bar[j]
                } else {
                    0.0
                };
            }
            let wsum: f64 = w.iter().sum();
            mixed[j] = if c_bar[j] > 1e-18 && wsum > 1e-18 {
                mix_gaussians(&self.models, &w, self.models[0].timestamp)?
            } else {
                self.models[j]
            };
        }
        for j in 0..IMM_MODEL_COUNT {
            self.models[j] = match predict_imm_model(mixed[j], timestamp, model_kind(j), cfg, ws) {
                Ok(s) => s,
                Err(_) => mixed[j],
            };
        }
        Ok(())
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn jpda_update(
        &mut self,
        hyps: &[ProbabilityHypothesis],
        n_hyps: usize,
        detections: &[Detection],
        det_models: &[CartesianToBearingRange],
        association_time: Timestamp,
        updater: &mut ExtendedKalmanUpdater,
        cfg: &ImmEkfConfig,
        ws: &mut KalmanWorkspace,
    ) -> Result<(), Error> {
        if n_hyps == 0 {
            return Err(Error::InvalidDimension);
        }
        let c_bar = mixing_normalizer(&self.mu, &cfg.tpm);
        let mut likelihoods = [0.0; IMM_MODEL_COUNT];
        let mut updated = self.models;
        for j in 0..IMM_MODEL_COUNT {
            likelihoods[j] = jpda_mixture_likelihood(
                self.models[j],
                model_kind(j),
                association_time,
                hyps,
                n_hyps,
                detections,
                det_models,
                updater,
                cfg,
                ws,
            )?;
            let retargeted = retarget_hypotheses(
                self.models[j],
                model_kind(j),
                association_time,
                hyps,
                n_hyps,
                detections,
                cfg,
                ws,
            )?;
            updated[j] = match crate::associator::ekf_pda_merged_update(
                &retargeted,
                n_hyps,
                detections,
                det_models,
                updater,
                ws,
            ) {
                Ok(s) => s,
                Err(Error::SingularInnovation)
                | Err(Error::InvalidParameter)
                | Err(Error::NumericalFailure) => self.models[j],
                Err(e) => return Err(e),
            };
        }
        self.models = updated;
        let mut mu_new = [0.0; IMM_MODEL_COUNT];
        let mut sum = 0.0;
        for j in 0..IMM_MODEL_COUNT {
            mu_new[j] = c_bar[j] * likelihoods[j];
            sum += mu_new[j];
        }
        if sum > 1e-300 {
            for j in 0..IMM_MODEL_COUNT {
                self.mu[j] = mu_new[j] / sum;
            }
        }
        Ok(())
    }
}

fn model_kind(j: usize) -> ImmModelKind {
    match j {
        0 => ImmModelKind::Cv,
        1 => ImmModelKind::Ca,
        _ => ImmModelKind::Ct,
    }
}

fn mixing_normalizer(mu: &[f64; IMM_MODEL_COUNT], tpm: &[[f64; 3]; 3]) -> [f64; IMM_MODEL_COUNT] {
    let mut c = [0.0; IMM_MODEL_COUNT];
    for j in 0..IMM_MODEL_COUNT {
        for i in 0..IMM_MODEL_COUNT {
            c[j] += mu[i] * tpm[i][j];
        }
    }
    c
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn mix_gaussians(
    states: &[GaussianState; IMM_MODEL_COUNT],
    weights: &[f64; IMM_MODEL_COUNT],
    timestamp: Timestamp,
) -> Result<GaussianState, Error> {
    let wsum: f64 = weights.iter().sum();
    if wsum <= 1e-18 {
        let mut best = 0usize;
        let mut bw = weights[0];
        for i in 1..IMM_MODEL_COUNT {
            if weights[i] > bw {
                bw = weights[i];
                best = i;
            }
        }
        return GaussianState::new(states[best].mean, states[best].cov, timestamp);
    }
    let means = [
        states[0].mean,
        states[1].mean,
        states[2].mean,
    ];
    let covs = [states[0].cov, states[1].cov, states[2].cov];
    let (mean, cov) = gm_reduce_single(&means, &covs, weights)?;
    GaussianState::new(mean, cov, timestamp)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn predict_imm_model(
    prior: GaussianState,
    timestamp: Timestamp,
    kind: ImmModelKind,
    cfg: &ImmEkfConfig,
    ws: &mut KalmanWorkspace,
) -> Result<GaussianState, Error> {
    match kind {
        ImmModelKind::Cv => cfg.cv.predict(&prior, timestamp, ws),
        ImmModelKind::Ca => cfg.ca.predict(&prior, timestamp, ws),
        ImmModelKind::Ct => cfg.ct.predict(&prior, timestamp, ws),
    }
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn retarget_hypotheses(
    model_state: GaussianState,
    kind: ImmModelKind,
    association_time: Timestamp,
    hyps: &[ProbabilityHypothesis],
    n_hyps: usize,
    detections: &[Detection],
    cfg: &ImmEkfConfig,
    ws: &mut KalmanWorkspace,
) -> Result<[ProbabilityHypothesis; crate::associator::MAX_PDA_HYPOTHESES], Error> {
    use crate::associator::MAX_PDA_HYPOTHESES;
    let mut out = [ProbabilityHypothesis {
        prediction: model_state,
        detection_index: None,
        probability: 0.0,
    }; MAX_PDA_HYPOTHESES];
    for (k, h) in hyps.iter().take(n_hyps).enumerate() {
        let ts = match h.detection_index {
            None => association_time,
            Some(i) => detections.get(i).map(|d| d.timestamp).unwrap_or(association_time),
        };
        let prediction = match predict_imm_model(model_state, ts, kind, cfg, ws) {
            Ok(s) => s,
            Err(_) => model_state,
        };
        out[k] = ProbabilityHypothesis {
            prediction,
            detection_index: h.detection_index,
            probability: h.probability,
        };
    }
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn jpda_mixture_likelihood(
    model_state: GaussianState,
    kind: ImmModelKind,
    association_time: Timestamp,
    hyps: &[ProbabilityHypothesis],
    n_hyps: usize,
    detections: &[Detection],
    det_models: &[CartesianToBearingRange],
    updater: &mut ExtendedKalmanUpdater,
    cfg: &ImmEkfConfig,
    ws: &mut KalmanWorkspace,
) -> Result<f64, Error> {
    let mut like: f64 = 0.0;
    for h in hyps.iter().take(n_hyps) {
        let ts = match h.detection_index {
            None => association_time,
            Some(i) => detections.get(i).map(|d| d.timestamp).unwrap_or(association_time),
        };
        let pred = match predict_imm_model(model_state, ts, kind, cfg, ws) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if let Some(i) = h.detection_index {
            let det = detections.get(i).ok_or(Error::InvalidDimension)?;
            updater.measurement = det_models.get(i).copied().ok_or(Error::InvalidDimension)?;
            let meas = match updater.predict_measurement(&pred, ws) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let innov = match CartesianToBearingRange::innovation(det.vector.0, meas.mean.0) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Ok(m2) = squared_mahalanobis(
                VecN(innov),
                &crate::filter::kalman::GaussianMeasurementPrediction {
                    mean: VecN([0.0, 0.0]),
                    cov: meas.cov,
                },
            ) {
                like += h.probability * libm::exp(-0.5 * m2);
            }
        } else {
            like += h.probability;
        }
    }
    Ok(like.max(1e-12))
}
