//! 3-model IMM-EKF: Singer CV, constant-acceleration, and coordinated-turn (truth-aligned dynamics).

use crate::associator::{ProbabilityHypothesis, squared_mahalanobis, MAX_PDA_HYPOTHESES};
use crate::error::Error;
use crate::filter::ekf::ExtendedKalmanUpdater;
use crate::filter::kalman::{GaussianMeasurementPrediction, KalmanWorkspace};
use crate::linalg::fixed::inv2;
use crate::linalg::{Mat, VecN};
use crate::models::bearing_range::CartesianToBearingRange;
use crate::state::GaussianState;
use crate::time::Timestamp;
use crate::types::Detection;

use super::imm_dynamics::{ConstantAcceleration6d, CoordinatedTurn6d, SingerCv6d};
use super::imm_state6::{
    gm_reduce_6, predict_linear6, GaussianState6, IMM_STATE_DIM,
};

pub const IMM_MODEL_COUNT: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImmModelKind {
    Cv = 0,
    Ca = 1,
    Ct = 2,
}

#[derive(Debug, Clone, Copy)]
pub struct ImmEkfConfig {
    pub cv: SingerCv6d,
    pub ca: ConstantAcceleration6d,
    pub ct: CoordinatedTurn6d,
    pub tpm: [[f64; IMM_MODEL_COUNT]; IMM_MODEL_COUNT],
}

impl ImmEkfConfig {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn dual_radar_default() -> Result<Self, Error> {
        let tpm = [
            [0.94, 0.03, 0.03],
            [0.03, 0.94, 0.03],
            [0.03, 0.03, 0.94],
        ];
        Ok(Self {
            cv: SingerCv6d {
                q_accel: 25.0,
                alpha: 2.0,
            },
            ca: ConstantAcceleration6d { q_accel: 900.0 },
            ct: CoordinatedTurn6d {
                omega_rad_s: 0.12,
                q_vel: 400.0,
            },
            tpm,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImmEkfState {
    pub models: [GaussianState6; IMM_MODEL_COUNT],
    pub mu: [f64; IMM_MODEL_COUNT],
}

impl ImmEkfState {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn from_gaussian(state: GaussianState) -> Result<Self, Error> {
        let lifted = GaussianState6::from_position_velocity_4d(&state, 200.0)?;
        Ok(Self {
            models: [lifted, lifted, lifted],
            mu: [1.0 / 3.0; IMM_MODEL_COUNT],
        })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn combined(&self) -> Result<GaussianState, Error> {
        let means = [
            self.models[0].mean,
            self.models[1].mean,
            self.models[2].mean,
        ];
        let covs = [
            self.models[0].cov,
            self.models[1].cov,
            self.models[2].cov,
        ];
        let (mean6, cov6) = gm_reduce_6(&means, &covs, &self.mu)?;
        let s6 = GaussianState6::new(mean6, cov6, self.models[0].timestamp)?;
        s6.marginalize_to_4d()
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn interaction_and_predict(
        &mut self,
        timestamp: Timestamp,
        cfg: &ImmEkfConfig,
        _ws: &mut KalmanWorkspace,
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
                mix_gaussians_6(&self.models, &w, self.models[0].timestamp)?
            } else {
                self.models[j]
            };
        }
        for j in 0..IMM_MODEL_COUNT {
            self.models[j] = match predict_imm_model(mixed[j], timestamp, model_kind(j), cfg) {
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
            likelihoods[j] = jpda_mixture_likelihood_6(
                self.models[j],
                model_kind(j),
                association_time,
                hyps,
                n_hyps,
                detections,
                det_models,
                updater,
                cfg,
            )?;
            let retargeted = retarget_hypotheses_6(
                self.models[j],
                model_kind(j),
                association_time,
                hyps,
                n_hyps,
                detections,
                cfg,
            )?;
            updated[j] = match ekf_pda_merged_update_6(
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
fn mix_gaussians_6(
    states: &[GaussianState6; IMM_MODEL_COUNT],
    weights: &[f64; IMM_MODEL_COUNT],
    timestamp: Timestamp,
) -> Result<GaussianState6, Error> {
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
        return GaussianState6::new(states[best].mean, states[best].cov, timestamp);
    }
    let means = [states[0].mean, states[1].mean, states[2].mean];
    let covs = [states[0].cov, states[1].cov, states[2].cov];
    let (mean, cov) = gm_reduce_6(&means, &covs, weights)?;
    GaussianState6::new(mean, cov, timestamp)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn predict_imm_model(
    prior: GaussianState6,
    timestamp: Timestamp,
    kind: ImmModelKind,
    cfg: &ImmEkfConfig,
) -> Result<GaussianState6, Error> {
    if prior.timestamp == timestamp {
        return Ok(prior);
    }
    let dt = timestamp.checked_sub(prior.timestamp)?;
    match kind {
        ImmModelKind::Cv => {
            let (f, q) = cfg.cv.f_q(dt)?;
            predict_linear6(&prior, &f, &q, timestamp)
        }
        ImmModelKind::Ca => {
            let (f, q) = cfg.ca.f_q(dt)?;
            predict_linear6(&prior, &f, &q, timestamp)
        }
        ImmModelKind::Ct => cfg.ct.predict(&prior, timestamp),
    }
}

#[derive(Debug, Clone, Copy)]
struct Hypothesis6 {
    prediction: GaussianState6,
    detection_index: Option<usize>,
    probability: f64,
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn retarget_hypotheses_6(
    model_state: GaussianState6,
    kind: ImmModelKind,
    association_time: Timestamp,
    hyps: &[ProbabilityHypothesis],
    n_hyps: usize,
    detections: &[Detection],
    cfg: &ImmEkfConfig,
) -> Result<[Hypothesis6; MAX_PDA_HYPOTHESES], Error> {
    let mut out = [Hypothesis6 {
        prediction: model_state,
        detection_index: None,
        probability: 0.0,
    }; MAX_PDA_HYPOTHESES];
    for (k, h) in hyps.iter().take(n_hyps).enumerate() {
        let ts = match h.detection_index {
            None => association_time,
            Some(i) => detections.get(i).map(|d| d.timestamp).unwrap_or(association_time),
        };
        let prediction = match predict_imm_model(model_state, ts, kind, cfg) {
            Ok(s) => s,
            Err(_) => model_state,
        };
        out[k] = Hypothesis6 {
            prediction,
            detection_index: h.detection_index,
            probability: h.probability,
        };
    }
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn jpda_mixture_likelihood_6(
    model_state: GaussianState6,
    kind: ImmModelKind,
    association_time: Timestamp,
    hyps: &[ProbabilityHypothesis],
    n_hyps: usize,
    detections: &[Detection],
    det_models: &[CartesianToBearingRange],
    updater: &mut ExtendedKalmanUpdater,
    cfg: &ImmEkfConfig,
) -> Result<f64, Error> {
    let mut like: f64 = 0.0;
    for h in hyps.iter().take(n_hyps) {
        let ts = match h.detection_index {
            None => association_time,
            Some(i) => detections.get(i).map(|d| d.timestamp).unwrap_or(association_time),
        };
        let pred = match predict_imm_model(model_state, ts, kind, cfg) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if let Some(i) = h.detection_index {
            let det = detections.get(i).ok_or(Error::InvalidDimension)?;
            updater.measurement = det_models.get(i).copied().ok_or(Error::InvalidDimension)?;
            let meas = match predict_measurement_6(updater, &pred) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let innov = match CartesianToBearingRange::innovation(det.vector.0, meas.mean.0) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Ok(m2) = squared_mahalanobis(
                VecN(innov),
                &GaussianMeasurementPrediction {
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

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn predict_measurement_6(
    updater: &ExtendedKalmanUpdater,
    predicted: &GaussianState6,
) -> Result<GaussianMeasurementPrediction, Error> {
    let h = updater.measurement.jacobian6(&predicted.mean)?;
    let r = updater.measurement.r();
    let z = updater.measurement.h6(&predicted.mean)?;
    let hp = mul2x6_6x6(h, predicted.cov);
    let ht = transpose2x6(h);
    let hph = mul2x6_6x2(hp, ht);
    let cov = add2x2(hph, r)?;
    Ok(GaussianMeasurementPrediction {
        mean: VecN(z),
        cov,
    })
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn ekf_update_6(
    updater: &ExtendedKalmanUpdater,
    predicted: &GaussianState6,
    detection: &Detection,
) -> Result<GaussianState6, Error> {
    let h = updater.measurement.jacobian6(&predicted.mean)?;
    let r = updater.measurement.r();
    let z_pred = updater.measurement.h6(&predicted.mean)?;
    let innovation = CartesianToBearingRange::innovation(detection.vector.0, z_pred)?;
    let hp = mul2x6_6x6(h, predicted.cov);
    let ht = transpose2x6(h);
    let hph = mul2x6_6x2(hp, ht);
    let s = add2x2(hph, r)?;
    let s_inv = inv2(s)?;
    let pht = mul6x6_6x2(predicted.cov, ht);
    let k = mul6x2_2x2(pht, s_inv);
    let k_innov = mul6x2_2x1(k, VecN(innovation));
    let mut mean = predicted.mean;
    for i in 0..IMM_STATE_DIM {
        mean[i] += k_innov.0[i];
    }
    let kh = mul6x2_2x6(k, h);
    let i_kh = i_minus_kh6(kh)?;
    let cov = mul6x6_6x6(i_kh, predicted.cov);
    GaussianState6::new(mean, cov, predicted.timestamp)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn ekf_pda_merged_update_6(
    hyps: &[Hypothesis6],
    n: usize,
    detections: &[Detection],
    det_models: &[CartesianToBearingRange],
    updater: &mut ExtendedKalmanUpdater,
    _ws: &mut KalmanWorkspace,
) -> Result<GaussianState6, Error> {
    if n == 0 {
        return Err(Error::InvalidDimension);
    }
    let mut means = [[0.0; IMM_STATE_DIM]; MAX_PDA_HYPOTHESES];
    let mut covs = [[[0.0; IMM_STATE_DIM]; IMM_STATE_DIM]; MAX_PDA_HYPOTHESES];
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
                match ekf_update_6(updater, &h.prediction, det) {
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
    let (m, c) = gm_reduce_6(&means[..n], &covs[..n], &weights[..n])?;
    GaussianState6::new(m, c, ts)
}

// --- small 6D EKF linear algebra ---

type Cov6 = [[f64; IMM_STATE_DIM]; IMM_STATE_DIM];

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn mul2x6_6x6(h: Mat<2, 6>, p: Cov6) -> Mat<2, 6> {
    let mut out = Mat::<2, 6>::zeros();
    for r in 0..2 {
        for c in 0..IMM_STATE_DIM {
            let mut s = 0.0;
            for k in 0..IMM_STATE_DIM {
                s += h.0[r][k] * p[k][c];
            }
            let _ = out.set(r, c, s);
        }
    }
    out
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn transpose2x6(h: Mat<2, 6>) -> Mat<6, 2> {
    let mut out = Mat::<6, 2>::zeros();
    for r in 0..2 {
        for c in 0..IMM_STATE_DIM {
            let _ = out.set(c, r, h.0[r][c]);
        }
    }
    out
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn mul2x6_6x2(a: Mat<2, 6>, b: Mat<6, 2>) -> Mat<2, 2> {
    let mut out = Mat::<2, 2>::zeros();
    for r in 0..2 {
        for c in 0..2 {
            let mut s = 0.0;
            for k in 0..IMM_STATE_DIM {
                s += a.0[r][k] * b.0[k][c];
            }
            let _ = out.set(r, c, s);
        }
    }
    out
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn add2x2(a: Mat<2, 2>, b: Mat<2, 2>) -> Result<Mat<2, 2>, Error> {
    let mut out = Mat::<2, 2>::zeros();
    for r in 0..2 {
        for c in 0..2 {
            out.set(r, c, a.0[r][c] + b.0[r][c])?;
        }
    }
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn mul6x6_6x2(p: Cov6, ht: Mat<6, 2>) -> Mat<6, 2> {
    let mut out = Mat::<6, 2>::zeros();
    for r in 0..IMM_STATE_DIM {
        for c in 0..2 {
            let mut s = 0.0;
            for k in 0..IMM_STATE_DIM {
                s += p[r][k] * ht.0[k][c];
            }
            let _ = out.set(r, c, s);
        }
    }
    out
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn mul6x2_2x2(a: Mat<6, 2>, b: Mat<2, 2>) -> Mat<6, 2> {
    let mut out = Mat::<6, 2>::zeros();
    for r in 0..IMM_STATE_DIM {
        for c in 0..2 {
            let mut s = 0.0;
            for k in 0..2 {
                s += a.0[r][k] * b.0[k][c];
            }
            let _ = out.set(r, c, s);
        }
    }
    out
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn mul6x2_2x1(k: Mat<6, 2>, v: VecN<2>) -> VecN<6> {
    let mut out = [0.0; 6];
    for r in 0..IMM_STATE_DIM {
        out[r] = k.0[r][0] * v.0[0] + k.0[r][1] * v.0[1];
    }
    VecN(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn mul6x2_2x6(k: Mat<6, 2>, h: Mat<2, 6>) -> Cov6 {
    let mut out = [[0.0; IMM_STATE_DIM]; IMM_STATE_DIM];
    for r in 0..IMM_STATE_DIM {
        for c in 0..IMM_STATE_DIM {
            out[r][c] = k.0[r][0] * h.0[0][c] + k.0[r][1] * h.0[1][c];
        }
    }
    out
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn i_minus_kh6(kh: Cov6) -> Result<Cov6, Error> {
    let mut out = [[0.0; IMM_STATE_DIM]; IMM_STATE_DIM];
    for r in 0..IMM_STATE_DIM {
        for c in 0..IMM_STATE_DIM {
            let v = if r == c { 1.0 - kh[r][c] } else { -kh[r][c] };
            if !v.is_finite() {
                return Err(Error::NumericalFailure);
            }
            out[r][c] = v;
        }
    }
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn mul6x6_6x6(a: Cov6, b: Cov6) -> Cov6 {
    super::imm_state6::mat6_mul(&a, &b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::Duration;

    #[test]
    fn ca_model_integrates_acceleration_on_mean() {
        let cfg = ImmEkfConfig::dual_radar_default().expect("cfg");
        let t0 = Timestamp::ZERO;
        let dt = Duration::from_seconds(1.0).expect("dt");
        let mut mean = [0.0; IMM_STATE_DIM];
        mean[0] = 100_000.0;
        mean[1] = 200.0;
        mean[2] = 5.0;
        mean[3] = 50_000.0;
        mean[4] = -50.0;
        mean[5] = -2.0;
        let cov = [[0.0; IMM_STATE_DIM]; IMM_STATE_DIM];
        let prior = GaussianState6::new(mean, cov, t0).expect("prior");
        let t1 = t0.checked_add(dt).expect("t1");
        let pred = predict_imm_model(prior, t1, ImmModelKind::Ca, &cfg).expect("pred");
        assert!((pred.mean[0] - (100_000.0 + 200.0 + 2.5)).abs() < 1e-6);
        assert!((pred.mean[1] - 205.0).abs() < 1e-6);
    }

    #[test]
    fn interaction_and_predict_coasts_without_measurements() {
        let cfg = ImmEkfConfig::dual_radar_default().expect("cfg");
        let mut ws = KalmanWorkspace {
            f: Mat::<4, 4>::zeros(),
            q: Mat::<4, 4>::zeros(),
            h: Mat::<2, 4>::zeros(),
            r: Mat::<2, 2>::zeros(),
        };
        let t0 = Timestamp::ZERO;
        let dt = Duration::from_seconds(0.25).expect("dt");
        let mut cov = Mat::<4, 4>::zeros();
        for i in 0..4 {
            cov.0[i][i] = 1e4;
        }
        let mut imm = ImmEkfState::from_gaussian(
            GaussianState::new_from_array([100_000.0, 200.0, 50_000.0, -100.0], cov, t0)
                .expect("state"),
        )
        .expect("imm");
        let p0 = imm.combined().expect("combined").cov.0[0][0];
        let t1 = t0.checked_add(dt).expect("t1");
        imm.interaction_and_predict(t1, &cfg, &mut ws)
            .expect("predict");
        let p1 = imm.combined().expect("combined").cov.0[0][0];
        assert!(p1 > p0, "coast step should inflate position variance: {p0} -> {p1}");
    }
}
