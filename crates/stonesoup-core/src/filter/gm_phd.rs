//! Gaussian-mixture PHD filter (Stone Soup `PHDUpdater` + mixture reduction).

#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::too_many_arguments
)]

use crate::associator::{
    distance_hypotheses, gm_reduce_single, squared_mahalanobis, MeasurementHypothesis,
};
use crate::error::Error;
use crate::filter::kalman::{KalmanPredictor, KalmanUpdater, KalmanWorkspace};
use crate::linalg::fixed::{inv4, mat4_vec4};
use crate::linalg::{require_finite, Mat, VecN};
use crate::state::GaussianState;
use crate::time::Timestamp;
use crate::types::Detection;

pub const MAX_GMPHD_COMPONENTS: usize = 48;
pub const MAX_GMPHD_DETECTIONS: usize = 16;
/// Upper bound on pre-reduction mixture size: all gated components per detection plus missed.
pub const MAX_GMPHD_UPDATED: usize = MAX_GMPHD_COMPONENTS * (MAX_GMPHD_DETECTIONS + 1);
pub const MAX_GMPHD_HYPS: usize = MAX_GMPHD_DETECTIONS + 1;

const LN_2PI: f64 = 1.8378770664093453;
const WEIGHT_EPS: f64 = f64::MIN_POSITIVE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhdTag {
    Birth,
    Untagged,
    Track(u32),
}

#[derive(Debug, Clone, Copy)]
pub struct PhdComponent {
    pub state: GaussianState,
    pub weight: f64,
    pub tag: PhdTag,
}

#[derive(Debug, Clone, Copy)]
pub struct PhdHypothesis {
    pub component: PhdComponent,
    pub detection_index: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct GmPhdConfig {
    pub prob_detection: f64,
    pub prob_survival: f64,
    pub clutter_spatial_density: f64,
    pub missed_distance: f64,
    pub merge_threshold_sq: f64,
    pub prune_threshold: f64,
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn logpdf2(z: VecN<2>, mean: VecN<2>, cov: Mat<2, 2>) -> Result<f64, Error> {
    let mp = crate::filter::kalman::GaussianMeasurementPrediction { mean, cov };
    let sq = squared_mahalanobis(z, &mp)?;
    let det = cov.0[0][0] * cov.0[1][1] - cov.0[0][1] * cov.0[1][0];
    if det <= 0.0 {
        return Err(Error::NumericalFailure);
    }
    Ok(-0.5 * (2.0 * LN_2PI + libm::log(det) + sq))
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn squared_mahalanobis_state(a: &GaussianState, b: &GaussianState) -> Result<f64, Error> {
    let inv = inv4(a.cov)?;
    let mut delta = VecN([0.0; 4]);
    for i in 0..4 {
        delta.0[i] = a.mean.0[i] - b.mean.0[i];
    }
    let t = mat4_vec4(inv, delta)?;
    let mut sq = 0.0;
    for i in 0..4 {
        sq += t.0[i] * delta.0[i];
    }
    require_finite(sq)?;
    if sq < 0.0 {
        return Err(Error::NumericalFailure);
    }
    Ok(sq)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn merge_two(a: PhdComponent, b: PhdComponent) -> Result<PhdComponent, Error> {
    let wsum = a.weight + b.weight;
    if wsum <= 0.0 {
        return Err(Error::NumericalFailure);
    }
    let (mean, cov) = gm_reduce_single(
        &[a.state.mean, b.state.mean],
        &[a.state.cov, b.state.cov],
        &[a.weight, b.weight],
    )?;
    let mut weight = wsum;
    if weight > 1.0 {
        weight = 1.0;
    }
    let tag = if a.weight >= b.weight { a.tag } else { b.tag };
    let ts = if a.weight >= b.weight {
        a.state.timestamp
    } else {
        b.state.timestamp
    };
    Ok(PhdComponent {
        state: GaussianState::new(mean, cov, ts)?,
        weight,
        tag,
    })
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn gm_phd_prune(
    components: &mut [PhdComponent],
    n: &mut usize,
    threshold: f64,
) -> Result<(), Error> {
    let mut pruned = 0.0;
    let mut keep = 0usize;
    for i in 0..*n {
        if components[i].weight < threshold {
            pruned += components[i].weight;
        } else {
            if keep != i {
                components[keep] = components[i];
            }
            keep += 1;
        }
    }
    if keep == 0 {
        *n = 0;
        return Ok(());
    }
    let share = pruned / (keep as f64);
    for c in components.iter_mut().take(keep) {
        c.weight += share;
    }
    *n = keep;
    Ok(())
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn gm_phd_merge(
    components: &mut [PhdComponent],
    n: &mut usize,
    threshold_sq: f64,
    pool: &mut [PhdComponent],
) -> Result<(), Error> {
    if *n <= 1 {
        return Ok(());
    }
    if pool.len() < *n {
        return Err(Error::CapacityExceeded);
    }
    for i in 1..*n {
        let mut j = i;
        while j > 0 && components[j - 1].weight > components[j].weight {
            components.swap(j, j - 1);
            j -= 1;
        }
    }
    let mut len = *n;
    pool[..len].copy_from_slice(&components[..len]);
    let mut out_n = 0usize;
    while len > 0 {
        let mut best = pool[len - 1];
        len -= 1;
        let mut i = 0usize;
        while i < len {
            let dist = squared_mahalanobis_state(&best.state, &pool[i].state)?;
            if dist < threshold_sq {
                let other = pool[i];
                len -= 1;
                pool[i] = pool[len];
                best = merge_two(best, other)?;
            } else {
                i += 1;
            }
        }
        if out_n >= components.len() {
            return Err(Error::CapacityExceeded);
        }
        components[out_n] = best;
        out_n += 1;
    }
    *n = out_n;
    Ok(())
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn gm_phd_reduce(
    components: &mut [PhdComponent],
    n: &mut usize,
    config: &GmPhdConfig,
    merge_pool: &mut [PhdComponent],
) -> Result<(), Error> {
    if *n == 0 {
        return Ok(());
    }
    gm_phd_prune(components, n, config.prune_threshold)?;
    if *n > 1 {
        gm_phd_merge(components, n, config.merge_threshold_sq, merge_pool)?;
    }
    Ok(())
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn prediction_tag(component: PhdComponent, detection_index: Option<usize>) -> PhdTag {
    match (component.tag, detection_index) {
        (PhdTag::Birth, None) => PhdTag::Birth,
        (PhdTag::Birth, Some(_)) => PhdTag::Untagged,
        (other, _) => other,
    }
}

/// Collect missed-detection hypotheses for all components (one pass).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn gm_phd_collect_missed(
    components: &[PhdComponent],
    n_components: usize,
    detections: &[Detection],
    n_detections: usize,
    timestamp: Timestamp,
    missed_distance: f64,
    predictor: &KalmanPredictor,
    updater: &KalmanUpdater,
    ws: &mut KalmanWorkspace,
    missed: &mut [PhdHypothesis; MAX_GMPHD_COMPONENTS],
    missed_n: &mut usize,
) -> Result<(), Error> {
    *missed_n = 0;
    let mut dummy_cov = Mat::<4, 4>::zeros();
    for i in 0..4 {
        dummy_cov.set(i, i, 1.0)?;
    }
    let dummy_pred =
        GaussianState::new_from_array([0.0, 1.0, 0.0, 1.0], dummy_cov, Timestamp::ZERO)?;
    let mut scratch = [MeasurementHypothesis {
        prediction: dummy_pred,
        detection_index: None,
        distance: 0.0,
    }; MAX_GMPHD_HYPS];
    for comp in components.iter().take(n_components) {
        let n_h = distance_hypotheses(
            &comp.state,
            timestamp,
            &detections[..n_detections],
            missed_distance,
            predictor,
            updater,
            ws,
            &mut scratch,
        )?;
        for h in scratch.iter().take(n_h) {
            if h.detection_index.is_some() {
                continue;
            }
            let tagged = PhdComponent {
                state: h.prediction,
                weight: comp.weight,
                tag: prediction_tag(*comp, None),
            };
            if *missed_n >= MAX_GMPHD_COMPONENTS {
                return Err(Error::CapacityExceeded);
            }
            missed[*missed_n] = PhdHypothesis {
                component: tagged,
                detection_index: None,
            };
            *missed_n += 1;
        }
    }
    Ok(())
}

/// Hypotheses associating `detection_index` with each component (gated).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn gm_phd_hypotheses_for_detection(
    components: &[PhdComponent],
    n_components: usize,
    detections: &[Detection],
    n_detections: usize,
    detection_index: usize,
    timestamp: Timestamp,
    missed_distance: f64,
    predictor: &KalmanPredictor,
    updater: &KalmanUpdater,
    ws: &mut KalmanWorkspace,
    out: &mut [PhdHypothesis; MAX_GMPHD_COMPONENTS],
    out_n: &mut usize,
) -> Result<(), Error> {
    *out_n = 0;
    if detection_index >= n_detections {
        return Err(Error::InvalidDimension);
    }
    let mut dummy_cov = Mat::<4, 4>::zeros();
    for i in 0..4 {
        dummy_cov.set(i, i, 1.0)?;
    }
    let dummy_pred =
        GaussianState::new_from_array([0.0, 1.0, 0.0, 1.0], dummy_cov, Timestamp::ZERO)?;
    let mut scratch = [MeasurementHypothesis {
        prediction: dummy_pred,
        detection_index: None,
        distance: 0.0,
    }; MAX_GMPHD_HYPS];
    for comp in components.iter().take(n_components) {
        let n_h = distance_hypotheses(
            &comp.state,
            timestamp,
            &detections[..n_detections],
            missed_distance,
            predictor,
            updater,
            ws,
            &mut scratch,
        )?;
        for h in scratch.iter().take(n_h) {
            if h.detection_index != Some(detection_index) {
                continue;
            }
            let pred_state =
                predictor.predict(&comp.state, detections[detection_index].timestamp, ws)?;
            let tagged = PhdComponent {
                state: pred_state,
                weight: comp.weight,
                tag: prediction_tag(*comp, Some(detection_index)),
            };
            if *out_n >= MAX_GMPHD_COMPONENTS {
                return Err(Error::CapacityExceeded);
            }
            out[*out_n] = PhdHypothesis {
                component: tagged,
                detection_index: Some(detection_index),
            };
            *out_n += 1;
        }
    }
    Ok(())
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn gm_phd_append_detection_update(
    detection: &Detection,
    group: &[PhdHypothesis],
    group_n: usize,
    config: &GmPhdConfig,
    updater: &KalmanUpdater,
    ws: &mut KalmanWorkspace,
    out: &mut [PhdComponent],
    out_n: &mut usize,
) -> Result<(), Error> {
    let start = *out_n;
    let mut weight_sum = 0.0;
    for gi in 0..group_n {
        let hyp = group[gi];
        let meas_pred = updater.predict_measurement(&hyp.component.state, ws)?;
        let logp = logpdf2(detection.vector, meas_pred.mean, meas_pred.cov)?;
        let q = libm::exp(logp);
        require_finite(q)?;
        let mut new_weight = config.prob_detection * hyp.component.weight * q;
        if hyp.component.tag != PhdTag::Birth {
            new_weight *= config.prob_survival;
        }
        weight_sum += new_weight;
        let post = updater.update(&hyp.component.state, detection, ws)?;
        if *out_n >= out.len() {
            return Err(Error::CapacityExceeded);
        }
        out[*out_n] = PhdComponent {
            state: post,
            weight: new_weight,
            tag: hyp.component.tag,
        };
        *out_n += 1;
    }
    let norm = weight_sum + config.clutter_spatial_density;
    if norm <= 0.0 {
        return Err(Error::NumericalFailure);
    }
    for c in out.iter_mut().take(*out_n).skip(start) {
        c.weight /= norm;
    }
    Ok(())
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn gm_phd_append_missed_update(
    missed: &[PhdHypothesis],
    missed_n: usize,
    config: &GmPhdConfig,
    out: &mut [PhdComponent],
    out_n: &mut usize,
) -> Result<(), Error> {
    let l1 = 1.0;
    for mi in 0..missed_n {
        let hyp = missed[mi];
        if hyp.component.tag == PhdTag::Birth {
            continue;
        }
        let w = hyp.component.weight * (1.0 - config.prob_detection) * l1 * config.prob_survival;
        if *out_n >= out.len() {
            return Err(Error::CapacityExceeded);
        }
        out[*out_n] = PhdComponent {
            state: hyp.component.state,
            weight: w,
            tag: hyp.component.tag,
        };
        *out_n += 1;
    }
    for c in out.iter_mut().take(*out_n) {
        if c.weight == 0.0 {
            c.weight = WEIGHT_EPS;
        }
    }
    Ok(())
}

/// Build PHD hypothesis groups ordered by detection, with missed group last.
#[allow(dead_code)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn gm_phd_hypothesise(
    components: &[PhdComponent],
    n_components: usize,
    detections: &[Detection],
    n_detections: usize,
    timestamp: Timestamp,
    missed_distance: f64,
    predictor: &KalmanPredictor,
    updater: &KalmanUpdater,
    ws: &mut KalmanWorkspace,
    det_groups: &mut [[PhdHypothesis; MAX_GMPHD_COMPONENTS]; MAX_GMPHD_DETECTIONS],
    det_group_n: &mut [usize; MAX_GMPHD_DETECTIONS],
    missed: &mut [PhdHypothesis; MAX_GMPHD_COMPONENTS],
    missed_n: &mut usize,
) -> Result<(), Error> {
    for n in det_group_n.iter_mut() {
        *n = 0;
    }
    *missed_n = 0;
    let mut dummy_cov = Mat::<4, 4>::zeros();
    for i in 0..4 {
        dummy_cov.set(i, i, 1.0)?;
    }
    let dummy_pred =
        GaussianState::new_from_array([0.0, 1.0, 0.0, 1.0], dummy_cov, Timestamp::ZERO)?;
    let mut scratch = [MeasurementHypothesis {
        prediction: dummy_pred,
        detection_index: None,
        distance: 0.0,
    }; MAX_GMPHD_HYPS];
    for comp in components.iter().take(n_components) {
        let n_h = distance_hypotheses(
            &comp.state,
            timestamp,
            &detections[..n_detections],
            missed_distance,
            predictor,
            updater,
            ws,
            &mut scratch,
        )?;
        for h in scratch.iter().take(n_h) {
            let pred_state = if let Some(di) = h.detection_index {
                predictor.predict(&comp.state, detections[di].timestamp, ws)?
            } else {
                h.prediction
            };
            let tagged = PhdComponent {
                state: pred_state,
                weight: comp.weight,
                tag: prediction_tag(*comp, h.detection_index),
            };
            let ph = PhdHypothesis {
                component: tagged,
                detection_index: h.detection_index,
            };
            match h.detection_index {
                None => {
                    if *missed_n >= MAX_GMPHD_COMPONENTS {
                        return Err(Error::CapacityExceeded);
                    }
                    missed[*missed_n] = ph;
                    *missed_n += 1;
                }
                Some(di) => {
                    if di >= MAX_GMPHD_DETECTIONS {
                        return Err(Error::CapacityExceeded);
                    }
                    let slot = &mut det_group_n[di];
                    if *slot >= MAX_GMPHD_COMPONENTS {
                        return Err(Error::CapacityExceeded);
                    }
                    det_groups[di][*slot] = ph;
                    *slot += 1;
                }
            }
        }
    }
    let _ = detections;
    Ok(())
}

#[allow(dead_code)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn gm_phd_update(
    det_groups: &[[PhdHypothesis; MAX_GMPHD_COMPONENTS]; MAX_GMPHD_DETECTIONS],
    det_group_n: &[usize; MAX_GMPHD_DETECTIONS],
    n_detections: usize,
    missed: &[PhdHypothesis],
    missed_n: usize,
    detections: &[Detection],
    config: &GmPhdConfig,
    updater: &KalmanUpdater,
    ws: &mut KalmanWorkspace,
    out: &mut [PhdComponent],
    out_n: &mut usize,
) -> Result<(), Error> {
    *out_n = 0;
    for di in 0..n_detections {
        let gn = det_group_n[di];
        let det = detections[di];
        let start = *out_n;
        let mut weight_sum = 0.0;
        for gi in 0..gn {
            let hyp = det_groups[di][gi];
            let meas_pred = updater.predict_measurement(&hyp.component.state, ws)?;
            let logp = logpdf2(det.vector, meas_pred.mean, meas_pred.cov)?;
            let q = libm::exp(logp);
            require_finite(q)?;
            let mut new_weight = config.prob_detection * hyp.component.weight * q;
            if hyp.component.tag != PhdTag::Birth {
                new_weight *= config.prob_survival;
            }
            weight_sum += new_weight;
            let post = updater.update(&hyp.component.state, &det, ws)?;
            if *out_n >= MAX_GMPHD_UPDATED {
                return Err(Error::CapacityExceeded);
            }
            out[*out_n] = PhdComponent {
                state: post,
                weight: new_weight,
                tag: hyp.component.tag,
            };
            *out_n += 1;
        }
        let norm = weight_sum + config.clutter_spatial_density;
        if norm <= 0.0 {
            return Err(Error::NumericalFailure);
        }
        for c in out.iter_mut().take(*out_n).skip(start) {
            c.weight /= norm;
        }
    }
    let l1 = 1.0;
    for mi in 0..missed_n {
        let hyp = missed[mi];
        if hyp.component.tag == PhdTag::Birth {
            continue;
        }
        let w = hyp.component.weight * (1.0 - config.prob_detection) * l1 * config.prob_survival;
        if *out_n >= MAX_GMPHD_UPDATED {
            return Err(Error::CapacityExceeded);
        }
        out[*out_n] = PhdComponent {
            state: hyp.component.state,
            weight: w,
            tag: hyp.component.tag,
        };
        *out_n += 1;
    }
    for c in out.iter_mut().take(*out_n) {
        if c.weight == 0.0 {
            c.weight = WEIGHT_EPS;
        }
    }
    Ok(())
}
