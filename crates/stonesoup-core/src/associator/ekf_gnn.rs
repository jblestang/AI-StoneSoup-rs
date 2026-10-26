//! GNN assignment with EKF bearing–range measurements (sensor management tutorial).

#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::too_many_arguments
)]

use alloc::vec::Vec;

use crate::error::Error;
use crate::filter::ekf::ExtendedKalmanUpdater;
use crate::filter::kalman::{GaussianMeasurementPrediction, KalmanPredictor, KalmanWorkspace};
use crate::linalg::fixed::{add2x2, mul2x4_4x2, mul2x4_4x4, transpose2x4};
use crate::linalg::VecN;
use crate::models::bearing_range::CartesianToBearingRange;
use crate::state::GaussianState;
use crate::time::Timestamp;
use crate::types::Detection;

use super::distance::{mahalanobis, MAX_DISTANCE_HYPOTHESES};
use super::gnn_assignment::{assign_from_per_track_hypotheses, GnnAssignment, MAX_GNN_ASSIGN_DETS};
use super::nearest_neighbour::MeasurementHypothesis;

fn missed_hypothesis(prediction: GaussianState, gate: f64) -> MeasurementHypothesis {
    MeasurementHypothesis {
        prediction,
        detection_index: None,
        distance: gate,
    }
}

pub(crate) fn ekf_meas_pred(
    model: &CartesianToBearingRange,
    predicted: &GaussianState,
    ws: &mut KalmanWorkspace,
) -> Result<GaussianMeasurementPrediction, Error> {
    ws.h = model.jacobian(predicted)?;
    ws.r = model.r();
    let mean = model.predict_measurement(predicted)?;
    let hp = mul2x4_4x4(ws.h, predicted.cov)?;
    let ht = transpose2x4(ws.h)?;
    let hph = mul2x4_4x2(hp, ht)?;
    let cov = add2x2(hph, ws.r)?;
    Ok(GaussianMeasurementPrediction { mean, cov })
}

fn ekf_distance_hypotheses(
    track_tail: &GaussianState,
    association_time: Timestamp,
    detections: &[Detection],
    gate: f64,
    predictor: &KalmanPredictor,
    updater: &ExtendedKalmanUpdater,
    det_models: Option<&[CartesianToBearingRange]>,
    ws: &mut KalmanWorkspace,
    out: &mut [MeasurementHypothesis],
) -> Result<usize, Error> {
    if out.is_empty() {
        return Err(Error::CapacityExceeded);
    }
    let missed_prediction = predictor.predict(track_tail, association_time, ws)?;
    out[0] = missed_hypothesis(missed_prediction, gate);
    let mut n = 1usize;
    for (i, det) in detections.iter().enumerate() {
        if n >= out.len() {
            return Err(Error::CapacityExceeded);
        }
        let prediction = predictor.predict(track_tail, det.timestamp, ws)?;
        let meas_pred = if let Some(models) = det_models {
            let model = models.get(i).ok_or(Error::InvalidDimension)?;
            ekf_meas_pred(model, &prediction, ws)?
        } else {
            updater.predict_measurement(&prediction, ws)?
        };
        let innovation = CartesianToBearingRange::innovation(det.vector.0, meas_pred.mean.0)?;
        let distance = mahalanobis(
            VecN(innovation),
            &GaussianMeasurementPrediction {
                mean: VecN([0.0, 0.0]),
                cov: meas_pred.cov,
            },
        )?;
        if distance < gate {
            out[n] = MeasurementHypothesis {
                prediction,
                detection_index: Some(i),
                distance,
            };
            n += 1;
        }
    }
    Ok(n)
}

/// GNN with [`ExtendedKalmanUpdater`] (Tutorial sensor management stack).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn associate_gnn_2d_ekf(
    track_tails: &[GaussianState],
    association_time: Timestamp,
    detections: &[Detection],
    gate: f64,
    predictor: &KalmanPredictor,
    updater: &ExtendedKalmanUpdater,
    ws: &mut KalmanWorkspace,
    out: &mut [GnnAssignment],
) -> Result<usize, Error> {
    associate_gnn_2d_ekf_det_models(
        track_tails,
        association_time,
        detections,
        gate,
        predictor,
        updater,
        None,
        ws,
        out,
    )
}

/// Like [`associate_gnn_2d_ekf`], with optional per-detection bearing–range models (multi-sensor).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn associate_gnn_2d_ekf_det_models(
    track_tails: &[GaussianState],
    association_time: Timestamp,
    detections: &[Detection],
    gate: f64,
    predictor: &KalmanPredictor,
    updater: &ExtendedKalmanUpdater,
    det_models: Option<&[CartesianToBearingRange]>,
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
        let n = ekf_distance_hypotheses(
            tail,
            association_time,
            detections,
            gate,
            predictor,
            updater,
            det_models,
            ws,
            &mut scratch,
        )?;
        per_track.push(scratch[..n].to_vec());
    }

    assign_from_per_track_hypotheses(n_tracks, n_dets, gate, template, &per_track, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Mat;
    use crate::models::transition::ConstantVelocity2d;
    use crate::types::SensorId;

    #[test]
    fn lot10_brute_step1_track2_det1_gates() {
        let half_deg = 0.5 * core::f64::consts::PI / 180.0;
        let mut r = Mat::<2, 2>::zeros();
        r.set(0, 0, half_deg * half_deg).unwrap();
        r.set(1, 1, 1.0).unwrap();
        let radar = CartesianToBearingRange {
            sensor_x: 10.0,
            sensor_y: 0.0,
            noise_covar: r,
            antenna_heading_rad: 2.2165695462962,
        };
        let transition = ConstantVelocity2d::new(0.005, 0.005).unwrap();
        let predictor = KalmanPredictor::new(transition);
        let updater = ExtendedKalmanUpdater::new(radar);
        let mut ws = KalmanWorkspace {
            f: Mat::zeros(),
            q: Mat::zeros(),
            h: Mat::zeros(),
            r: Mat::zeros(),
        };
        let mut cov = Mat::<4, 4>::zeros();
        for i in 0..4 {
            cov.set(i, i, 0.5).unwrap();
        }
        let mean = [0.0, 1.2, 20.1, -1.2];
        let tail = GaussianState::new_from_array(mean, cov, Timestamp::ZERO).expect("prior");
        let ts = Timestamp::from_nanos(crate::time::Duration::SEC.0);
        let det = Detection::new([-0.2033561326114124, 20.956463305501085], ts, SensorId(0))
            .expect("det");
        let mut scratch = [missed_hypothesis(tail, 5.0); MAX_DISTANCE_HYPOTHESES];
        let prediction = predictor.predict(&tail, ts, &mut ws).expect("pred");
        let meas_pred = updater
            .predict_measurement(&prediction, &mut ws)
            .expect("mp");
        let innovation =
            CartesianToBearingRange::innovation(det.vector.0, meas_pred.mean.0).unwrap();
        let dist = mahalanobis(
            VecN(innovation),
            &GaussianMeasurementPrediction {
                mean: VecN([0.0, 0.0]),
                cov: meas_pred.cov,
            },
        )
        .unwrap();
        let n = ekf_distance_hypotheses(
            &tail,
            ts,
            core::slice::from_ref(&det),
            5.0,
            &predictor,
            &updater,
            None,
            &mut ws,
            &mut scratch,
        )
        .expect("hyp");
        assert!(n >= 2, "expected gated det, got n={n} dist={dist}");
        assert!(scratch[1].distance < 5.0, "dist {}", scratch[1].distance);
    }

    #[test]
    fn lot10_brute_step1_track1_update() {
        let half_deg = 0.5 * core::f64::consts::PI / 180.0;
        let mut r = Mat::<2, 2>::zeros();
        r.set(0, 0, half_deg * half_deg).unwrap();
        r.set(1, 1, 1.0).unwrap();
        let radar = CartesianToBearingRange {
            sensor_x: 10.0,
            sensor_y: 0.0,
            noise_covar: r,
            antenna_heading_rad: 2.2165695462962,
        };
        let transition = ConstantVelocity2d::new(0.005, 0.005).unwrap();
        let predictor = KalmanPredictor::new(transition);
        let updater = ExtendedKalmanUpdater::new(radar);
        let mut ws = KalmanWorkspace {
            f: Mat::zeros(),
            q: Mat::zeros(),
            h: Mat::zeros(),
            r: Mat::zeros(),
        };
        let mut cov = Mat::<4, 4>::zeros();
        for i in 0..4 {
            cov.set(i, i, 0.5).unwrap();
        }
        let mean = [0.0, -1.2, 10.1, -1.2];
        let tail = GaussianState::new_from_array(mean, cov, Timestamp::ZERO).expect("prior");
        let ts = Timestamp::from_nanos(crate::time::Duration::SEC.0);
        let det =
            Detection::new([0.22514697825657493, 13.81544708413785], ts, SensorId(0)).expect("det");
        let prediction = predictor.predict(&tail, ts, &mut ws).expect("pred");
        let mp = updater
            .predict_measurement(&prediction, &mut ws)
            .expect("mp");
        assert!(
            (mp.mean.0[0] - 0.2535572625304434).abs() < 1e-4,
            "bearing pred got {} ref 0.2536",
            mp.mean.0[0]
        );
        let post = updater.update(&prediction, &det, &mut ws).expect("post");
        let m = post.mean.0;
        assert!(
            (m[0] - (-0.7589922878426933)).abs() < 2e-3,
            "x got {} ref -0.759",
            m[0]
        );
        assert!(
            (m[2] - 9.060724772803546).abs() < 2e-3,
            "y got {} ref 9.061",
            m[2]
        );
    }
}
