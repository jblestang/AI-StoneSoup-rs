#[allow(dead_code)]
mod fixtures {
    include!("lot10_single_sensor_fixtures.rs");
}
use fixtures::*;

use stonesoup_core::linalg::Mat;
use stonesoup_core::{
    associate_gnn_2d_ekf, CartesianToBearingRange, ConstantVelocity2d, Detection, Duration,
    ExtendedKalmanUpdater, GaussianState, GnnAssignment, KalmanPredictor, KalmanWorkspace,
    MeasurementHypothesis, SensorId, Timestamp,
};

use crate::lot3_clutter::MAX_DETS_PER_STEP;
use crate::registry::ScenarioError;
use crate::scenario::{empty_det_flags, empty_det_xy, empty_resample_xy, ViewerSnapshot};

pub const MAX_LOT10_TRACKS: usize = LOT10_NUM_TRUTHS;
pub const MAX_LOT10_DETS: usize = 8;
pub const MAX_LOT10_STEPS: usize = LOT10_NUM_STEPS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lot10SensorManagerMode {
    Random,
    BruteForce,
}

pub struct Lot10SingleSensorScenario {
    mode: Lot10SensorManagerMode,
    predictor: KalmanPredictor,
    updater: ExtendedKalmanUpdater,
    ws: KalmanWorkspace,
    tracks: [GaussianState; MAX_LOT10_TRACKS],
    step: usize,
    meas_offsets: &'static [usize],
    meas_br: &'static [[f64; 2]],
}

pub const LOT10_SENSOR_XY: (f64, f64) = (10.0, 0.0);

fn lot10_bearing_range_to_xy(bearing: f64, range: f64) -> (f64, f64) {
    (
        LOT10_SENSOR_XY.0 + range * libm::cos(bearing),
        LOT10_SENSOR_XY.1 + range * libm::sin(bearing),
    )
}

fn push_bearing_range_measurements<const MAX: usize>(
    meas_offsets: &[usize],
    meas_br: &[[f64; 2]],
    step: usize,
    ts: Timestamp,
    out: &mut [Detection; MAX],
) -> Result<usize, ScenarioError> {
    let start = meas_offsets
        .get(step)
        .copied()
        .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
    let end = meas_offsets
        .get(step + 1)
        .copied()
        .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
    let count = end - start;
    if count > MAX {
        return Err(ScenarioError::Core(stonesoup_core::Error::CapacityExceeded));
    }
    for (j, idx) in (start..end).enumerate() {
        let br = meas_br
            .get(idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        out[j] = Detection::new(br, ts, SensorId(0)).map_err(ScenarioError::Core)?;
    }
    Ok(count)
}

impl Lot10SingleSensorScenario {
    fn lot10_radar() -> Result<CartesianToBearingRange, ScenarioError> {
        let half_deg = 0.5 * core::f64::consts::PI / 180.0;
        let mut r = Mat::<2, 2>::zeros();
        r.set(0, 0, half_deg * half_deg)
            .map_err(ScenarioError::Core)?;
        r.set(1, 1, 1.0).map_err(ScenarioError::Core)?;
        Ok(CartesianToBearingRange {
            sensor_x: 10.0,
            sensor_y: 0.0,
            noise_covar: r,
            antenna_heading_rad: 0.0,
        })
    }

    fn priors(ts: Timestamp) -> Result<[GaussianState; MAX_LOT10_TRACKS], ScenarioError> {
        let mut out = [GaussianState::new_from_array([0.0; 4], Mat::zeros(), ts)
            .map_err(ScenarioError::Core)?; MAX_LOT10_TRACKS];
        let mut cov = Mat::<4, 4>::zeros();
        for i in 0..4 {
            cov.set(i, i, 0.5).map_err(ScenarioError::Core)?;
        }
        for (i, mean) in LOT10_PRIOR_MEANS.iter().enumerate() {
            out[i] = GaussianState::new_from_array(*mean, cov, ts).map_err(ScenarioError::Core)?;
        }
        Ok(out)
    }

    pub fn tutorial_defaults(mode: Lot10SensorManagerMode) -> Result<Self, ScenarioError> {
        let (meas_offsets, meas_br) = match mode {
            Lot10SensorManagerMode::Random => (
                &LOT10_RANDOM_DET_OFFSETS as &[usize],
                &LOT10_RANDOM_DETECTIONS as &[[f64; 2]],
            ),
            Lot10SensorManagerMode::BruteForce => (
                &LOT10_BRUTE_DET_OFFSETS as &[usize],
                &LOT10_BRUTE_DETECTIONS as &[[f64; 2]],
            ),
        };
        let transition = ConstantVelocity2d::new(0.005, 0.005).map_err(ScenarioError::Core)?;
        let predictor = KalmanPredictor::new(transition);
        let updater = ExtendedKalmanUpdater::new(Self::lot10_radar()?);
        let ts = Timestamp::ZERO;
        Ok(Self {
            mode,
            predictor,
            updater,
            ws: KalmanWorkspace {
                f: Mat::zeros(),
                q: Mat::zeros(),
                h: Mat::zeros(),
                r: Mat::zeros(),
            },
            tracks: Self::priors(ts)?,
            step: 0,
            meas_offsets,
            meas_br,
        })
    }

    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        self.tracks = Self::priors(Timestamp::ZERO)?;
        self.step = 0;
        Ok(())
    }

    fn ts(step: usize) -> Timestamp {
        Timestamp::from_nanos((step as i64) * Duration::SEC.0)
    }

    pub fn step_index(&self) -> usize {
        self.step
    }

    pub fn sorted_track_means(&self, out: &mut [[f64; 4]; MAX_LOT10_TRACKS]) -> usize {
        let mut n = 0usize;
        for t in &self.tracks {
            let m = t.mean.0;
            out[n] = [m[0], m[1], m[2], m[3]];
            n += 1;
        }
        for i in 1..n {
            let mut j = i;
            while j > 0 {
                let a = out[j];
                let b = out[j - 1];
                let ord = a[0]
                    .partial_cmp(&b[0])
                    .unwrap_or(core::cmp::Ordering::Equal)
                    .then(
                        a[2].partial_cmp(&b[2])
                            .unwrap_or(core::cmp::Ordering::Equal),
                    );
                if ord == core::cmp::Ordering::Less {
                    out[j] = b;
                    out[j - 1] = a;
                    j -= 1;
                } else {
                    break;
                }
            }
        }
        n
    }

    pub fn reference_track_means_at(
        step: usize,
        mode: Lot10SensorManagerMode,
    ) -> Result<&'static [[f64; 4]], ScenarioError> {
        let (offsets, means) = match mode {
            Lot10SensorManagerMode::Random => {
                (&LOT10_RANDOM_TRACK_OFFSETS, &LOT10_RANDOM_TRACK_MEANS)
            }
            Lot10SensorManagerMode::BruteForce => {
                (&LOT10_BRUTE_TRACK_OFFSETS, &LOT10_BRUTE_TRACK_MEANS)
            }
        };
        if step == 0 {
            return Ok(&LOT10_PRIOR_MEANS);
        }
        let idx = step - 1;
        let start = offsets
            .get(idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        let end = offsets
            .get(idx + 1)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        means
            .get(start..end)
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }

    pub fn step(&mut self) -> Result<bool, ScenarioError> {
        if self.step == 0 {
            self.step = 1;
            return Ok(self.step < MAX_LOT10_STEPS);
        }
        if self.step >= MAX_LOT10_STEPS {
            return Ok(false);
        }
        let ts = Self::ts(self.step);
        let mut det_buf = [Detection {
            vector: stonesoup_core::VecN([0.0, 0.0]),
            timestamp: ts,
            sensor: SensorId(0),
        }; MAX_LOT10_DETS];
        let step_idx = self.step - 1;
        let dwell = match self.mode {
            Lot10SensorManagerMode::Random => LOT10_RANDOM_DWELL_RAD
                .get(step_idx)
                .copied()
                .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?,
            Lot10SensorManagerMode::BruteForce => LOT10_BRUTE_DWELL_RAD
                .get(step_idx)
                .copied()
                .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?,
        };
        self.updater.measurement.antenna_heading_rad = dwell;
        let count = push_bearing_range_measurements(
            self.meas_offsets,
            self.meas_br,
            step_idx,
            ts,
            &mut det_buf,
        )?;
        let detections = &det_buf[..count];

        let tails = self.tracks;
        let mut assigns = [GnnAssignment {
            hypothesis: MeasurementHypothesis {
                prediction: tails[0],
                detection_index: None,
                distance: LOT10_MISSED_DISTANCE,
            },
        }; MAX_LOT10_TRACKS];
        associate_gnn_2d_ekf(
            &tails,
            ts,
            detections,
            LOT10_MISSED_DISTANCE,
            &self.predictor,
            &self.updater,
            &mut self.ws,
            &mut assigns,
        )
        .map_err(ScenarioError::Core)?;

        for (t, assign) in assigns.iter().enumerate().take(MAX_LOT10_TRACKS) {
            let h = &assign.hypothesis;
            self.tracks[t] = if let Some(di) = h.detection_index {
                self.updater
                    .update(&h.prediction, &detections[di], &mut self.ws)
                    .map_err(ScenarioError::Core)?
            } else {
                h.prediction
            };
        }

        self.step += 1;
        Ok(self.step < MAX_LOT10_STEPS)
    }

    pub fn run_to_end(&mut self) -> Result<(), ScenarioError> {
        while self.step()? {}
        Ok(())
    }

    pub fn mode(&self) -> Lot10SensorManagerMode {
        self.mode
    }

    pub fn viewer_snapshot(&self) -> Result<ViewerSnapshot, ScenarioError> {
        let step = self.step;
        let idx = if step == 0 { 0 } else { step.saturating_sub(1) };
        let truth = LOT10_TRUTH
            .get(idx)
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        let mut sorted = [[0.0; 4]; MAX_LOT10_TRACKS];
        let n = self.sorted_track_means(&mut sorted);
        let estimate_xy = if n > 0 {
            (sorted[0][0], sorted[0][2])
        } else {
            (0.0, 0.0)
        };
        let estimate_b_xy = if n > 1 {
            Some((sorted[1][0], sorted[1][2]))
        } else {
            None
        };
        let t0 = truth[0];
        let t1 = truth[1];
        let mut det_xy = empty_det_xy();
        let mut det_is_true = empty_det_flags();
        let det_count = if step == 0 {
            0
        } else {
            let start = self
                .meas_offsets
                .get(idx)
                .copied()
                .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
            let end = self
                .meas_offsets
                .get(idx + 1)
                .copied()
                .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
            let count = (end - start).min(MAX_DETS_PER_STEP);
            for j in 0..count {
                let br = self.meas_br[start + j];
                let xy = lot10_bearing_range_to_xy(br[0], br[1]);
                det_xy[j] = [xy.0, xy.1];
                det_is_true[j] = true;
            }
            count
        };
        let (measurement_xy, bearing_range) = if det_count > 0 {
            let br = self.meas_br[self.meas_offsets[idx]];
            let xy = lot10_bearing_range_to_xy(br[0], br[1]);
            (xy, Some((br[0], br[1])))
        } else {
            (estimate_xy, None)
        };
        Ok(ViewerSnapshot {
            step,
            max_steps: MAX_LOT10_STEPS,
            truth_xy: (t0[0], t0[2]),
            estimate_xy,
            measurement_xy,
            bearing_range,
            ellipse_semi: None,
            ellipse_angle_rad: 0.0,
            rmse: 0.0,
            finished: step >= MAX_LOT10_STEPS,
            det_count: det_count as u8,
            det_xy,
            det_is_true,
            associated_xy: None,
            truth_b_xy: Some((t1[0], t1[2])),
            estimate_b_xy,
            resample_case: u8::MAX,
            resample_nparts: 0,
            resample_xy: empty_resample_xy(),
        })
    }
}
