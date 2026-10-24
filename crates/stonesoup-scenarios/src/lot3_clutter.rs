include!("lot3_fixtures.rs");

use stonesoup_core::linalg::ellipse_from_cov2;
use stonesoup_core::{
    associate_nearest_neighbour, AssociationOutcome, ConstantVelocity2d, Detection, Duration,
    GaussianState, KalmanPredictor, KalmanUpdater, KalmanWorkspace, LinearGaussianMeasurement, Mat,
    RmseAccumulator, SensorId, Timestamp,
};

use crate::lot1_kalman::tutorial_cov;
use crate::registry::ScenarioError;

pub const MAX_LOT3_STEPS: usize = NUM_STEPS;
pub const LOT3_GATE: f64 = 3.0;

#[derive(Debug, Clone, Copy)]
pub struct Lot3MeasPoint {
    pub xy: (f64, f64),
    pub is_true: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Lot3Snapshot {
    pub step: usize,
    pub truth_xy: (f64, f64),
    pub estimate_xy: (f64, f64),
    pub measurements: [Lot3MeasPoint; MAX_DETS_PER_STEP],
    pub measurement_count: usize,
    pub associated_xy: Option<(f64, f64)>,
    pub ellipse_semi: (f64, f64),
    pub ellipse_angle_rad: f64,
    pub rmse: f64,
    pub finished: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lot3StepOutcome {
    Stepped,
    Finished,
}

pub struct Lot3ClutterScenario {
    predictor: KalmanPredictor,
    updater: KalmanUpdater,
    ws: KalmanWorkspace,
    prior: GaussianState,
    step: usize,
    rmse: RmseAccumulator,
    conf: f64,
    last_associated: Option<(f64, f64)>,
}

impl Lot3ClutterScenario {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn tutorial_defaults() -> Result<Self, ScenarioError> {
        let transition = ConstantVelocity2d::new(0.005, 0.005).map_err(ScenarioError::Core)?;
        let predictor = KalmanPredictor::new(transition);
        let mut r = Mat::<2, 2>::zeros();
        r.set(0, 0, 0.75).map_err(ScenarioError::Core)?;
        r.set(1, 1, 0.75).map_err(ScenarioError::Core)?;
        let measurement = LinearGaussianMeasurement::position_2d(r).map_err(ScenarioError::Core)?;
        let updater = KalmanUpdater::new(measurement);
        let cov = tutorial_cov()?;
        let prior = GaussianState::new_from_array([0.0, 1.0, 0.0, 1.0], cov, Timestamp::ZERO)
            .map_err(ScenarioError::Core)?;
        Ok(Self {
            predictor,
            updater,
            ws: KalmanWorkspace {
                f: Mat::zeros(),
                q: Mat::zeros(),
                h: Mat::zeros(),
                r: Mat::zeros(),
            },
            prior,
            step: 0,
            rmse: RmseAccumulator::new(),
            conf: 2.0,
            last_associated: None,
        })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    fn timestamp_for(step: usize) -> Timestamp {
        Timestamp::from_nanos((step as i64) * Duration::SEC.0)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    fn push_measurements_for_step(
        step: usize,
        out: &mut [Detection; MAX_DETS_PER_STEP],
    ) -> Result<usize, ScenarioError> {
        let start = MEAS_OFFSETS
            .get(step)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        let end = MEAS_OFFSETS
            .get(step + 1)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        let count = end - start;
        if count > MAX_DETS_PER_STEP {
            return Err(ScenarioError::Core(stonesoup_core::Error::CapacityExceeded));
        }
        let ts = Self::timestamp_for(step);
        for (j, idx) in (start..end).enumerate() {
            let xy = MEAS_XY
                .get(idx)
                .copied()
                .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
            out[j] = Detection::new(xy, ts, SensorId(0)).map_err(ScenarioError::Core)?;
        }
        Ok(count)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        *self = Self::tutorial_defaults()?;
        Ok(())
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn step(&mut self) -> Result<Lot3StepOutcome, ScenarioError> {
        if self.step >= NUM_STEPS {
            return Ok(Lot3StepOutcome::Finished);
        }
        let assoc_time = Self::timestamp_for(self.step);
        let mut det_buf = [Detection {
            vector: stonesoup_core::VecN([0.0, 0.0]),
            timestamp: assoc_time,
            sensor: SensorId(0),
        }; MAX_DETS_PER_STEP];
        let count = Self::push_measurements_for_step(self.step, &mut det_buf)?;
        let detections = &det_buf[..count];
        let outcome = associate_nearest_neighbour(
            &self.prior,
            assoc_time,
            detections,
            LOT3_GATE,
            &self.predictor,
            &self.updater,
            &mut self.ws,
        )
        .map_err(ScenarioError::Core)?;

        self.last_associated = None;
        match outcome {
            AssociationOutcome::Matched {
                detection_index,
                prediction,
            } => {
                let det = &detections[detection_index];
                self.last_associated = Some((det.vector.0[0], det.vector.0[1]));
                let post = self
                    .updater
                    .update(&prediction, det, &mut self.ws)
                    .map_err(ScenarioError::Core)?;
                self.prior = post;
            }
            AssociationOutcome::Missed { prediction } => {
                self.prior = prediction;
            }
        }

        let truth = TRUTH
            .get(self.step)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        let ex = self.prior.mean.0[0] - truth[0];
        let ey = self.prior.mean.0[2] - truth[2];
        self.rmse.push(ex, ey).map_err(ScenarioError::Core)?;

        self.step = self
            .step
            .checked_add(1)
            .ok_or(ScenarioError::Core(stonesoup_core::Error::CapacityExceeded))?;
        if self.step >= NUM_STEPS {
            Ok(Lot3StepOutcome::Finished)
        } else {
            Ok(Lot3StepOutcome::Stepped)
        }
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn run_to_end(&mut self) -> Result<(), ScenarioError> {
        while self.step < NUM_STEPS {
            match self.step()? {
                Lot3StepOutcome::Stepped => {}
                Lot3StepOutcome::Finished => break,
            }
        }
        Ok(())
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn snapshot(&self) -> Result<Lot3Snapshot, ScenarioError> {
        let idx = if self.step == 0 { 0 } else { self.step - 1 };
        let truth = TRUTH
            .get(idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        let start = MEAS_OFFSETS
            .get(idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        let end = MEAS_OFFSETS
            .get(idx + 1)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        let mut measurements = [Lot3MeasPoint {
            xy: (0.0, 0.0),
            is_true: false,
        }; MAX_DETS_PER_STEP];
        let mut measurement_count = 0usize;
        for (i, m_idx) in (start..end).enumerate() {
            if i >= MAX_DETS_PER_STEP {
                break;
            }
            let xy = MEAS_XY
                .get(m_idx)
                .copied()
                .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
            let is_true = MEAS_IS_TRUE.get(m_idx).copied().unwrap_or(0) != 0;
            measurements[i] = Lot3MeasPoint {
                xy: (xy[0], xy[1]),
                is_true,
            };
            measurement_count += 1;
        }
        let (ex, ey) = self.prior.position_xy().map_err(ScenarioError::Core)?;
        let (c00, c01, c11) = self
            .prior
            .position_cov_block()
            .map_err(ScenarioError::Core)?;
        let ellipse = ellipse_from_cov2(c00, c01, c11, self.conf).map_err(ScenarioError::Core)?;
        let ellipse_angle_rad = libm::atan2(2.0 * c01, c00 - c11) * 0.5;
        Ok(Lot3Snapshot {
            step: self.step,
            truth_xy: (truth[0], truth[2]),
            estimate_xy: (ex, ey),
            measurements,
            measurement_count,
            associated_xy: self.last_associated,
            ellipse_semi: ellipse,
            ellipse_angle_rad,
            rmse: self.rmse.rmse().unwrap_or(0.0),
            finished: self.step >= NUM_STEPS,
        })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn estimate_state(&self) -> &GaussianState {
        &self.prior
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn step_index(&self) -> usize {
        self.step
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn reference_track_at(&self, i: usize) -> Option<[f64; 4]> {
        TRACK_REF.get(i).copied()
    }
}
