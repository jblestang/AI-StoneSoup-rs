include!("lot3_multitarget_fixtures.rs");

use stonesoup_core::{
    associate_global_nearest_neighbour, AssociationOutcome, ConstantVelocity2d, Detection,
    Duration, GaussianState, GnnTrackAssignment, KalmanPredictor, KalmanUpdater, KalmanWorkspace,
    LinearGaussianMeasurement, Mat, RmseAccumulator, Timestamp, MAX_GNN_TRACKS,
};

use crate::lot1_kalman::tutorial_cov;
use crate::lot3_meas::push_measurements;
use crate::registry::ScenarioError;

pub const MAX_LOT3_MT_STEPS: usize = NUM_STEPS;
pub const LOT3_MT_GATE: f64 = 3.0;

pub struct Lot3MultiTargetScenario {
    predictor: KalmanPredictor,
    updater: KalmanUpdater,
    ws: KalmanWorkspace,
    tracks: [GaussianState; MAX_GNN_TRACKS],
    step: usize,
    rmse: [RmseAccumulator; MAX_GNN_TRACKS],
}

impl Lot3MultiTargetScenario {
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
        let track_a = GaussianState::new_from_array([0.0, 1.0, 0.0, 1.0], cov, Timestamp::ZERO)
            .map_err(ScenarioError::Core)?;
        let track_b = GaussianState::new_from_array([0.0, 1.0, 20.0, -1.0], cov, Timestamp::ZERO)
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
            tracks: [track_a, track_b],
            step: 0,
            rmse: [RmseAccumulator::new(), RmseAccumulator::new()],
        })
    }

    fn ts(step: usize) -> Timestamp {
        Timestamp::from_nanos((step as i64) * Duration::SEC.0)
    }

    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        *self = Self::tutorial_defaults()?;
        Ok(())
    }

    pub fn step(&mut self) -> Result<bool, ScenarioError> {
        if self.step >= NUM_STEPS {
            return Ok(false);
        }
        let ts = Self::ts(self.step);
        let mut det_buf = [Detection {
            vector: stonesoup_core::VecN([0.0, 0.0]),
            timestamp: ts,
            sensor: stonesoup_core::SensorId(0),
        }; MAX_DETS_PER_STEP];
        let count = push_measurements(&MEAS_OFFSETS, &MEAS_XY, self.step, ts, &mut det_buf)?;
        let detections = &det_buf[..count];
        let mut assigns = [GnnTrackAssignment {
            outcome: AssociationOutcome::Missed {
                prediction: self.tracks[0],
            },
        }; MAX_GNN_TRACKS];
        associate_global_nearest_neighbour(
            &self.tracks,
            ts,
            detections,
            LOT3_MT_GATE,
            &self.predictor,
            &self.updater,
            &mut self.ws,
            &mut assigns,
        )
        .map_err(ScenarioError::Core)?;
        for (t, assign) in assigns.iter().enumerate() {
            self.tracks[t] = match assign.outcome {
                AssociationOutcome::Matched {
                    detection_index,
                    prediction,
                } => self
                    .updater
                    .update(&prediction, &detections[detection_index], &mut self.ws)
                    .map_err(ScenarioError::Core)?,
                AssociationOutcome::Missed { prediction } => prediction,
            };
            let truth = if t == 0 {
                TRUTH_A.get(self.step)
            } else {
                TRUTH_B.get(self.step)
            }
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
            let ex = self.tracks[t].mean.0[0] - truth[0];
            let ey = self.tracks[t].mean.0[2] - truth[2];
            self.rmse[t].push(ex, ey).map_err(ScenarioError::Core)?;
        }
        self.step += 1;
        Ok(self.step < NUM_STEPS)
    }

    pub fn run_to_end(&mut self) -> Result<(), ScenarioError> {
        while self.step()? {}
        Ok(())
    }

    pub fn track_state(&self, i: usize) -> Option<&GaussianState> {
        self.tracks.get(i)
    }

    pub fn reference_track_at(&self, track: usize, step: usize) -> Option<[f64; 4]> {
        match track {
            0 => TRACK_A_REF.get(step).copied(),
            1 => TRACK_B_REF.get(step).copied(),
            _ => None,
        }
    }

    pub fn step_index(&self) -> usize {
        self.step
    }

    pub fn rmse_value(&self) -> f64 {
        let a = self.rmse[0].rmse().unwrap_or(0.0);
        let b = self.rmse[1].rmse().unwrap_or(0.0);
        libm::sqrt(0.5 * (a * a + b * b))
    }
}
