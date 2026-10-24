include!("lot3_pda_fixtures.rs");

use stonesoup_core::{pda_hypotheses, pda_merged_update};
use stonesoup_core::{
    ConstantVelocity2d, Detection, Duration, GaussianState, KalmanPredictor, KalmanUpdater,
    KalmanWorkspace, LinearGaussianMeasurement, Mat, ProbabilityHypothesis, RmseAccumulator,
    SensorId, Timestamp, MAX_PDA_HYPOTHESES,
};

use crate::lot1_kalman::tutorial_cov;
use crate::lot3_meas::push_measurements;
use crate::registry::ScenarioError;

pub const MAX_LOT3_PDA_STEPS: usize = NUM_STEPS;
pub const LOT3_PDA_PROB_DETECT: f64 = 0.9;
pub const LOT3_PDA_CLUTTER_DENSITY: f64 = 0.125;

pub struct Lot3PdaScenario {
    predictor: KalmanPredictor,
    updater: KalmanUpdater,
    ws: KalmanWorkspace,
    state: GaussianState,
    step: usize,
    rmse: RmseAccumulator,
}

impl Lot3PdaScenario {
    pub fn tutorial_defaults() -> Result<Self, ScenarioError> {
        let transition = ConstantVelocity2d::new(0.005, 0.005).map_err(ScenarioError::Core)?;
        let predictor = KalmanPredictor::new(transition);
        let mut r = Mat::<2, 2>::zeros();
        r.set(0, 0, 0.75).map_err(ScenarioError::Core)?;
        r.set(1, 1, 0.75).map_err(ScenarioError::Core)?;
        let measurement = LinearGaussianMeasurement::position_2d(r).map_err(ScenarioError::Core)?;
        let updater = KalmanUpdater::new(measurement);
        let cov = tutorial_cov()?;
        let state = GaussianState::new_from_array([0.0, 1.0, 0.0, 1.0], cov, Timestamp::ZERO)
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
            state,
            step: 0,
            rmse: RmseAccumulator::new(),
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
            sensor: SensorId(0),
        }; MAX_DETS_PER_STEP];
        let count = push_measurements(&MEAS_OFFSETS, &MEAS_XY, self.step, ts, &mut det_buf)?;
        let detections = &det_buf[..count];
        let mut hyps = [ProbabilityHypothesis {
            prediction: self.state,
            detection_index: None,
            probability: 0.0,
        }; MAX_PDA_HYPOTHESES];
        let n = pda_hypotheses(
            &self.state,
            ts,
            detections,
            LOT3_PDA_PROB_DETECT,
            LOT3_PDA_CLUTTER_DENSITY,
            &self.predictor,
            &self.updater,
            &mut self.ws,
            &mut hyps,
        )
        .map_err(ScenarioError::Core)?;
        self.state = pda_merged_update(&hyps, n, detections, &self.updater, &mut self.ws)
            .map_err(ScenarioError::Core)?;
        let truth = TRUTH
            .get(self.step)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        let ex = self.state.mean.0[0] - truth[0];
        let ey = self.state.mean.0[2] - truth[2];
        self.rmse.push(ex, ey).map_err(ScenarioError::Core)?;
        self.step += 1;
        Ok(self.step < NUM_STEPS)
    }

    pub fn run_to_end(&mut self) -> Result<(), ScenarioError> {
        while self.step()? {}
        Ok(())
    }

    pub fn estimate_state(&self) -> &GaussianState {
        &self.state
    }

    pub fn reference_track_at(&self, i: usize) -> Option<[f64; 4]> {
        TRACK_REF.get(i).copied()
    }

    pub fn step_index(&self) -> usize {
        self.step
    }
}
