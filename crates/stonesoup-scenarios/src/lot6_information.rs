include!("lot6_if_fixtures.rs");

use stonesoup_core::linalg::ellipse_from_cov2;
use stonesoup_core::{
    ConstantVelocity2d, Detection, Duration, GaussianState, InformationKalmanPredictor,
    InformationKalmanUpdater, InformationKalmanWorkspace, InformationState,
    LinearGaussianMeasurement, Mat, RmseAccumulator, SensorId, Timestamp,
};

use crate::registry::ScenarioError;

pub const MAX_LOT6_STEPS: usize = LOT6_NUM_STEPS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lot6StepOutcome {
    Stepped,
    Finished,
}

#[derive(Debug, Clone, Copy)]
pub struct Lot6Snapshot {
    pub step: usize,
    pub truth_xy: (f64, f64),
    pub measurement_xy: (f64, f64),
    pub estimate_xy: (f64, f64),
    pub ellipse_semi: (f64, f64),
    pub ellipse_angle_rad: f64,
    pub rmse: f64,
    pub finished: bool,
}

pub struct Lot6InformationScenario {
    predictor: InformationKalmanPredictor,
    updater: InformationKalmanUpdater,
    ws: InformationKalmanWorkspace,
    prior: InformationState,
    estimate: GaussianState,
    step: usize,
    rmse: RmseAccumulator,
    conf: f64,
}

impl Lot6InformationScenario {
    pub fn tutorial_defaults() -> Result<Self, ScenarioError> {
        let transition = ConstantVelocity2d::new(0.05, 0.05).map_err(ScenarioError::Core)?;
        let predictor = InformationKalmanPredictor::new(transition);
        let mut r = Mat::<2, 2>::zeros();
        r.set(0, 0, 5.0).map_err(ScenarioError::Core)?;
        r.set(1, 1, 5.0).map_err(ScenarioError::Core)?;
        let measurement = LinearGaussianMeasurement::position_2d(r).map_err(ScenarioError::Core)?;
        let updater = InformationKalmanUpdater::new(measurement);
        let mut cov = Mat::<4, 4>::zeros();
        for (i, &v) in LOT6_PRIOR_COV_DIAG.iter().enumerate() {
            cov.set(i, i, v).map_err(ScenarioError::Core)?;
        }
        let gaussian = GaussianState::new_from_array(LOT6_PRIOR_MEAN, cov, Timestamp::ZERO)
            .map_err(ScenarioError::Core)?;
        let prior = InformationState::from_gaussian(&gaussian).map_err(ScenarioError::Core)?;
        Ok(Self {
            predictor,
            updater,
            ws: InformationKalmanWorkspace {
                f: Mat::zeros(),
                q: Mat::zeros(),
                h: Mat::zeros(),
                r: Mat::zeros(),
            },
            prior,
            estimate: gaussian,
            step: 0,
            rmse: RmseAccumulator::new(),
            conf: 2.0,
        })
    }

    fn timestamp_for(step: usize) -> Timestamp {
        Timestamp::from_nanos((step as i64) * Duration::SEC.0)
    }

    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        *self = Self::tutorial_defaults()?;
        Ok(())
    }

    pub fn step(&mut self) -> Result<Lot6StepOutcome, ScenarioError> {
        if self.step >= LOT6_NUM_STEPS {
            return Ok(Lot6StepOutcome::Finished);
        }
        let ts = Self::timestamp_for(self.step);
        let predicted = self
            .predictor
            .predict(&self.prior, ts, &mut self.ws)
            .map_err(ScenarioError::Core)?;
        let m = LOT6_MEASUREMENTS[self.step];
        let det = Detection::new(m, ts, SensorId(0)).map_err(ScenarioError::Core)?;
        let post = self
            .updater
            .update(&predicted, &det)
            .map_err(ScenarioError::Core)?;
        self.estimate = post.to_gaussian().map_err(ScenarioError::Core)?;
        let truth = LOT6_TRUTH[self.step];
        let ex = self.estimate.mean.0[0] - truth[0];
        let ey = self.estimate.mean.0[2] - truth[2];
        self.rmse.push(ex, ey).map_err(ScenarioError::Core)?;
        self.prior = post;
        self.step += 1;
        if self.step >= LOT6_NUM_STEPS {
            Ok(Lot6StepOutcome::Finished)
        } else {
            Ok(Lot6StepOutcome::Stepped)
        }
    }

    pub fn snapshot(&self) -> Result<Lot6Snapshot, ScenarioError> {
        let idx = if self.step == 0 { 0 } else { self.step - 1 };
        let truth = LOT6_TRUTH[idx];
        let m = LOT6_MEASUREMENTS[idx];
        let (ex, ey) = self.estimate.position_xy().map_err(ScenarioError::Core)?;
        let (c00, c01, c11) = self
            .estimate
            .position_cov_block()
            .map_err(ScenarioError::Core)?;
        let ellipse = ellipse_from_cov2(c00, c01, c11, self.conf).map_err(ScenarioError::Core)?;
        let ellipse_angle_rad = libm::atan2(2.0 * c01, c00 - c11) * 0.5;
        Ok(Lot6Snapshot {
            step: self.step,
            truth_xy: (truth[0], truth[2]),
            measurement_xy: (m[0], m[1]),
            estimate_xy: (ex, ey),
            ellipse_semi: ellipse,
            ellipse_angle_rad,
            rmse: self.rmse.rmse().unwrap_or(0.0),
            finished: self.step >= LOT6_NUM_STEPS,
        })
    }

    pub fn reference_track_at(step: usize) -> Option<[f64; 4]> {
        LOT6_TRACK.get(step).copied()
    }

    pub fn estimate_state(&self) -> &GaussianState {
        &self.estimate
    }
}
