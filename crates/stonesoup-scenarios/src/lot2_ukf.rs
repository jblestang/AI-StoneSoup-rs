mod ukf_fixtures {
    include!("lot2_ukf_fixtures.rs");
}

use stonesoup_core::linalg::ellipse_from_cov2;
use stonesoup_core::{
    CartesianToBearingRange, ConstantVelocity2d, Detection, GaussianState, KalmanWorkspace, Mat,
    RmseAccumulator, SensorId, Timestamp, UnscentedKalmanPredictor, UnscentedKalmanUpdater,
};

use crate::lot1_kalman::tutorial_cov;
use crate::lot2_ekf::{
    fixture_row, Lot2EkfScenario, Lot2Snapshot, Lot2StepOutcome, MEASUREMENTS, NUM_STEPS, TRUTH,
};
use crate::registry::ScenarioError;

pub struct Lot2UkfScenario {
    predictor: UnscentedKalmanPredictor,
    updater: UnscentedKalmanUpdater,
    ws: KalmanWorkspace,
    prior: GaussianState,
    step: usize,
    rmse: RmseAccumulator,
    conf: f64,
}

impl Lot2UkfScenario {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn tutorial_defaults() -> Result<Self, ScenarioError> {
        let transition = ConstantVelocity2d::new(0.05, 0.05).map_err(ScenarioError::Core)?;
        let predictor = UnscentedKalmanPredictor::new(transition);
        let measurement = CartesianToBearingRange::tutorial02().map_err(ScenarioError::Core)?;
        let updater = UnscentedKalmanUpdater::new(measurement);
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
        })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        *self = Self::tutorial_defaults()?;
        Ok(())
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn step(&mut self) -> Result<Lot2StepOutcome, ScenarioError> {
        if self.step >= NUM_STEPS {
            return Ok(Lot2StepOutcome::Finished);
        }
        let ts = Lot2EkfScenario::timestamp_for(self.step);
        let predicted = self
            .predictor
            .predict(&self.prior, ts, &mut self.ws)
            .map_err(ScenarioError::Core)?;
        let m = fixture_row(&MEASUREMENTS, self.step)?;
        let det = Detection::new(m, ts, SensorId(0)).map_err(ScenarioError::Core)?;
        let post = self
            .updater
            .update(&predicted, &det, &mut self.ws)
            .map_err(ScenarioError::Core)?;
        let truth = fixture_row(&TRUTH, self.step)?;
        self.rmse
            .push(post.mean.0[0] - truth[0], post.mean.0[2] - truth[2])
            .map_err(ScenarioError::Core)?;
        self.prior = post;
        self.step += 1;
        Ok(if self.step >= NUM_STEPS {
            Lot2StepOutcome::Finished
        } else {
            Lot2StepOutcome::Stepped
        })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn estimate_state(&self) -> &GaussianState {
        &self.prior
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn reference_track_at(&self, i: usize) -> Option<[f64; 4]> {
        ukf_fixtures::UKF_TRACK_REF.get(i).copied()
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn snapshot(&self) -> Result<Lot2Snapshot, ScenarioError> {
        let idx = if self.step == 0 { 0 } else { self.step - 1 };
        let truth = fixture_row(&TRUTH, idx)?;
        let m = fixture_row(&MEASUREMENTS, idx)?;
        let (ex, ey) = self.prior.position_xy().map_err(ScenarioError::Core)?;
        let (c00, c01, c11) = self
            .prior
            .position_cov_block()
            .map_err(ScenarioError::Core)?;
        let ellipse = ellipse_from_cov2(c00, c01, c11, self.conf).map_err(ScenarioError::Core)?;
        let ellipse_angle_rad = libm::atan2(2.0 * c01, c00 - c11) * 0.5;
        Ok(Lot2Snapshot {
            step: self.step,
            truth_xy: (truth[0], truth[2]),
            measurement_br: (m[0], m[1]),
            estimate_xy: (ex, ey),
            ellipse_semi: ellipse,
            ellipse_angle_rad,
            rmse: self.rmse.rmse().unwrap_or(0.0),
            finished: self.step >= NUM_STEPS,
        })
    }
}
