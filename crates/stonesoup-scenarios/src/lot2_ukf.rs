mod ukf_fixtures {
    include!("lot2_ukf_fixtures.rs");
}

use stonesoup_core::{
    CartesianToBearingRange, ConstantVelocity2d, Detection, GaussianState, KalmanWorkspace, Mat,
    RmseAccumulator, SensorId, Timestamp, UnscentedKalmanPredictor, UnscentedKalmanUpdater,
};

use crate::lot1_kalman::tutorial_cov;
use crate::lot2_ekf::{
    fixture_row, Lot2EkfScenario, Lot2StepOutcome, MEASUREMENTS, NUM_STEPS, TRUTH,
};
use crate::registry::ScenarioError;

pub struct Lot2UkfScenario {
    predictor: UnscentedKalmanPredictor,
    updater: UnscentedKalmanUpdater,
    ws: KalmanWorkspace,
    prior: GaussianState,
    step: usize,
    rmse: RmseAccumulator,
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
        })
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
}
