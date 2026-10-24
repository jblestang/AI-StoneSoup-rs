include!("lot1_fixtures.rs");

use stonesoup_core::linalg::ellipse_from_cov2;
use stonesoup_core::{
    ConstantVelocity2d, Detection, Duration, GaussianState, KalmanPredictor, KalmanUpdater,
    KalmanWorkspace, LinearGaussianMeasurement, Mat, RmseAccumulator, SensorId, Timestamp,
};

use crate::registry::ScenarioError;

pub const MAX_LOT1_STEPS: usize = NUM_STEPS;

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn tutorial_cov() -> Result<Mat<4, 4>, ScenarioError> {
    let mut cov = Mat::<4, 4>::zeros();
    cov.set(0, 0, 1.5)?;
    cov.set(1, 1, 0.5)?;
    cov.set(2, 2, 1.5)?;
    cov.set(3, 3, 0.5)?;
    Ok(cov)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn fixture_row<const N: usize>(
    table: &[[f64; N]; NUM_STEPS],
    i: usize,
) -> Result<[f64; N], ScenarioError> {
    table
        .get(i)
        .copied()
        .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
}

#[derive(Debug, Clone, Copy)]
pub struct Lot1Snapshot {
    pub step: usize,
    pub truth_xy: (f64, f64),
    pub measurement_xy: (f64, f64),
    pub estimate_xy: (f64, f64),
    pub ellipse_semi: (f64, f64),
    /// Orientation (rad) of the major axis in the x–y plane.
    pub ellipse_angle_rad: f64,
    pub rmse: f64,
    pub finished: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lot1StepOutcome {
    Stepped,
    Finished,
}

pub struct Lot1KalmanScenario {
    predictor: KalmanPredictor,
    updater: KalmanUpdater,
    ws: KalmanWorkspace,
    prior: GaussianState,
    step: usize,
    rmse: RmseAccumulator,
    conf: f64,
}

impl Lot1KalmanScenario {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn tutorial_defaults() -> Result<Self, ScenarioError> {
        let transition = match ConstantVelocity2d::new(0.05, 0.05) {
            Ok(v) => v,
            Err(e) => return Err(ScenarioError::Core(e)),
        };
        let predictor = KalmanPredictor::new(transition);
        let mut r = Mat::<2, 2>::zeros();
        if r.set(0, 0, 5.0).is_err() || r.set(1, 1, 5.0).is_err() {
            return Err(ScenarioError::Core(stonesoup_core::Error::NumericalFailure));
        }
        let measurement = match LinearGaussianMeasurement::position_2d(r) {
            Ok(v) => v,
            Err(e) => return Err(ScenarioError::Core(e)),
        };
        let updater = KalmanUpdater::new(measurement);
        let cov = match tutorial_cov() {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let prior = match GaussianState::new_from_array([0.0, 1.0, 0.0, 1.0], cov, Timestamp::ZERO)
        {
            Ok(v) => v,
            Err(e) => return Err(ScenarioError::Core(e)),
        };
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
    fn timestamp_for(step: usize) -> Timestamp {
        Timestamp::from_nanos((step as i64) * Duration::SEC.0)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        *self = Self::tutorial_defaults()?;
        Ok(())
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn step(&mut self) -> Result<Lot1StepOutcome, ScenarioError> {
        if self.step >= NUM_STEPS {
            return Ok(Lot1StepOutcome::Finished);
        }
        let ts = Self::timestamp_for(self.step);
        let predicted = self.predictor.predict(&self.prior, ts, &mut self.ws)?;
        let m = fixture_row(&MEASUREMENTS, self.step)?;
        let det = Detection::new(m, ts, SensorId(0))?;
        let post = self.updater.update(&predicted, &det, &mut self.ws)?;
        let truth = fixture_row(&TRUTH, self.step)?;
        let ex = post.mean.0[0] - truth[0];
        let ey = post.mean.0[2] - truth[2];
        self.rmse.push(ex, ey)?;
        self.prior = post;
        self.step = self
            .step
            .checked_add(1)
            .ok_or(ScenarioError::Core(stonesoup_core::Error::CapacityExceeded))?;
        if self.step >= NUM_STEPS {
            Ok(Lot1StepOutcome::Finished)
        } else {
            Ok(Lot1StepOutcome::Stepped)
        }
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn run_to_end(&mut self) -> Result<(), ScenarioError> {
        while self.step < NUM_STEPS {
            match self.step()? {
                Lot1StepOutcome::Stepped => {}
                Lot1StepOutcome::Finished => break,
            }
        }
        Ok(())
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn snapshot(&self) -> Result<Lot1Snapshot, ScenarioError> {
        let idx = if self.step == 0 { 0 } else { self.step - 1 };
        let truth = fixture_row(&TRUTH, idx)?;
        let m = fixture_row(&MEASUREMENTS, idx)?;
        let (ex, ey) = self.prior.position_xy()?;
        let (c00, c01, c11) = self.prior.position_cov_block()?;
        let ellipse = ellipse_from_cov2(c00, c01, c11, self.conf)?;
        let ellipse_angle_rad = libm::atan2(2.0 * c01, c00 - c11) * 0.5;
        let rmse = self.rmse.rmse().unwrap_or(0.0);
        Ok(Lot1Snapshot {
            step: self.step,
            truth_xy: (truth[0], truth[2]),
            measurement_xy: (m[0], m[1]),
            estimate_xy: (ex, ey),
            ellipse_semi: ellipse,
            ellipse_angle_rad,
            rmse,
            finished: self.step >= NUM_STEPS,
        })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn estimate_state(&self) -> &GaussianState {
        &self.prior
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn step_index(&self) -> usize {
        self.step
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn reference_track_at(&self, i: usize) -> Option<[f64; 4]> {
        TRACK_REF.get(i).copied()
    }
}
