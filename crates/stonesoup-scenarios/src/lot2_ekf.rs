include!("lot2_fixtures.rs");

use stonesoup_core::linalg::ellipse_from_cov2;
use stonesoup_core::{
    CartesianToBearingRange, ConstantVelocity2d, Detection, Duration, ExtendedKalmanUpdater,
    GaussianState, KalmanPredictor, KalmanWorkspace, Mat, RmseAccumulator, SensorId, Timestamp,
};

use crate::lot1_kalman::tutorial_cov;
use crate::registry::ScenarioError;

pub const MAX_LOT2_STEPS: usize = NUM_STEPS;

/// Tutorial 02 sensor platform (bearing–range).
pub const TUTORIAL02_SENSOR_XY: (f64, f64) = (50.0, 0.0);

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn bearing_range_to_xy(bearing: f64, range: f64) -> (f64, f64) {
    let (sx, sy) = TUTORIAL02_SENSOR_XY;
    (
        sx + range * libm::cos(bearing),
        sy + range * libm::sin(bearing),
    )
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub(crate) fn fixture_row<const N: usize>(
    table: &[[f64; N]; NUM_STEPS],
    i: usize,
) -> Result<[f64; N], ScenarioError> {
    table
        .get(i)
        .copied()
        .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
}

#[derive(Debug, Clone, Copy)]
pub struct Lot2Snapshot {
    pub step: usize,
    pub truth_xy: (f64, f64),
    pub measurement_br: (f64, f64),
    pub estimate_xy: (f64, f64),
    pub ellipse_semi: (f64, f64),
    pub ellipse_angle_rad: f64,
    pub rmse: f64,
    pub finished: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lot2StepOutcome {
    Stepped,
    Finished,
}

pub struct Lot2EkfScenario {
    predictor: KalmanPredictor,
    updater: ExtendedKalmanUpdater,
    ws: KalmanWorkspace,
    prior: GaussianState,
    step: usize,
    rmse: RmseAccumulator,
    conf: f64,
}

impl Lot2EkfScenario {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn tutorial_defaults() -> Result<Self, ScenarioError> {
        let transition = ConstantVelocity2d::new(0.05, 0.05).map_err(ScenarioError::Core)?;
        let predictor = KalmanPredictor::new(transition);
        let measurement = CartesianToBearingRange::tutorial02().map_err(ScenarioError::Core)?;
        let updater = ExtendedKalmanUpdater::new(measurement);
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
    pub(crate) fn timestamp_for(step: usize) -> Timestamp {
        Timestamp::from_nanos((step as i64) * Duration::SEC.0)
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
            Ok(Lot2StepOutcome::Finished)
        } else {
            Ok(Lot2StepOutcome::Stepped)
        }
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn run_to_end(&mut self) -> Result<(), ScenarioError> {
        while self.step < NUM_STEPS {
            match self.step()? {
                Lot2StepOutcome::Stepped => {}
                Lot2StepOutcome::Finished => break,
            }
        }
        Ok(())
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn snapshot(&self) -> Result<Lot2Snapshot, ScenarioError> {
        let idx = if self.step == 0 { 0 } else { self.step - 1 };
        let truth = fixture_row(&TRUTH, idx)?;
        let m = fixture_row(&MEASUREMENTS, idx)?;
        let (ex, ey) = self.prior.position_xy()?;
        let (c00, c01, c11) = self.prior.position_cov_block()?;
        let ellipse = ellipse_from_cov2(c00, c01, c11, self.conf)?;
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

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn reference_track_at(&self, i: usize) -> Option<[f64; 4]> {
        TRACK_REF.get(i).copied()
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn estimate_state(&self) -> &GaussianState {
        &self.prior
    }
}
