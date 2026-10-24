include!("lot8_asd_fixtures.rs");

use stonesoup_core::linalg::ellipse_from_cov2;
use stonesoup_core::{
    AsdGaussianState, AsdKalmanPredictor, AsdKalmanUpdater, ConstantVelocity2d, Detection,
    Duration, KalmanWorkspace, LinearGaussianMeasurement, Mat, RmseAccumulator, SensorId,
    Timestamp, VecN,
};

use crate::registry::ScenarioError;

pub const MAX_LOT8_STEPS: usize = LOT8_NUM_UPDATES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lot8StepOutcome {
    Stepped,
    Finished,
}

#[derive(Debug, Clone, Copy)]
pub struct Lot8Snapshot {
    pub step: usize,
    pub truth_xy: (f64, f64),
    pub measurement_xy: (f64, f64),
    pub estimate_xy: (f64, f64),
    pub ellipse_semi: (f64, f64),
    pub ellipse_angle_rad: f64,
    pub rmse: f64,
    pub finished: bool,
}

pub struct Lot8AsdScenario {
    predictor: AsdKalmanPredictor,
    updater: AsdKalmanUpdater,
    ws: KalmanWorkspace,
    prior: AsdGaussianState,
    act_state_mean: [f64; 4],
    act_state_cov: Mat<4, 4>,
    update_step: usize,
    rmse: RmseAccumulator,
    conf: f64,
}

impl Lot8AsdScenario {
    pub fn tutorial_defaults() -> Result<Self, ScenarioError> {
        let transition = ConstantVelocity2d::new(LOT8_Q, LOT8_Q).map_err(ScenarioError::Core)?;
        let predictor = AsdKalmanPredictor::new(transition);
        let mut r = Mat::<2, 2>::zeros();
        r.set(0, 0, LOT8_MEASUREMENT_NOISE_DIAG[0])
            .map_err(ScenarioError::Core)?;
        r.set(1, 1, LOT8_MEASUREMENT_NOISE_DIAG[1])
            .map_err(ScenarioError::Core)?;
        let measurement = LinearGaussianMeasurement::position_2d(r).map_err(ScenarioError::Core)?;
        let updater = AsdKalmanUpdater::new(measurement);
        let mut cov = Mat::<4, 4>::zeros();
        for (i, &v) in LOT8_PRIOR_COV_DIAG.iter().enumerate() {
            cov.set(i, i, v).map_err(ScenarioError::Core)?;
        }
        let prior = AsdGaussianState::from_single(
            VecN(LOT8_PRIOR_MEAN),
            cov,
            Timestamp::ZERO,
            LOT8_MAX_NSTEP,
        )
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
            act_state_mean: LOT8_PRIOR_MEAN,
            act_state_cov: cov,
            update_step: 0,
            rmse: RmseAccumulator::new(),
            conf: 2.0,
        })
    }

    fn timestamp_for_measurement_index(idx: usize) -> Timestamp {
        let sec = LOT8_MEASUREMENT_TIMES_SEC[idx];
        Timestamp::from_nanos(sec * Duration::SEC.0)
    }

    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        *self = Self::tutorial_defaults()?;
        Ok(())
    }

    pub fn step(&mut self) -> Result<Lot8StepOutcome, ScenarioError> {
        if self.update_step >= LOT8_NUM_UPDATES {
            return Ok(Lot8StepOutcome::Finished);
        }
        let meas_idx = LOT8_UPDATE_MEAS_INDEX[self.update_step];
        let ts = Self::timestamp_for_measurement_index(meas_idx);
        let m = LOT8_MEASUREMENTS[meas_idx];
        let det = Detection::new(m, ts, SensorId(0)).map_err(ScenarioError::Core)?;
        let prediction = self
            .predictor
            .predict(&self.prior, ts, &mut self.ws)
            .map_err(ScenarioError::Core)?;
        let post = self
            .updater
            .update(&prediction, &det, &mut self.ws)
            .map_err(ScenarioError::Core)?;
        let t_index = post.index_of_timestamp(ts).map_err(ScenarioError::Core)?;
        let act = post.state_at(t_index).map_err(ScenarioError::Core)?;
        self.act_state_mean = act.mean.0;
        self.act_state_cov = act.cov;
        let truth = LOT8_TRUTH_XY[meas_idx];
        let ex = self.act_state_mean[0] - truth[0];
        let ey = self.act_state_mean[2] - truth[1];
        self.rmse.push(ex, ey).map_err(ScenarioError::Core)?;
        self.prior = post;
        self.update_step += 1;
        if self.update_step >= LOT8_NUM_UPDATES {
            Ok(Lot8StepOutcome::Finished)
        } else {
            Ok(Lot8StepOutcome::Stepped)
        }
    }

    pub fn snapshot(&self) -> Result<Lot8Snapshot, ScenarioError> {
        let idx = if self.update_step == 0 {
            0
        } else {
            LOT8_UPDATE_MEAS_INDEX[self.update_step - 1]
        };
        let truth = LOT8_TRUTH_XY[idx];
        let m = LOT8_MEASUREMENTS[idx];
        let (ex, ey) = (self.act_state_mean[0], self.act_state_mean[2]);
        let (c00, c01, c11) = (
            self.act_state_cov.0[0][0],
            self.act_state_cov.0[0][2],
            self.act_state_cov.0[2][2],
        );
        let ellipse = ellipse_from_cov2(c00, c01, c11, self.conf).map_err(ScenarioError::Core)?;
        let ellipse_angle_rad = libm::atan2(2.0 * c01, c00 - c11) * 0.5;
        Ok(Lot8Snapshot {
            step: self.update_step,
            truth_xy: (truth[0], truth[1]),
            measurement_xy: (m[0], m[1]),
            estimate_xy: (ex, ey),
            ellipse_semi: ellipse,
            ellipse_angle_rad,
            rmse: self.rmse.rmse().unwrap_or(0.0),
            finished: self.update_step >= LOT8_NUM_UPDATES,
        })
    }

    pub fn reference_act_mean_at(update_step: usize) -> Option<[f64; 4]> {
        LOT8_UPDATE_ACT_MEAN.get(update_step).copied()
    }

    pub fn act_mean(&self) -> [f64; 4] {
        self.act_state_mean
    }

}
