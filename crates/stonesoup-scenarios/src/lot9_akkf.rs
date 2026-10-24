include!("lot9_akkf_fixtures.rs");

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use stonesoup_core::linalg::ellipse_from_cov2;
use stonesoup_core::{
    transition_particles, AdaptiveKernelKalmanPredictor, AdaptiveKernelKalmanUpdater,
    ConstantVelocity2d, Detection, Duration, KernelParticleState, Mat, RmseAccumulator, SensorId,
    Timestamp,
};
use stonesoup_sim::{initialise_lot9_filter_rng, multivariate_normal4_batch, NumpyRng};

use crate::registry::ScenarioError;

pub const MAX_LOT9_STEPS: usize = LOT9_NUM_STEPS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lot9StepOutcome {
    Stepped,
    Finished,
}

#[derive(Debug, Clone, Copy)]
pub struct Lot9Snapshot {
    pub step: usize,
    pub truth_xy: (f64, f64),
    pub estimate_xy: (f64, f64),
    pub bearing_rad: f64,
    pub ellipse_semi: (f64, f64),
    pub ellipse_angle_rad: f64,
    pub rmse: f64,
    pub finished: bool,
}

pub struct Lot9AkkfScenario {
    predictor: AdaptiveKernelKalmanPredictor,
    updater: AdaptiveKernelKalmanUpdater,
    transition: ConstantVelocity2d,
    prior: KernelParticleState,
    rng: NumpyRng,
    step: usize,
    rmse: RmseAccumulator,
    conf: f64,
}

impl Lot9AkkfScenario {
    pub fn tutorial_defaults() -> Result<Self, ScenarioError> {
        let transition = ConstantVelocity2d::new(LOT9_Q, LOT9_Q).map_err(ScenarioError::Core)?;
        let predictor =
            AdaptiveKernelKalmanPredictor::tutorial_akkf().map_err(ScenarioError::Core)?;
        let updater = AdaptiveKernelKalmanUpdater::tutorial_akkf().map_err(ScenarioError::Core)?;
        let (rng, particles) = initialise_lot9_filter_rng().map_err(ScenarioError::Core)?;
        let prior =
            KernelParticleState::new_uniform(&array_from_particles(&particles), Timestamp::ZERO)
                .map_err(ScenarioError::Core)?;
        Ok(Self {
            predictor,
            updater,
            transition,
            prior,
            rng,
            step: 0,
            rmse: RmseAccumulator::new(),
            conf: 2.0,
        })
    }

    fn timestamp_for(step: usize) -> Timestamp {
        Timestamp::from_nanos((step as i64) * Duration::SEC.0)
    }

    fn proposal_cols(prior: &KernelParticleState) -> Result<Vec<f64>, ScenarioError> {
        if let Some(ref p) = prior.proposal_cols {
            return Ok(p.clone());
        }
        Ok(prior.state_cols.clone())
    }

    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        *self = Self::tutorial_defaults()?;
        Ok(())
    }

    pub fn step(&mut self) -> Result<Lot9StepOutcome, ScenarioError> {
        if self.step >= LOT9_NUM_STEPS {
            return Ok(Lot9StepOutcome::Finished);
        }
        let ts = Self::timestamp_for(self.step);
        let dt = ts
            .checked_sub(self.prior.timestamp)
            .map_err(|_| ScenarioError::Core(stonesoup_core::Error::InvalidTimeStep))?;
        let n = LOT9_NUM_PARTICLES;
        let proposal = Self::proposal_cols(&self.prior)?;
        // Stone Soup `predict` applies `F @ proposal` with `noise=False`.
        let zero_noise = vec![[0.0; 4]; n];
        let new_state = transition_particles(&self.transition, dt, &proposal, n, &zero_noise)
            .map_err(ScenarioError::Core)?;
        let predicted = self
            .predictor
            .predict(&self.prior, ts, &proposal, &new_state)
            .map_err(ScenarioError::Core)?;

        let bearing = LOT9_BEARINGS[self.step];
        let det = Detection::new([bearing, 0.0], ts, SensorId(0)).map_err(ScenarioError::Core)?;

        let (updated_weights, updated_covariance) = self
            .updater
            .measure_update(&predicted, &det)
            .map_err(ScenarioError::Core)?;
        let (prop_mean, prop_cov) = AdaptiveKernelKalmanUpdater::proposal_distribution(
            &predicted,
            &updated_weights,
            &updated_covariance,
        )
        .map_err(ScenarioError::Core)?;
        let proposals = multivariate_normal4_batch(&mut self.rng, prop_mean, &prop_cov, n)
            .map_err(ScenarioError::Core)?;
        let mut proposal_cols = vec![0.0; 4 * n];
        for i in 0..n {
            for d in 0..4 {
                proposal_cols[d * n + i] = proposals[i][d];
            }
        }

        let post = KernelParticleState::from_parts(
            predicted.state_cols.clone(),
            updated_weights,
            updated_covariance,
            Some(proposal_cols),
            ts,
        )
        .map_err(ScenarioError::Core)?;

        let mean = post.weighted_mean().map_err(ScenarioError::Core)?;
        let truth = LOT9_TRUTH_XY[self.step];
        let ex = mean[0] - truth[0];
        let ey = mean[2] - truth[1];
        self.rmse.push(ex, ey).map_err(ScenarioError::Core)?;

        self.prior = post;
        self.step += 1;
        if self.step >= LOT9_NUM_STEPS {
            Ok(Lot9StepOutcome::Finished)
        } else {
            Ok(Lot9StepOutcome::Stepped)
        }
    }

    pub fn snapshot(&self) -> Result<Lot9Snapshot, ScenarioError> {
        let step = self.step.min(LOT9_NUM_STEPS);
        let idx = if step == 0 { 0 } else { step - 1 };
        let mean = self.prior.weighted_mean().map_err(ScenarioError::Core)?;
        let truth = LOT9_TRUTH_XY[idx];
        let bearing = LOT9_BEARINGS[idx];
        let (semi, angle) = Self::position_ellipse(&self.prior, self.conf)?;
        Ok(Lot9Snapshot {
            step,
            truth_xy: (truth[0], truth[1]),
            estimate_xy: (mean[0], mean[2]),
            bearing_rad: bearing,
            ellipse_semi: semi,
            ellipse_angle_rad: angle,
            rmse: self.rmse.rmse().unwrap_or(0.0),
            finished: step >= LOT9_NUM_STEPS,
        })
    }

    fn position_ellipse(
        prior: &KernelParticleState,
        conf: f64,
    ) -> Result<((f64, f64), f64), ScenarioError> {
        let mean = prior.weighted_mean().map_err(ScenarioError::Core)?;
        let n = prior.n_particles;
        let mut cov = Mat::<2, 2>::zeros();
        for i in 0..n {
            let dx = prior.state_cols[i] - mean[0];
            let dy = prior.state_cols[2 * n + i] - mean[2];
            let w = prior.weights[i];
            cov.set(
                0,
                0,
                cov.get(0, 0).map_err(ScenarioError::Core)? + w * dx * dx,
            )
            .map_err(ScenarioError::Core)?;
            cov.set(
                0,
                1,
                cov.get(0, 1).map_err(ScenarioError::Core)? + w * dx * dy,
            )
            .map_err(ScenarioError::Core)?;
            cov.set(
                1,
                0,
                cov.get(1, 0).map_err(ScenarioError::Core)? + w * dy * dx,
            )
            .map_err(ScenarioError::Core)?;
            cov.set(
                1,
                1,
                cov.get(1, 1).map_err(ScenarioError::Core)? + w * dy * dy,
            )
            .map_err(ScenarioError::Core)?;
        }
        let (c00, c01, c11) = (
            cov.get(0, 0).map_err(ScenarioError::Core)?,
            cov.get(0, 1).map_err(ScenarioError::Core)?,
            cov.get(1, 1).map_err(ScenarioError::Core)?,
        );
        let angle = libm::atan2(2.0 * c01, c00 - c11) * 0.5;
        let semi = match ellipse_from_cov2(c00, c01, c11, conf) {
            Ok(s) => s,
            Err(_) => (0.0, 0.0),
        };
        Ok((semi, angle))
    }

    pub fn reference_mean_at(step: usize) -> Result<[f64; 4], ScenarioError> {
        LOT9_REF_MEAN
            .get(step)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }
}

fn array_from_particles(particles: &[[f64; 4]]) -> [[f64; 4]; LOT9_NUM_PARTICLES] {
    let mut out = [[0.0; 4]; LOT9_NUM_PARTICLES];
    for (i, p) in particles.iter().enumerate().take(LOT9_NUM_PARTICLES) {
        out[i] = *p;
    }
    out
}
