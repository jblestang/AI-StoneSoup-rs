//! Stone Soup sampling tutorial — particle resamplers (`ResamplingTutorial.py`).

use crate::lot5_resampling_fixtures as fx;
use crate::registry::ScenarioError;
use stonesoup_sim::{
    ess_resample_if_needed_with, multinomial_resample, replay_resampling_tutorial_preamble,
    residual_resample, stratified_resample, systematic_resample, EssInnerResampler, NumpyRng,
    ResidualStage2Method, LOT2_NUMPY_SEED, MAX_RESAMPLE_PARTICLES,
};

pub const MAX_LOT5_RESAMPLING_STEPS: usize = fx::LOT5_RESAMPLING_CASES;

#[derive(Debug, Clone)]
pub struct Lot5ResamplingScenario {
    step: usize,
    rng: NumpyRng,
    particles: [[f64; 4]; MAX_RESAMPLE_PARTICLES],
    log_w: [f64; MAX_RESAMPLE_PARTICLES],
    nparts: usize,
    finished: bool,
}

impl Lot5ResamplingScenario {
    pub fn tutorial_defaults() -> Result<Self, ScenarioError> {
        Ok(Self {
            step: 0,
            rng: NumpyRng::seed(LOT2_NUMPY_SEED),
            particles: [[0.0; 4]; MAX_RESAMPLE_PARTICLES],
            log_w: [0.0; MAX_RESAMPLE_PARTICLES],
            nparts: 0,
            finished: false,
        })
    }

    fn init_equal(&mut self) {
        for i in 0..5 {
            self.particles[i][0] = i as f64;
            self.log_w[i] = libm::log(0.2);
        }
    }

    fn init_weighted(&mut self) {
        let w = [0.1, 0.5, 0.15, 0.05, 0.2];
        for (i, &wi) in w.iter().enumerate() {
            self.particles[i][0] = i as f64;
            self.log_w[i] = libm::log(wi);
        }
    }

    pub fn step(&mut self) -> Result<bool, ScenarioError> {
        if self.finished {
            return Ok(false);
        }
        match self.step {
            0 => {
                replay_resampling_tutorial_preamble(&mut self.rng, 5);
                self.init_equal();
                multinomial_resample(&mut self.particles, &mut self.log_w, 5, 5, &mut self.rng)
                    .map_err(ScenarioError::Core)?;
                self.nparts = 5;
            }
            1 => {
                self.init_equal();
                multinomial_resample(&mut self.particles, &mut self.log_w, 5, 10, &mut self.rng)
                    .map_err(ScenarioError::Core)?;
                self.nparts = 10;
            }
            2 => {
                self.rng = NumpyRng::seed(LOT2_NUMPY_SEED);
                replay_resampling_tutorial_preamble(&mut self.rng, 5);
                self.init_equal();
                systematic_resample(&mut self.particles, &mut self.log_w, 5, 5, &mut self.rng)
                    .map_err(ScenarioError::Core)?;
                self.nparts = 5;
            }
            3 => {
                self.rng = NumpyRng::seed(LOT2_NUMPY_SEED);
                replay_resampling_tutorial_preamble(&mut self.rng, 5);
                self.init_equal();
                stratified_resample(&mut self.particles, &mut self.log_w, 5, 5, &mut self.rng)
                    .map_err(ScenarioError::Core)?;
                self.nparts = 5;
            }
            4 => {
                self.rng = NumpyRng::seed(LOT2_NUMPY_SEED);
                replay_resampling_tutorial_preamble(&mut self.rng, 5);
                self.init_weighted();
                residual_resample(
                    &mut self.particles,
                    &mut self.log_w,
                    5,
                    ResidualStage2Method::Systematic,
                    &mut self.rng,
                )
                .map_err(ScenarioError::Core)?;
                self.nparts = 5;
            }
            5 => {
                self.rng = NumpyRng::seed(LOT2_NUMPY_SEED);
                replay_resampling_tutorial_preamble(&mut self.rng, 5);
                self.init_equal();
                let _ = ess_resample_if_needed_with(
                    &mut self.particles,
                    &mut self.log_w,
                    5,
                    2.5,
                    &mut self.rng,
                    EssInnerResampler::ResidualSystematic,
                )
                .map_err(ScenarioError::Core)?;
                self.nparts = 5;
            }
            _ => {
                self.finished = true;
                return Ok(false);
            }
        }
        self.step += 1;
        if self.step >= MAX_LOT5_RESAMPLING_STEPS {
            self.finished = true;
        }
        Ok(!self.finished)
    }

    /// First particle state value at current step (1-D tutorial particles).
    pub fn primary_particle_value(&self) -> f64 {
        self.particles[0][0]
    }

    pub fn nparts(&self) -> usize {
        self.nparts
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }
}
