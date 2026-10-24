//! Stone Soup sampling tutorial — particle resamplers (`ResamplingTutorial.py`).

use crate::lot5_resampling_fixtures as fx;
use crate::registry::ScenarioError;
use crate::scenario::{empty_resample_xy, ViewerSnapshot, MAX_VIEWER_RESAMPLE_PARTS};
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

    /// Number of resampler demo steps completed (0 before the first [`step`]).
    pub fn completed_steps(&self) -> usize {
        self.step
    }

    pub fn resample_state(&self, index: usize) -> f64 {
        self.particles[index][0]
    }

    pub fn resample_log_weight(&self, index: usize) -> f64 {
        self.log_w[index]
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        *self = Self::tutorial_defaults()?;
        Ok(())
    }

    pub fn viewer_snapshot(&self) -> Result<ViewerSnapshot, ScenarioError> {
        let case = self.step.saturating_sub(1);
        let mut resample_xy = empty_resample_xy();
        let nparts = self.nparts.min(MAX_VIEWER_RESAMPLE_PARTS);
        for (i, slot) in resample_xy.iter_mut().enumerate().take(nparts) {
            *slot = [i as f64, self.particles[i][0]];
        }
        let resample_case = if self.step == 0 {
            u8::MAX
        } else {
            case.min(u8::MAX as usize) as u8
        };
        Ok(ViewerSnapshot {
            step: self.step,
            max_steps: MAX_LOT5_RESAMPLING_STEPS,
            truth_xy: (0.0, 0.0),
            estimate_xy: (0.0, 0.0),
            measurement_xy: (0.0, 0.0),
            bearing_range: None,
            ellipse_semi: None,
            ellipse_angle_rad: 0.0,
            rmse: 0.0,
            finished: self.finished,
            det_count: 0,
            det_xy: crate::scenario::empty_det_xy(),
            det_is_true: crate::scenario::empty_det_flags(),
            associated_xy: None,
            truth_b_xy: None,
            estimate_b_xy: None,
            resample_case,
            resample_nparts: nparts as u8,
            resample_xy,
        })
    }
}

pub fn lot5_case_name(case: u8) -> &'static str {
    match case {
        0 => fx::LOT5_0_NAME,
        1 => fx::LOT5_1_NAME,
        2 => fx::LOT5_2_NAME,
        3 => fx::LOT5_3_NAME,
        4 => fx::LOT5_4_NAME,
        5 => fx::LOT5_5_NAME,
        _ => "—",
    }
}
