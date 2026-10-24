//! Tutorial 04 particle filter (bearing–range, NumPy RNG).
#![allow(clippy::needless_range_loop)]

include!("lot2_pf_fixtures.rs");

use stonesoup_core::{
    mod_bearing, CartesianToBearingRange, ConstantVelocity2d, Duration, RmseAccumulator,
};
use stonesoup_sim::{mvn4_lot2_q, sample_lot2_pf_prior_particle, NumpyRng, LOT2_NUMPY_SEED};

use crate::lot2_ekf::{
    fixture_row, Lot2EkfScenario, Lot2Snapshot, Lot2StepOutcome, MEASUREMENTS, NUM_STEPS, TRUTH,
};
use crate::registry::ScenarioError;

pub const LOT2_PF_PARTICLES: usize = 1000;

pub struct Lot2ParticleScenario {
    particles: [[f64; 4]; LOT2_PF_PARTICLES],
    log_w: [f64; LOT2_PF_PARTICLES],
    rng: NumpyRng,
    transition: ConstantVelocity2d,
    measurement: CartesianToBearingRange,
    step: usize,
    rmse: RmseAccumulator,
}

impl Lot2ParticleScenario {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn tutorial_defaults() -> Result<Self, ScenarioError> {
        let mut rng = NumpyRng::seed(LOT2_NUMPY_SEED);
        let transition = ConstantVelocity2d::new(0.05, 0.05).map_err(ScenarioError::Core)?;
        let measurement = CartesianToBearingRange::tutorial02().map_err(ScenarioError::Core)?;
        let mean = [0.0, 1.0, 0.0, 1.0];
        let mut particles = [[0.0; 4]; LOT2_PF_PARTICLES];
        let log_uniform = -libm::log(LOT2_PF_PARTICLES as f64);
        let mut log_w = [0.0; LOT2_PF_PARTICLES];
        for i in 0..LOT2_PF_PARTICLES {
            particles[i] =
                sample_lot2_pf_prior_particle(&mut rng, mean).map_err(ScenarioError::Core)?;
            log_w[i] = log_uniform;
        }
        Ok(Self {
            particles,
            log_w,
            rng,
            transition,
            measurement,
            step: 0,
            rmse: RmseAccumulator::new(),
        })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        *self = Self::tutorial_defaults()?;
        Ok(())
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    fn weighted_mean(&self) -> [f64; 4] {
        let mut m = [0.0; 4];
        let mut max_log = f64::NEG_INFINITY;
        for &lw in &self.log_w {
            if lw > max_log {
                max_log = lw;
            }
        }
        let mut sum = 0.0;
        for i in 0..LOT2_PF_PARTICLES {
            let w = libm::exp(self.log_w[i] - max_log);
            sum += w;
            for d in 0..4 {
                m[d] += w * self.particles[i][d];
            }
        }
        if sum > 0.0 {
            for d in 0..4 {
                m[d] /= sum;
            }
        }
        m
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn step(&mut self) -> Result<Lot2StepOutcome, ScenarioError> {
        if self.step >= NUM_STEPS {
            return Ok(Lot2StepOutcome::Finished);
        }
        let ts = Lot2EkfScenario::timestamp_for(self.step);
        let predict_dt = if self.step == 0 {
            Duration::ZERO
        } else {
            Duration::SEC
        };
        let f = self
            .transition
            .transition_matrix(predict_dt)
            .map_err(ScenarioError::Core)?;
        for i in 0..LOT2_PF_PARTICLES {
            let mut x = stonesoup_core::linalg::fixed::mat4_vec4(
                f,
                stonesoup_core::VecN(self.particles[i]),
            )
            .map_err(ScenarioError::Core)?;
            if self.step > 0 {
                let noise = mvn4_lot2_q(&mut self.rng).map_err(ScenarioError::Core)?;
                for d in 0..4 {
                    x.0[d] += noise[d];
                }
            }
            self.particles[i] = x.0;
        }

        let m = fixture_row(&MEASUREMENTS, self.step)?;
        let r = self.measurement.r();
        let det_r =
            r.get(0, 0).map_err(ScenarioError::Core)? * r.get(1, 1).map_err(ScenarioError::Core)?;
        for i in 0..LOT2_PF_PARTICLES {
            let z = self
                .measurement
                .h(self.particles[i])
                .map_err(ScenarioError::Core)?;
            let innov0 = mod_bearing(m[0] - z[0]).map_err(ScenarioError::Core)?;
            let innov1 = m[1] - z[1];
            let ll = -0.5
                * (innov0 * innov0 / r.get(0, 0).map_err(ScenarioError::Core)?
                    + innov1 * innov1 / r.get(1, 1).map_err(ScenarioError::Core)?
                    + libm::log(2.0 * core::f64::consts::PI * det_r));
            self.log_w[i] += ll;
        }
        self.normalize_log_weights()?;
        let truth = fixture_row(&TRUTH, self.step)?;
        let mean = self.weighted_mean();
        self.rmse
            .push(mean[0] - truth[0], mean[2] - truth[2])
            .map_err(ScenarioError::Core)?;
        let _ = ts;
        self.step += 1;
        Ok(if self.step >= NUM_STEPS {
            Lot2StepOutcome::Finished
        } else {
            Lot2StepOutcome::Stepped
        })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    fn normalize_log_weights(&mut self) -> Result<(), ScenarioError> {
        let mut max_log = f64::NEG_INFINITY;
        for &lw in &self.log_w {
            if lw > max_log {
                max_log = lw;
            }
        }
        let mut sum = 0.0;
        for i in 0..LOT2_PF_PARTICLES {
            sum += libm::exp(self.log_w[i] - max_log);
        }
        let log_sum = max_log + libm::log(sum);
        for i in 0..LOT2_PF_PARTICLES {
            self.log_w[i] -= log_sum;
        }
        Ok(())
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
        let mean = self.weighted_mean();
        Ok(Lot2Snapshot {
            step: self.step,
            truth_xy: (truth[0], truth[2]),
            measurement_br: (m[0], m[1]),
            estimate_xy: (mean[0], mean[2]),
            ellipse_semi: (0.0, 0.0),
            ellipse_angle_rad: 0.0,
            rmse: self.rmse.rmse().unwrap_or(0.0),
            finished: self.step >= NUM_STEPS,
        })
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn mean_state(&self) -> [f64; 4] {
        self.weighted_mean()
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn reference_mean_at(&self, i: usize) -> Option<[f64; 4]> {
        PF_TRACK_REF.get(i).copied()
    }
}
