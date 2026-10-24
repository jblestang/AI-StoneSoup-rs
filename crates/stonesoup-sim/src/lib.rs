#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

mod lot1_tutorial;
mod lot2_tutorial;
mod lot9_tutorial;
mod mvn;
mod numpy_random;
mod resampler;
mod rng;
mod sensor;
mod sym4;
mod truth;

pub use lot1_tutorial::{
    generate_lot1_measurements, generate_lot1_tutorial, Lot1TutorialData, LOT1_NUMPY_SEED,
    LOT1_NUM_STEPS,
};
pub use lot2_tutorial::{
    advance_lot2_pf_preamble, generate_lot2_tutorial, mvn4_lot2_q, sample_lot2_pf_prior_particle,
    Lot2TutorialData, LOT2_NUMPY_SEED, LOT2_NUM_STEPS,
};
pub use lot9_tutorial::{
    advance_lot9_preamble, initialise_lot9_filter_rng, prior_covariance, process_covariance,
    LOT9_NUMPY_SEED, LOT9_NUM_PARTICLES, LOT9_NUM_STEPS, LOT9_PRIOR_MEAN,
};
pub use mvn::multivariate_normal4_batch;
pub use numpy_random::NumpyRng;
pub use resampler::{
    effective_sample_size, ess_resample_if_needed, ess_resample_if_needed_with,
    multinomial_resample, replay_resampling_tutorial_preamble, residual_resample,
    stratified_resample, systematic_resample, EssInnerResampler, ResidualStage2Method,
    MAX_RESAMPLE_PARTICLES,
};
pub use rng::{Rng, SplitMix64};
pub use sensor::{LinearPositionSensor, SensorSim};
pub use sym4::mvn_factor_a;
pub use truth::{GroundTruthPath, TruthStepConfig};
