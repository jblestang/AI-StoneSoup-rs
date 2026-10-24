#![no_std]
#![forbid(unsafe_code)]

mod lot1_kalman;
mod lot2_ekf;
mod lot2_ukf;
mod particle_lot2;
mod registry;
pub mod scenario;

pub use lot1_kalman::{
    tutorial_cov, Lot1KalmanScenario, Lot1Snapshot, Lot1StepOutcome, MAX_LOT1_STEPS, MEASUREMENTS,
    TRUTH,
};
pub use lot2_ekf::{Lot2EkfScenario, Lot2Snapshot, Lot2StepOutcome, MAX_LOT2_STEPS};
pub use lot2_ukf::Lot2UkfScenario;
pub use particle_lot2::{Lot2ParticleScenario, LOT2_PF_PARTICLES};
pub use registry::{ScenarioDescriptor, ScenarioError, ScenarioId, ScenarioRegistry};
