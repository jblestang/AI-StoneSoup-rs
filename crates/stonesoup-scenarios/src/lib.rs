#![no_std]
#![forbid(unsafe_code)]

mod lot1_kalman;
mod lot2_ekf;
mod registry;
pub mod scenario;

pub use lot1_kalman::{
    tutorial_cov, Lot1KalmanScenario, Lot1Snapshot, Lot1StepOutcome, MAX_LOT1_STEPS, MEASUREMENTS,
    TRUTH,
};
pub use lot2_ekf::{Lot2EkfScenario, Lot2Snapshot, Lot2StepOutcome, MAX_LOT2_STEPS};
pub use registry::{ScenarioDescriptor, ScenarioError, ScenarioId, ScenarioRegistry};
