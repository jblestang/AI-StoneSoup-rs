#![no_std]
#![forbid(unsafe_code)]

mod lot1_kalman;
mod registry;
pub mod scenario;

pub use lot1_kalman::{
    tutorial_cov, Lot1KalmanScenario, Lot1Snapshot, Lot1StepOutcome, MAX_LOT1_STEPS,
};
pub use registry::{ScenarioDescriptor, ScenarioError, ScenarioId, ScenarioRegistry};
