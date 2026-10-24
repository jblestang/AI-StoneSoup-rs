#![no_std]
#![forbid(unsafe_code)]

mod lot1_kalman;
mod lot2_ekf;
mod lot2_ukf;
mod lot3_clutter;
mod lot3_jpda;
mod lot3_meas;
mod lot3_multitarget;
mod lot3_pda;
mod lot4_tracker;
mod lot4_tutorial09;
mod lot4_tutorial10;
mod lot5_resampling;
pub mod lot5_resampling_fixtures;
mod lot6_information;
mod lot7_gm_phd;
mod particle_lot2;
mod registry;
pub mod scenario;

pub use lot1_kalman::{
    tutorial_cov, Lot1KalmanScenario, Lot1Snapshot, Lot1StepOutcome, MAX_LOT1_STEPS, MEASUREMENTS,
    TRUTH,
};
pub use lot2_ekf::{
    bearing_range_to_xy, Lot2EkfScenario, Lot2Snapshot, Lot2StepOutcome, MAX_LOT2_STEPS,
    TUTORIAL02_SENSOR_XY,
};
pub use lot2_ukf::Lot2UkfScenario;
pub use lot3_clutter::{
    Lot3ClutterScenario, Lot3MeasPoint, Lot3Snapshot, Lot3StepOutcome, LOT3_GATE, MAX_LOT3_STEPS,
};
pub use lot3_jpda::{Lot3JpdaScenario, MAX_LOT3_JPDA_STEPS};
pub use lot3_multitarget::{Lot3MultiTargetScenario, MAX_LOT3_MT_STEPS};
pub use lot3_pda::{Lot3PdaScenario, MAX_LOT3_PDA_STEPS};
pub use lot4_tutorial09::{Lot4Tutorial09Scenario, MAX_LOT4_T09_STEPS};
pub use lot4_tutorial10::{Lot4Tutorial10Scenario, MAX_LOT4_T10_STEPS};
pub use lot5_resampling::{lot5_case_name, Lot5ResamplingScenario, MAX_LOT5_RESAMPLING_STEPS};
pub use lot6_information::{
    Lot6InformationScenario, Lot6Snapshot, Lot6StepOutcome, MAX_LOT6_STEPS,
};
pub use lot7_gm_phd::{Lot7GmPhdScenario, Lot7StepOutcome, LOT7_REDUCED, MAX_LOT7_STEPS};
pub use particle_lot2::{Lot2ParticleScenario, LOT2_PF_PARTICLES};
pub use registry::{ScenarioDescriptor, ScenarioError, ScenarioId, ScenarioRegistry};
pub use scenario::{AnyScenario, Scenario, ScenarioStepOutcome, ViewerSnapshot};
