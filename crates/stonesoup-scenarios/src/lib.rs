#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

mod dual_radar_clutter;
mod lot10_single_sensor;
mod lot11_kdtree;
mod lot13_multi_sensor;
mod lot14_optimised_sensor;
mod lot15_network_info_arch;
mod lot16_data_incest;
mod lot17_arch_intro;
mod lot18_simple_oosm;
mod lot19_kalman_oosm;
mod lot1_kalman;
mod lot20_pf_oosm;
mod lot21_oosm_algorithm;
mod lot22_bearing_only;
mod lot23_multi_tracker;
mod lot24_ensemble_filter;
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
mod lot8_asd;
mod lot9_akkf;
mod particle_lot2;
mod registry;
pub mod scenario;

pub use lot10_single_sensor::{
    Lot10SensorManagerMode, Lot10SingleSensorScenario, LOT10_SENSOR_XY, MAX_LOT10_STEPS,
    MAX_LOT10_TRACKS,
};
pub use lot11_kdtree::{Lot11KdtreeScenario, MAX_LOT11_KDTREE_STEPS};
pub use lot13_multi_sensor::{
    Lot13MultiSensorScenario, Lot13SensorManagerMode, LOT13_SENSOR_X, MAX_LOT13_STEPS,
    MAX_LOT13_TRACKS,
};
pub use lot14_optimised_sensor::{
    Lot14OptimisedSensorMode, Lot14OptimisedSensorScenario, LOT14_SENSOR_X, MAX_LOT14_STEPS,
    MAX_LOT14_TRACKS,
};
pub use lot15_network_info_arch::{
    Lot15ArchitectureMode, Lot15NetworkInfoArchScenario, MAX_LOT15_STEPS, MAX_LOT15_TRACKS,
};
pub use lot16_data_incest::{
    Lot16DataIncestMode, Lot16DataIncestScenario, MAX_LOT16_STEPS, MAX_LOT16_TRACKS,
};
pub use lot17_arch_intro::{Lot17ArchIntroScenario, LOT17_INTRO_EDGE_COUNT, MAX_LOT17_STEPS};
pub use lot18_simple_oosm::{
    Lot18OosmMode, Lot18SimpleOosmScenario, MAX_LOT18_SCANS, MAX_LOT18_STEPS,
};
pub use lot19_kalman_oosm::{
    Lot19KalmanOosmMode, Lot19KalmanOosmScenario, MAX_LOT19_LOOPS, MAX_LOT19_STEPS,
};
pub use lot1_kalman::{
    tutorial_cov, Lot1KalmanScenario, Lot1Snapshot, Lot1StepOutcome, MAX_LOT1_STEPS, MEASUREMENTS,
    TRUTH,
};
pub use lot20_pf_oosm::{Lot20PfOosmMode, Lot20PfOosmScenario, MAX_LOT20_SCANS, MAX_LOT20_STEPS};
pub use lot21_oosm_algorithm::{
    Lot21OosmAlgorithmMode, Lot21OosmAlgorithmScenario, LOT21_NUM_SCANS, MAX_LOT21_STEPS,
};
pub use lot22_bearing_only::{Lot22BearingOnlyScenario, MAX_LOT22_STEPS};
pub use lot23_multi_tracker::{
    Lot23MultiTrackerMode, Lot23MultiTrackerScenario, MAX_LOT23_STEPS, MAX_LOT23_TRACKS,
};
pub use lot24_ensemble_filter::{Lot24EnsembleFilterScenario, MAX_LOT24_STEPS};
pub use dual_radar_clutter::{
    max_simulation_steps, CompletedTrack, DualRadarClutterSim, DualRadarStats, MAX_ACTIVE,
    TruthSensorPlotCounts,
    CLUTTER_FRACTION_TARGET, CLUTTER_LAMBDA, DT_S, FOV_DEG, NUM_TRUTHS, OVERLAP_AREA_FRACTION,
    RADAR_BEARING_SIGMA_DEG, RADAR_RANGE_SIGMA_M,
    PLOTS_PER_RADAR, PLOTS_PER_ROTATION, TARGETS_PER_ROTATION, PERIOD_A_S, PERIOD_B_S, RANGE_A_M,
    RANGE_B_M, SEED,
    SIM_DURATION_S, INITIATOR_MIN_POINTS, HOLDING_TIME_STEPS_DELETER, TRUTH_ASSOC_METERS,
    TRUTH_SPEED_250_KNOTS_MPS, TRUTH_SPEED_400_KNOTS_MPS, TRUTH_SPEED_800_KNOTS_MPS,
    TRUTH_SPEED_MACH1_MPS, TRUTH_SPEED_MACH2_MPS, TRUTH_MAX_MANEUVER_G,
};
pub use stonesoup_core::{ContinuousTruthTrack, TrackTruthAssociation};
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
pub use lot8_asd::{Lot8AsdScenario, Lot8Snapshot, Lot8StepOutcome, MAX_LOT8_STEPS};
pub use lot9_akkf::{Lot9AkkfScenario, Lot9Snapshot, Lot9StepOutcome, MAX_LOT9_STEPS};
pub use particle_lot2::{Lot2ParticleScenario, LOT2_PF_PARTICLES};
pub use registry::{ScenarioDescriptor, ScenarioError, ScenarioId, ScenarioRegistry};
pub use scenario::{AnyScenario, Scenario, ScenarioStepOutcome, ViewerSnapshot};
