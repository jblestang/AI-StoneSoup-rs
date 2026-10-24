#![no_std]
#![forbid(unsafe_code)]

//! Stone Soup core: tracking types, linear-Gaussian models, Kalman filter, metrics.
//!
//! State vector order for Lot 1 (2D nearly-constant velocity): `[x, vx, y, vy]` (SI: m, m/s).

mod assign;
mod associator;
mod deleter;
mod error;
mod filter;
mod initiator;
pub mod linalg;
mod metric;
mod models;
pub mod numeric;
mod state;
mod time;
mod types;

pub use assign::linear_sum_assignment_min;
pub use associator::{
    associate_global_nearest_neighbour, associate_gnn_2d, associate_nearest_neighbour,
    gm_reduce_single, jpda_associate, pda_hypotheses, pda_merged_update, AssociationOutcome,
    GnnAssignment, GnnTrackAssignment, JpdaHypothesisSet, MeasurementHypothesis,
    ProbabilityHypothesis, MAX_GNN_ASSIGN_DETS, MAX_GNN_ASSIGN_TRACKS, MAX_GNN_TRACKS,
    MAX_PDA_HYPOTHESES, PDA_GATE_CHI2_2D,
};
pub use deleter::{covariance_trace, should_delete_covariance_trace};
pub use error::Error;
pub use filter::ekf::ExtendedKalmanUpdater;
pub use filter::kalman::{
    GaussianMeasurementPrediction, KalmanPredictor, KalmanUpdater, KalmanWorkspace,
};
pub use filter::ukf::{UnscentedKalmanPredictor, UnscentedKalmanUpdater};
pub use initiator::{initiate_from_detection, tutorial09_prior, tutorial10_prior};
pub use linalg::{Mat, VecN};
pub use metric::rmse::RmseAccumulator;
pub use models::bearing_range::CartesianToBearingRange;
pub use models::measurement::LinearGaussianMeasurement;
pub use models::transition::ConstantVelocity2d;
pub use state::GaussianState;
pub use time::{Duration, Timestamp};
pub use types::{mod_bearing, Detection, GroundTruthState, SensorId, TrackId};
