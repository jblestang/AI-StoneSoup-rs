#![no_std]
#![forbid(unsafe_code)]

//! Stone Soup core: tracking types, linear-Gaussian models, Kalman filter, metrics.
//!
//! State vector order for Lot 1 (2D nearly-constant velocity): `[x, vx, y, vy]` (SI: m, m/s).

extern crate alloc;

mod assign;
mod associator;
mod deleter;
mod error;
mod filter;
mod initiator;
mod kernel;
pub mod linalg;
mod metric;
mod models;
pub mod numeric;
mod state;
mod time;
mod types;

pub use assign::linear_sum_assignment_min;
pub use associator::{
    associate_global_nearest_neighbour, associate_gnn_2d, associate_gnn_2d_ekf,
    associate_gnn_2d_ekf_det_models, associate_gnn_2d_kdtree, associate_nearest_neighbour,
    ekf_pda_hypotheses, ekf_pda_merged_update, gm_reduce_single, jpda_associate,
    jpda_associate_ekf_det_models, jpda_detection_weights, pda_hypotheses, pda_merged_update,
    AssociationOutcome, MAX_JPDA_EKF_TRACKS,
    GnnAssignment, GnnTrackAssignment, JpdaHypothesisSet, KdTree2D, MeasurementHypothesis,
    ProbabilityHypothesis, KDTREE_GNN_COV_MULTIPLIER, KDTREE_GNN_NEIGHBOURS, MAX_GNN_ASSIGN_DETS,
    MAX_GNN_TRACKS, MAX_KDTREE_POINTS, MAX_PDA_HYPOTHESES, PDA_GATE_CHI2_2D,
    assess_track_quality, associate_tracks_to_truths, continuous_tracks_per_truth,
    diagnose_identity_switches_from_assoc, list_coast_relock_handoffs, study_coast_relock_imm,
    study_coast_relocks, CoastRelockImmStudy, ImmDominantCounts,
    study_radar_a_coast_relocks, CoastRelockHandoff, CoastRelockStudy, RadarACoastRelockStudy,
    SwitchCause,
    SwitchDiagnosisSummary,
    squared_mahalanobis, ContinuousTruthQuality, ContinuousTruthTrack, FilterTrackQuality,
    TrackQualityReport,
    RadarRevisitTrackToTruth, TrackPath, TrackToTruthConfig, TrackTruthAssociation,
    TrackTruthPositionGate, TruthPath,
};
pub use deleter::{
    covariance_trace, holding_should_delete, should_delete_covariance_trace, HoldingDeleter,
    UpdateTimeStepsDeleter,
};
pub use error::Error;
pub use filter::akkf::{
    transition_particles, AdaptiveKernelKalmanPredictor, AdaptiveKernelKalmanUpdater,
    KernelParticleState, AKKF_MAX_PARTICLES,
};
pub use filter::asd::{
    AsdGaussianState, AsdKalmanPredictor, AsdKalmanUpdater, AsdMeasurementPrediction,
    AsdPrediction, ASD_DEFAULT_MAX_NSTEP, ASD_NDIM,
};
pub use filter::ekf::ExtendedKalmanUpdater;
pub use filter::gm_phd::{
    gm_phd_append_detection_update, gm_phd_append_missed_update, gm_phd_collect_missed,
    gm_phd_deduplicate_tags, gm_phd_hypotheses_for_detection, gm_phd_merge, gm_phd_merge_pair,
    gm_phd_prune, gm_phd_reduce, squared_mahalanobis_state, GmPhdConfig, PhdComponent,
    PhdHypothesis, PhdTag, MAX_GMPHD_COMPONENTS, MAX_GMPHD_DETECTIONS, MAX_GMPHD_UPDATED,
};
pub use filter::imm_ekf::{ImmEkfConfig, ImmEkfState, ImmModelKind, IMM_MODEL_COUNT};
pub use filter::information::{
    InformationKalmanPredictor, InformationKalmanUpdater, InformationKalmanWorkspace,
    InformationState,
};
pub use filter::kalman::{
    GaussianMeasurementPrediction, KalmanPredictor, KalmanUpdater, KalmanWorkspace,
};
pub use filter::ukf::{UnscentedKalmanPredictor, UnscentedKalmanUpdater};
pub use initiator::{
    initiate_from_detection, tutorial09_prior, tutorial10_prior, HoldingGaussianTrack,
    MultiMeasurementInitiator,
};
pub use kernel::QuadraticKernel;
pub use linalg::{Mat, VecN};
pub use metric::rmse::RmseAccumulator;
pub use models::bearing_only::Cartesian2DToBearing;
pub use models::bearing_range::CartesianToBearingRange;
pub use models::measurement::LinearGaussianMeasurement;
pub use models::transition::ConstantVelocity2d;
pub use state::GaussianState;
pub use time::{Duration, Timestamp};
pub use types::{mod_bearing, Detection, GroundTruthState, SensorId, TrackId};
