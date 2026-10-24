#![no_std]
#![forbid(unsafe_code)]

//! Stone Soup core: tracking types, linear-Gaussian models, Kalman filter, metrics.
//!
//! State vector order for Lot 1 (2D nearly-constant velocity): `[x, vx, y, vy]` (SI: m, m/s).

mod error;
mod filter;
pub mod linalg;
mod metric;
mod models;
mod state;
mod time;
mod types;

pub use error::Error;
pub use filter::kalman::{KalmanPredictor, KalmanUpdater, KalmanWorkspace};
pub use linalg::{Mat, VecN};
pub use metric::rmse::RmseAccumulator;
pub use models::measurement::LinearGaussianMeasurement;
pub use models::transition::ConstantVelocity2d;
pub use state::GaussianState;
pub use time::{Duration, Timestamp};
pub use types::{Detection, GroundTruthState, SensorId, TrackId};
