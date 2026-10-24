//! Simple measurement initiator (linear position mapping).

use crate::error::Error;
use crate::filter::kalman::{KalmanUpdater, KalmanWorkspace};
use crate::linalg::Mat;
use crate::state::GaussianState;
use crate::types::Detection;

/// Tutorial 09/10 prior: zero position, unit velocity variance on unobserved velocity.
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn tutorial09_prior() -> Result<GaussianState, Error> {
    let mut cov = Mat::<4, 4>::zeros();
    cov.set(1, 1, 1.0)?;
    cov.set(3, 3, 1.0)?;
    GaussianState::new_from_array([0.0, 0.0, 0.0, 0.0], cov, crate::time::Timestamp::ZERO)
}

/// Tutorial 10 prior diag(0, 0.5, 0, 0.5).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn tutorial10_prior() -> Result<GaussianState, Error> {
    let mut cov = Mat::<4, 4>::zeros();
    cov.set(1, 1, 0.5)?;
    cov.set(3, 3, 0.5)?;
    GaussianState::new_from_array([0.0, 0.0, 0.0, 0.0], cov, crate::time::Timestamp::ZERO)
}

fn zero_mapped_rows_cols(cov: &mut Mat<4, 4>, mapped: &[usize]) -> Result<(), Error> {
    for &d in mapped {
        for j in 0..4 {
            cov.set(d, j, 0.0)?;
            cov.set(j, d, 0.0)?;
        }
    }
    Ok(())
}

/// ``SimpleMeasurementInitiator`` for position linear Gaussian (mapping 0,2).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn initiate_from_detection(
    prior: &GaussianState,
    detection: &Detection,
    updater: &KalmanUpdater,
    _ws: &mut KalmanWorkspace,
) -> Result<GaussianState, Error> {
    let z = detection.vector;
    let mut mean = prior.mean.0;
    mean[0] = 0.0;
    mean[2] = 0.0;
    mean[0] = z.0[0];
    mean[2] = z.0[1];

    let r = updater.measurement.r();
    let mut cov = prior.cov;
    zero_mapped_rows_cols(&mut cov, &[0, 2])?;
    let r00 = r.get(0, 0)?;
    let r11 = r.get(1, 1)?;
    cov.set(0, 0, cov.get(0, 0)? + r00)?;
    cov.set(2, 2, cov.get(2, 2)? + r11)?;

    GaussianState::new_from_array(mean, cov, detection.timestamp)
}
