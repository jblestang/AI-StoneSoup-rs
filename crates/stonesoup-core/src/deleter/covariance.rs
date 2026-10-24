//! Covariance-based track deletion (Stone Soup ``CovarianceBasedDeleter``).

use crate::state::GaussianState;

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn covariance_trace(state: &GaussianState) -> Result<f64, crate::Error> {
    let c = &state.cov.0;
    let t = c[0][0] + c[1][1] + c[2][2] + c[3][3];
    if !t.is_finite() {
        return Err(crate::Error::NonFiniteInput);
    }
    Ok(t)
}

/// Returns true when track should be deleted (trace strictly above threshold).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn should_delete_covariance_trace(state: &GaussianState, threshold: f64) -> Result<bool, crate::Error> {
    Ok(covariance_trace(state)? > threshold)
}
