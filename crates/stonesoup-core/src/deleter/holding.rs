//! Deleters applied to tentative (holding) tracks inside ``MultiMeasurementInitiator``.

use super::covariance::should_delete_covariance_trace;
use super::time::UpdateTimeStepsDeleter;
use crate::state::GaussianState;
use crate::Error;

/// Stone Soup initiator holding-track deletion policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HoldingDeleter {
    CovarianceTrace(f64),
    UpdateTimeSteps(UpdateTimeStepsDeleter),
}

impl HoldingDeleter {
    #[must_use]
    pub const fn covariance_trace(threshold: f64) -> Self {
        Self::CovarianceTrace(threshold)
    }

    #[must_use]
    pub const fn update_time_steps(time_steps_since_update: usize) -> Self {
        Self::UpdateTimeSteps(UpdateTimeStepsDeleter::new(
            time_steps_since_update,
        ))
    }
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn holding_should_delete(
    tail: &GaussianState,
    steps_since_update: usize,
    deleter: &HoldingDeleter,
) -> Result<bool, Error> {
    match deleter {
        HoldingDeleter::CovarianceTrace(threshold) => {
            should_delete_covariance_trace(tail, *threshold)
        }
        HoldingDeleter::UpdateTimeSteps(d) => Ok(d.should_delete(steps_since_update)),
    }
}
