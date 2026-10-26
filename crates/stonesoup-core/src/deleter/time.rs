//! Time-based track deletion (Stone Soup ``UpdateTimeStepsDeleter``).

/// Delete when more than this many scan steps pass without a measurement update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UpdateTimeStepsDeleter {
    pub time_steps_since_update: usize,
}

impl UpdateTimeStepsDeleter {
    #[must_use]
    pub const fn new(time_steps_since_update: usize) -> Self {
        Self {
            time_steps_since_update,
        }
    }

    #[must_use]
    pub const fn should_delete(&self, steps_since_update: usize) -> bool {
        steps_since_update > self.time_steps_since_update
    }
}
