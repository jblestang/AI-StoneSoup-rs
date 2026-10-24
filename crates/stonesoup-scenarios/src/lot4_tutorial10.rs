#[allow(dead_code)]
mod fixtures {
    include!("lot4_tutorial10_fixtures.rs");
}
use fixtures::*;

use stonesoup_core::tutorial10_prior;

use crate::lot4_tracker::{Lot4MultiTargetTracker, Lot4TrackerParams, MAX_LOT4_ACTIVE_TRACKS};
use crate::registry::ScenarioError;

pub const MAX_LOT4_T10_STEPS: usize = NUM_STEPS;

pub struct Lot4Tutorial10Scenario {
    inner: Lot4MultiTargetTracker,
}

impl Lot4Tutorial10Scenario {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn tutorial_defaults() -> Result<Self, ScenarioError> {
        let prior = tutorial10_prior().map_err(ScenarioError::Core)?;
        let params = Lot4TrackerParams {
            gate: GATE,
            deleter_trace: DELETER_TRACE,
            min_points: MIN_POINTS,
            step_seconds: STEP_SECONDS,
            process_noise: PROCESS_NOISE,
            meas_var: MEAS_VAR,
            prior,
            gnn: crate::lot4_tracker::Lot4GnnAssociator::Linear,
        };
        let inner = Lot4MultiTargetTracker::new(params, NUM_STEPS, &MEAS_OFFSETS, &MEAS_XY)?;
        Ok(Self { inner })
    }

    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        self.inner.reset()
    }

    pub fn step(&mut self) -> Result<bool, ScenarioError> {
        self.inner.step()
    }

    pub fn run_to_end(&mut self) -> Result<(), ScenarioError> {
        self.inner.run_to_end()
    }

    pub fn step_index(&self) -> usize {
        self.inner.step_index()
    }

    pub fn reference_track_means_at(step: usize) -> Result<&'static [[f64; 4]], ScenarioError> {
        let start = SNAPSHOT_OFFSETS
            .get(step)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        let end = SNAPSHOT_OFFSETS
            .get(step + 1)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        SNAPSHOT_TRACK_MEANS
            .get(start..end)
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }

    pub fn snapshot_track_means(&self, out: &mut [[f64; 4]; MAX_LOT4_ACTIVE_TRACKS]) -> usize {
        self.inner.sorted_track_means(out)
    }

    pub fn viewer_snapshot(&self) -> Result<crate::scenario::ViewerSnapshot, ScenarioError> {
        crate::scenario::lot4_viewer_from(
            self.step_index(),
            MAX_LOT4_T10_STEPS,
            self,
            &MEAS_OFFSETS,
            &MEAS_XY,
        )
    }
}
