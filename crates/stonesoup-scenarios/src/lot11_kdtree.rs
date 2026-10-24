#[allow(dead_code)]
mod fixtures {
    include!("lot11_kdtree_fixtures.rs");
}
use fixtures::*;

use stonesoup_core::tutorial10_prior;

use crate::lot4_tracker::{
    Lot4GnnAssociator, Lot4MultiTargetTracker, Lot4TrackerParams, MAX_LOT4_ACTIVE_TRACKS,
};
use crate::registry::ScenarioError;

pub const MAX_LOT11_KDTREE_STEPS: usize = LOT11_NUM_STEPS;

pub struct Lot11KdtreeScenario {
    inner: Lot4MultiTargetTracker,
}

impl Lot11KdtreeScenario {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn tutorial_defaults() -> Result<Self, ScenarioError> {
        let prior = tutorial10_prior().map_err(ScenarioError::Core)?;
        let params = Lot4TrackerParams {
            gate: LOT11_GATE,
            deleter_trace: LOT11_DELETER_TRACE,
            min_points: LOT11_MIN_POINTS,
            step_seconds: LOT11_STEP_SECONDS,
            process_noise: LOT11_PROCESS_NOISE,
            meas_var: LOT11_MEAS_VAR,
            prior,
            gnn: Lot4GnnAssociator::KdTree,
        };
        let inner = Lot4MultiTargetTracker::new(
            params,
            LOT11_NUM_STEPS,
            &LOT11_MEAS_OFFSETS,
            &LOT11_MEAS_XY,
        )?;
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
        let start = LOT11_SNAPSHOT_OFFSETS
            .get(step)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        let end = LOT11_SNAPSHOT_OFFSETS
            .get(step + 1)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        LOT11_SNAPSHOT_TRACK_MEANS
            .get(start..end)
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }

    pub fn snapshot_track_means(&self, out: &mut [[f64; 4]; MAX_LOT4_ACTIVE_TRACKS]) -> usize {
        self.inner.sorted_track_means(out)
    }

    pub fn viewer_snapshot(&self) -> Result<crate::scenario::ViewerSnapshot, ScenarioError> {
        crate::scenario::lot4_viewer_from(
            self.step_index(),
            MAX_LOT11_KDTREE_STEPS,
            self,
            &LOT11_MEAS_OFFSETS,
            &LOT11_MEAS_XY,
        )
    }
}
