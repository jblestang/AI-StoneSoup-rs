#[allow(dead_code, clippy::excessive_precision, clippy::approx_constant)]
mod fixtures {
    include!("lot22_bearing_only_fixtures.rs");
}
use fixtures::*;

use crate::registry::ScenarioError;
use crate::scenario::{empty_det_flags, empty_det_xy, empty_resample_xy, ViewerSnapshot};

pub const MAX_LOT22_STEPS: usize = LOT22_NUM_STEPS + 1;

pub struct Lot22BearingOnlyScenario {
    step: usize,
    mean: [f64; 4],
}

impl Lot22BearingOnlyScenario {
    pub fn tutorial_defaults() -> Result<Self, ScenarioError> {
        Ok(Self {
            step: 0,
            mean: LOT22_TRACK[0],
        })
    }

    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        *self = Self::tutorial_defaults()?;
        Ok(())
    }

    pub fn reference_mean_at(step_idx: usize) -> Result<[f64; 4], ScenarioError> {
        LOT22_TRACK
            .get(step_idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }

    pub fn truth_at(step_idx: usize) -> Result<[f64; 4], ScenarioError> {
        LOT22_TRUTH
            .get(step_idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }

    pub fn step(&mut self) -> Result<bool, ScenarioError> {
        if self.step >= MAX_LOT22_STEPS {
            return Ok(false);
        }
        if self.step == 0 {
            self.step = 1;
            return Ok(self.step < MAX_LOT22_STEPS);
        }
        let idx = self.step - 1;
        if idx >= LOT22_NUM_STEPS {
            return Ok(false);
        }
        self.mean = Self::reference_mean_at(idx)?;
        self.step += 1;
        Ok(self.step < MAX_LOT22_STEPS)
    }

    pub fn run_to_end(&mut self) -> Result<(), ScenarioError> {
        while self.step()? {}
        Ok(())
    }

    pub fn estimate_mean(&self) -> [f64; 4] {
        self.mean
    }

    pub fn scan_index(&self) -> usize {
        if self.step == 0 {
            0
        } else {
            self.step.saturating_sub(1).min(LOT22_NUM_STEPS)
        }
    }

    pub fn viewer_snapshot(&self) -> Result<ViewerSnapshot, ScenarioError> {
        let idx = self.scan_index().min(LOT22_NUM_STEPS.saturating_sub(1));
        let truth = LOT22_TRUTH[idx];
        let plat = LOT22_PLATFORM_XY[idx];
        let m = self.mean;
        Ok(ViewerSnapshot {
            step: self.step,
            max_steps: MAX_LOT22_STEPS,
            truth_xy: (truth[0], truth[2]),
            estimate_xy: (m[0], m[2]),
            measurement_xy: (plat[0], plat[1]),
            bearing_range: None,
            ellipse_semi: None,
            ellipse_angle_rad: 0.0,
            rmse: 0.0,
            finished: self.step >= MAX_LOT22_STEPS,
            det_count: 0,
            det_xy: empty_det_xy(),
            det_is_true: empty_det_flags(),
            associated_xy: None,
            truth_b_xy: None,
            estimate_b_xy: None,
            resample_case: u8::MAX,
            resample_nparts: 0,
            resample_xy: empty_resample_xy(),
        })
    }
}
