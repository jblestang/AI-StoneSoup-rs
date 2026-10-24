#[allow(dead_code, clippy::excessive_precision, clippy::approx_constant)]
mod fixtures {
    include!("lot24_ensemble_filter_fixtures.rs");
}
use fixtures::*;

use crate::registry::ScenarioError;
use crate::scenario::{empty_det_flags, empty_det_xy, empty_resample_xy, ViewerSnapshot};

pub const MAX_LOT24_STEPS: usize = LOT24_NUM_STEPS + 1;

fn lot24_bearing_range_to_xy(bearing: f64, range: f64) -> (f64, f64) {
    (
        LOT24_SENSOR_X + range * libm::cos(bearing),
        LOT24_SENSOR_Y + range * libm::sin(bearing),
    )
}

pub struct Lot24EnsembleFilterScenario {
    step: usize,
    mean: [f64; 4],
}

impl Lot24EnsembleFilterScenario {
    pub fn tutorial_defaults() -> Result<Self, ScenarioError> {
        Ok(Self {
            step: 0,
            mean: LOT24_TRACK[0],
        })
    }

    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        *self = Self::tutorial_defaults()?;
        Ok(())
    }

    pub fn reference_mean_at(step_idx: usize) -> Result<[f64; 4], ScenarioError> {
        LOT24_TRACK
            .get(step_idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }

    pub fn truth_at(step_idx: usize) -> Result<[f64; 4], ScenarioError> {
        LOT24_TRUTH
            .get(step_idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }

    pub fn step(&mut self) -> Result<bool, ScenarioError> {
        if self.step >= MAX_LOT24_STEPS {
            return Ok(false);
        }
        if self.step == 0 {
            self.step = 1;
            return Ok(self.step < MAX_LOT24_STEPS);
        }
        let idx = self.step - 1;
        if idx >= LOT24_NUM_STEPS {
            return Ok(false);
        }
        self.mean = Self::reference_mean_at(idx)?;
        self.step += 1;
        Ok(self.step < MAX_LOT24_STEPS)
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
            self.step.saturating_sub(1).min(LOT24_NUM_STEPS)
        }
    }

    pub fn viewer_snapshot(&self) -> Result<ViewerSnapshot, ScenarioError> {
        let idx = self.scan_index().min(LOT24_NUM_STEPS.saturating_sub(1));
        let truth = LOT24_TRUTH[idx];
        let br = LOT24_MEAS_BR[idx];
        let meas_xy = lot24_bearing_range_to_xy(br[0], br[1]);
        let m = self.mean;
        Ok(ViewerSnapshot {
            step: self.step,
            max_steps: MAX_LOT24_STEPS,
            truth_xy: (truth[0], truth[2]),
            estimate_xy: (m[0], m[2]),
            measurement_xy: meas_xy,
            bearing_range: Some((br[0], br[1])),
            ellipse_semi: None,
            ellipse_angle_rad: 0.0,
            rmse: 0.0,
            finished: self.step >= MAX_LOT24_STEPS,
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
