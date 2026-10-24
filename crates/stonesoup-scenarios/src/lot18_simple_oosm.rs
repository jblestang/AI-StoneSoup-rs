#[allow(dead_code, clippy::excessive_precision, clippy::approx_constant)]
mod fixtures {
    include!("lot18_simple_oosm_fixtures.rs");
}
use fixtures::*;

use crate::registry::ScenarioError;
use crate::scenario::{empty_det_flags, empty_det_xy, empty_resample_xy, ViewerSnapshot};

pub const MAX_LOT18_SCANS: usize = LOT18_NUM_SCANS;
pub const MAX_LOT18_STEPS: usize = LOT18_NUM_SCANS + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lot18OosmMode {
    Naive,
    FixedLag,
}

pub struct Lot18SimpleOosmScenario {
    mode: Lot18OosmMode,
    step: usize,
    mean: [f64; 4],
}

fn reference_table(mode: Lot18OosmMode) -> &'static [[f64; 4]] {
    match mode {
        Lot18OosmMode::Naive => &LOT18_NAIVE_TRACK,
        Lot18OosmMode::FixedLag => &LOT18_OOSM_TRACK,
    }
}

impl Lot18SimpleOosmScenario {
    pub fn tutorial_defaults(mode: Lot18OosmMode) -> Result<Self, ScenarioError> {
        Ok(Self {
            mode,
            step: 0,
            mean: LOT18_PRIOR_MEAN,
        })
    }

    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        *self = Self::tutorial_defaults(self.mode)?;
        Ok(())
    }

    pub fn truth_at(truth_idx: usize) -> Result<[f64; 4], ScenarioError> {
        LOT18_TRUTH
            .get(truth_idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }

    pub fn reference_mean_at(
        arrival_idx: usize,
        mode: Lot18OosmMode,
    ) -> Result<[f64; 4], ScenarioError> {
        reference_table(mode)
            .get(arrival_idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }

    fn load_after_arrival(&mut self, arrival_idx: usize) -> Result<(), ScenarioError> {
        self.mean = Self::reference_mean_at(arrival_idx, self.mode)?;
        Ok(())
    }

    pub fn step(&mut self) -> Result<bool, ScenarioError> {
        if self.step >= MAX_LOT18_STEPS {
            return Ok(false);
        }
        if self.step == 0 {
            self.step = 1;
            return Ok(self.step < MAX_LOT18_STEPS);
        }
        let arrival = self.step - 1;
        if arrival >= LOT18_NUM_SCANS {
            return Ok(false);
        }
        self.load_after_arrival(arrival)?;
        self.step += 1;
        Ok(self.step < MAX_LOT18_STEPS)
    }

    pub fn run_to_end(&mut self) -> Result<(), ScenarioError> {
        while self.step()? {}
        Ok(())
    }

    pub fn estimate_mean(&self) -> [f64; 4] {
        self.mean
    }

    pub fn arrival_index(&self) -> usize {
        if self.step == 0 {
            0
        } else {
            self.step.saturating_sub(1).min(LOT18_NUM_SCANS)
        }
    }

    pub fn viewer_snapshot(&self) -> Result<ViewerSnapshot, ScenarioError> {
        let idx = if self.arrival_index() == 0 {
            0
        } else {
            self.arrival_index().saturating_sub(1)
        };
        let truth_idx = LOT18_ARRIVAL_TRUTH_INDEX.get(idx).copied().unwrap_or(0);
        let truth = LOT18_TRUTH[truth_idx.min(LOT18_NUM_SCANS - 1)];
        let meas = LOT18_MEAS_XY[truth_idx.min(LOT18_NUM_SCANS - 1)];
        let m = self.mean;
        Ok(ViewerSnapshot {
            step: self.step,
            max_steps: MAX_LOT18_STEPS,
            truth_xy: (truth[0], truth[2]),
            estimate_xy: (m[0], m[2]),
            measurement_xy: (meas[0], meas[1]),
            bearing_range: None,
            ellipse_semi: None,
            ellipse_angle_rad: 0.0,
            rmse: 0.0,
            finished: self.step >= MAX_LOT18_STEPS,
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
