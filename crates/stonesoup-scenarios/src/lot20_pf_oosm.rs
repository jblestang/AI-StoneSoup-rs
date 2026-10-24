#[allow(dead_code, clippy::excessive_precision, clippy::approx_constant)]
mod fixtures {
    include!("lot20_pf_oosm_fixtures.rs");
}
use fixtures::*;

use crate::registry::ScenarioError;
use crate::scenario::{empty_det_flags, empty_det_xy, empty_resample_xy, ViewerSnapshot};

pub const MAX_LOT20_SCANS: usize = LOT20_NUM_SCANS;
pub const MAX_LOT20_STEPS: usize = LOT20_NUM_SCANS + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lot20PfOosmMode {
    Oosm,
    IgnoreDelayed,
}

pub struct Lot20PfOosmScenario {
    mode: Lot20PfOosmMode,
    step: usize,
    mean: [f64; 4],
}

fn reference_table(mode: Lot20PfOosmMode) -> &'static [[f64; 4]] {
    match mode {
        Lot20PfOosmMode::Oosm => &LOT20_OOSM_TRACK,
        Lot20PfOosmMode::IgnoreDelayed => &LOT20_IGNORE_TRACK,
    }
}

impl Lot20PfOosmScenario {
    pub fn tutorial_defaults(mode: Lot20PfOosmMode) -> Result<Self, ScenarioError> {
        Ok(Self {
            mode,
            step: 0,
            mean: LOT20_PRIOR_MEAN,
        })
    }

    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        *self = Self::tutorial_defaults(self.mode)?;
        Ok(())
    }

    pub fn truth_at(truth_idx: usize) -> Result<[f64; 4], ScenarioError> {
        LOT20_TRUTH
            .get(truth_idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }

    pub fn reference_mean_at(
        arrival_idx: usize,
        mode: Lot20PfOosmMode,
    ) -> Result<[f64; 4], ScenarioError> {
        reference_table(mode)
            .get(arrival_idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }

    pub fn step(&mut self) -> Result<bool, ScenarioError> {
        if self.step >= MAX_LOT20_STEPS {
            return Ok(false);
        }
        if self.step == 0 {
            self.step = 1;
            return Ok(self.step < MAX_LOT20_STEPS);
        }
        let arrival = self.step - 1;
        if arrival >= LOT20_NUM_SCANS {
            return Ok(false);
        }
        self.mean = Self::reference_mean_at(arrival, self.mode)?;
        self.step += 1;
        Ok(self.step < MAX_LOT20_STEPS)
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
            self.step.saturating_sub(1).min(LOT20_NUM_SCANS)
        }
    }

    pub fn viewer_snapshot(&self) -> Result<ViewerSnapshot, ScenarioError> {
        let idx = if self.arrival_index() == 0 {
            0
        } else {
            self.arrival_index().saturating_sub(1)
        };
        let truth_idx = LOT20_ARRIVAL_TRUTH_INDEX.get(idx).copied().unwrap_or(0);
        let truth = LOT20_TRUTH[truth_idx.min(LOT20_NUM_SCANS - 1)];
        let m = self.mean;
        Ok(ViewerSnapshot {
            step: self.step,
            max_steps: MAX_LOT20_STEPS,
            truth_xy: (truth[0], truth[2]),
            estimate_xy: (m[0], m[2]),
            measurement_xy: (truth[0], truth[2]),
            bearing_range: None,
            ellipse_semi: None,
            ellipse_angle_rad: 0.0,
            rmse: 0.0,
            finished: self.step >= MAX_LOT20_STEPS,
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
