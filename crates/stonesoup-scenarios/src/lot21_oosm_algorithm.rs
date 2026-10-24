#[allow(dead_code, clippy::excessive_precision, clippy::approx_constant)]
mod fixtures {
    include!("lot21_oosm_algorithm_fixtures.rs");
}
use fixtures::*;

use crate::registry::ScenarioError;
use crate::scenario::{empty_det_flags, empty_det_xy, empty_resample_xy, ViewerSnapshot};

pub const LOT21_NUM_SCANS: usize = LOT21_NUM_STEPS;
pub const MAX_LOT21_STEPS: usize = LOT21_NUM_STEPS + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lot21OosmAlgorithmMode {
    OosmTarget0,
    OosmTarget1,
    IgnoreTarget0,
    IgnoreTarget1,
}

impl Lot21OosmAlgorithmMode {
    fn target_index(self) -> usize {
        match self {
            Self::OosmTarget0 | Self::IgnoreTarget0 => 0,
            Self::OosmTarget1 | Self::IgnoreTarget1 => 1,
        }
    }
}

pub struct Lot21OosmAlgorithmScenario {
    mode: Lot21OosmAlgorithmMode,
    step: usize,
    mean: [f64; 4],
}

fn reference_table(mode: Lot21OosmAlgorithmMode) -> &'static [[f64; 4]] {
    match mode {
        Lot21OosmAlgorithmMode::OosmTarget0 => &LOT21_OOSM_T0_TRACK,
        Lot21OosmAlgorithmMode::OosmTarget1 => &LOT21_OOSM_T1_TRACK,
        Lot21OosmAlgorithmMode::IgnoreTarget0 => &LOT21_NOOOSM_T0_TRACK,
        Lot21OosmAlgorithmMode::IgnoreTarget1 => &LOT21_NOOOSM_T1_TRACK,
    }
}

fn truth_table(target: usize) -> &'static [[f64; 4]] {
    if target == 0 {
        &LOT21_TRUTH_T0
    } else {
        &LOT21_TRUTH_T1
    }
}

impl Lot21OosmAlgorithmScenario {
    pub fn tutorial_defaults(mode: Lot21OosmAlgorithmMode) -> Result<Self, ScenarioError> {
        Ok(Self {
            mode,
            step: 0,
            mean: reference_table(mode)[0],
        })
    }

    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        *self = Self::tutorial_defaults(self.mode)?;
        Ok(())
    }

    pub fn reference_mean_at(
        scan_idx: usize,
        mode: Lot21OosmAlgorithmMode,
    ) -> Result<[f64; 4], ScenarioError> {
        reference_table(mode)
            .get(scan_idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }

    pub fn truth_at(target: usize, scan_idx: usize) -> Result<[f64; 4], ScenarioError> {
        truth_table(target)
            .get(scan_idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }

    pub fn step(&mut self) -> Result<bool, ScenarioError> {
        if self.step >= MAX_LOT21_STEPS {
            return Ok(false);
        }
        if self.step == 0 {
            self.step = 1;
            return Ok(self.step < MAX_LOT21_STEPS);
        }
        let idx = self.step - 1;
        if idx >= LOT21_NUM_STEPS {
            return Ok(false);
        }
        self.mean = Self::reference_mean_at(idx, self.mode)?;
        self.step += 1;
        Ok(self.step < MAX_LOT21_STEPS)
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
            self.step.saturating_sub(1).min(LOT21_NUM_STEPS)
        }
    }

    pub fn viewer_snapshot(&self) -> Result<ViewerSnapshot, ScenarioError> {
        let ti = self.mode.target_index();
        let scan = self.scan_index().min(LOT21_NUM_STEPS.saturating_sub(1));
        let truth = truth_table(ti)[scan];
        let m = self.mean;
        Ok(ViewerSnapshot {
            step: self.step,
            max_steps: MAX_LOT21_STEPS,
            truth_xy: (truth[0], truth[2]),
            estimate_xy: (m[0], m[2]),
            measurement_xy: (truth[0], truth[2]),
            bearing_range: None,
            ellipse_semi: None,
            ellipse_angle_rad: 0.0,
            rmse: 0.0,
            finished: self.step >= MAX_LOT21_STEPS,
            det_count: 0,
            det_xy: empty_det_xy(),
            det_is_true: empty_det_flags(),
            associated_xy: None,
            truth_b_xy: {
                let tb = truth_table(1 - ti)[scan];
                Some((tb[0], tb[2]))
            },
            estimate_b_xy: None,
            resample_case: u8::MAX,
            resample_nparts: 0,
            resample_xy: empty_resample_xy(),
        })
    }
}
