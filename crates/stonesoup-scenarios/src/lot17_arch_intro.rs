#[allow(dead_code)]
mod fixtures {
    include!("lot17_arch_intro_fixtures.rs");
}
use fixtures::*;

use crate::registry::ScenarioError;
use crate::scenario::{empty_det_flags, empty_det_xy, empty_resample_xy, ViewerSnapshot};

/// Initial + one display step (tutorial has no time simulation).
pub const MAX_LOT17_STEPS: usize = 2;
pub const LOT17_INTRO_EDGE_COUNT: usize = LOT17_NUM_EDGES;

pub struct Lot17ArchIntroScenario {
    step: usize,
}

impl Lot17ArchIntroScenario {
    pub fn tutorial_defaults() -> Result<Self, ScenarioError> {
        Ok(Self { step: 0 })
    }

    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        *self = Self::tutorial_defaults()?;
        Ok(())
    }

    pub fn step(&mut self) -> Result<bool, ScenarioError> {
        if self.step >= MAX_LOT17_STEPS {
            return Ok(false);
        }
        if self.step == 0 {
            self.step = 1;
            return Ok(true);
        }
        self.step += 1;
        Ok(self.step < MAX_LOT17_STEPS)
    }

    pub fn run_to_end(&mut self) -> Result<(), ScenarioError> {
        while self.step()? {}
        Ok(())
    }

    pub fn viewer_snapshot(&self) -> Result<ViewerSnapshot, ScenarioError> {
        let show = self.step >= 1;
        let (a, b, c) = if show {
            (LOT17_NODE_A, LOT17_NODE_B, LOT17_NODE_C)
        } else {
            ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
        };
        Ok(ViewerSnapshot {
            step: self.step,
            max_steps: MAX_LOT17_STEPS,
            truth_xy: (a[0], a[1]),
            estimate_xy: (b[0], b[1]),
            measurement_xy: (a[0], a[1]),
            bearing_range: None,
            ellipse_semi: None,
            ellipse_angle_rad: 0.0,
            rmse: LOT17_NUM_EDGES as f64,
            finished: self.step >= MAX_LOT17_STEPS,
            det_count: 0,
            det_xy: empty_det_xy(),
            det_is_true: empty_det_flags(),
            associated_xy: None,
            truth_b_xy: if show { Some((c[0], c[1])) } else { None },
            estimate_b_xy: None,
            resample_case: u8::MAX,
            resample_nparts: 0,
            resample_xy: empty_resample_xy(),
        })
    }
}
