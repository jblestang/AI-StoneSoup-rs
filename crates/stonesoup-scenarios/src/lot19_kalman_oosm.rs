#[allow(dead_code, clippy::excessive_precision, clippy::approx_constant)]
mod fixtures {
    include!("lot19_kalman_oosm_fixtures.rs");
}
use fixtures::*;

use crate::registry::ScenarioError;
use crate::scenario::{empty_det_flags, empty_det_xy, empty_resample_xy, ViewerSnapshot};

pub const MAX_LOT19_LOOPS: usize = LOT19_NUM_LOOPS;
pub const MAX_LOT19_STEPS: usize = LOT19_NUM_LOOPS + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lot19KalmanOosmMode {
    Tracker1,
    Tracker2,
    Tracker3,
}

pub struct Lot19KalmanOosmScenario {
    mode: Lot19KalmanOosmMode,
    step: usize,
    mean: [f64; 4],
}

fn reference_table(mode: Lot19KalmanOosmMode) -> &'static [[f64; 4]] {
    match mode {
        Lot19KalmanOosmMode::Tracker1 => &LOT19_T1_TRACK,
        Lot19KalmanOosmMode::Tracker2 => &LOT19_T2_TRACK,
        Lot19KalmanOosmMode::Tracker3 => &LOT19_T3_TRACK,
    }
}

impl Lot19KalmanOosmScenario {
    pub fn tutorial_defaults(mode: Lot19KalmanOosmMode) -> Result<Self, ScenarioError> {
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
        loop_idx: usize,
        mode: Lot19KalmanOosmMode,
    ) -> Result<[f64; 4], ScenarioError> {
        reference_table(mode)
            .get(loop_idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }

    pub fn step(&mut self) -> Result<bool, ScenarioError> {
        if self.step >= MAX_LOT19_STEPS {
            return Ok(false);
        }
        if self.step == 0 {
            self.step = 1;
            return Ok(self.step < MAX_LOT19_STEPS);
        }
        let idx = self.step - 1;
        if idx >= LOT19_NUM_LOOPS {
            return Ok(false);
        }
        self.mean = Self::reference_mean_at(idx, self.mode)?;
        self.step += 1;
        Ok(self.step < MAX_LOT19_STEPS)
    }

    pub fn run_to_end(&mut self) -> Result<(), ScenarioError> {
        while self.step()? {}
        Ok(())
    }

    pub fn estimate_mean(&self) -> [f64; 4] {
        self.mean
    }

    pub fn loop_index(&self) -> usize {
        if self.step == 0 {
            0
        } else {
            self.step.saturating_sub(1).min(LOT19_NUM_LOOPS)
        }
    }

    pub fn viewer_snapshot(&self) -> Result<ViewerSnapshot, ScenarioError> {
        let li = self.loop_index();
        let truth_idx = li.min(LOT19_NUM_TRUTHS - 1);
        let truth = LOT19_TRUTH[truth_idx];
        let m = self.mean;
        Ok(ViewerSnapshot {
            step: self.step,
            max_steps: MAX_LOT19_STEPS,
            truth_xy: (truth[0], truth[2]),
            estimate_xy: (m[0], m[2]),
            measurement_xy: (truth[0], truth[2]),
            bearing_range: None,
            ellipse_semi: None,
            ellipse_angle_rad: 0.0,
            rmse: 0.0,
            finished: self.step >= MAX_LOT19_STEPS,
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
