#[allow(dead_code, clippy::excessive_precision, clippy::approx_constant)]
mod fixtures {
    include!("lot16_data_incest_fixtures.rs");
}
use fixtures::*;

use crate::registry::ScenarioError;
use crate::scenario::{empty_det_flags, empty_det_xy, empty_resample_xy, ViewerSnapshot};

pub const MAX_LOT16_TRACKS: usize = LOT16_NUM_TRUTHS;
pub const MAX_LOT16_STEPS: usize = LOT16_NUM_STEPS + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lot16DataIncestMode {
    NonHierarchical,
    Hierarchical,
}

pub struct Lot16DataIncestScenario {
    mode: Lot16DataIncestMode,
    step: usize,
    track_buf: [[f64; 4]; MAX_LOT16_TRACKS],
    track_count: usize,
}

fn mode_tables(mode: Lot16DataIncestMode) -> (&'static [usize], &'static [[f64; 4]]) {
    match mode {
        Lot16DataIncestMode::NonHierarchical => (&LOT16_NH_TRACK_OFFSETS, &LOT16_NH_TRACK_MEANS),
        Lot16DataIncestMode::Hierarchical => (&LOT16_H_TRACK_OFFSETS, &LOT16_H_TRACK_MEANS),
    }
}

fn mode_cov_table(mode: Lot16DataIncestMode) -> &'static [f64] {
    match mode {
        Lot16DataIncestMode::NonHierarchical => &LOT16_NH_MEAN_COV_TRACE,
        Lot16DataIncestMode::Hierarchical => &LOT16_H_MEAN_COV_TRACE,
    }
}

impl Lot16DataIncestScenario {
    pub fn tutorial_defaults(mode: Lot16DataIncestMode) -> Result<Self, ScenarioError> {
        Ok(Self {
            mode,
            step: 0,
            track_buf: [[0.0; 4]; MAX_LOT16_TRACKS],
            track_count: 0,
        })
    }

    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        *self = Self::tutorial_defaults(self.mode)?;
        Ok(())
    }

    fn load_tracks_after_step(&mut self) -> Result<(), ScenarioError> {
        let means = if self.step == 0 {
            &[] as &[[f64; 4]]
        } else {
            Self::reference_track_means_at(self.step - 1, self.mode)?
        };
        self.track_count = means.len().min(MAX_LOT16_TRACKS);
        for (i, row) in means.iter().enumerate().take(MAX_LOT16_TRACKS) {
            self.track_buf[i] = *row;
        }
        Ok(())
    }

    pub fn reference_track_means_at(
        step: usize,
        mode: Lot16DataIncestMode,
    ) -> Result<&'static [[f64; 4]], ScenarioError> {
        if step == 0 {
            return Ok(&[]);
        }
        let (offsets, means) = mode_tables(mode);
        let idx = step - 1;
        let start = offsets
            .get(idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        let end = offsets
            .get(idx + 1)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        means
            .get(start..end)
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }

    pub fn reference_mean_cov_trace_at(
        step: usize,
        mode: Lot16DataIncestMode,
    ) -> Result<f64, ScenarioError> {
        if step == 0 {
            return Ok(f64::NAN);
        }
        let idx = step - 1;
        mode_cov_table(mode)
            .get(idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }

    fn current_mean_cov_trace(&self) -> f64 {
        if self.step == 0 {
            return 0.0;
        }
        Self::reference_mean_cov_trace_at(self.step - 1, self.mode).unwrap_or(f64::NAN)
    }

    pub fn step(&mut self) -> Result<bool, ScenarioError> {
        if self.step >= MAX_LOT16_STEPS {
            return Ok(false);
        }
        if self.step == 0 {
            self.step = 1;
            self.load_tracks_after_step()?;
            return Ok(self.step < MAX_LOT16_STEPS);
        }
        if self.step >= MAX_LOT16_STEPS {
            return Ok(false);
        }
        self.step += 1;
        self.load_tracks_after_step()?;
        Ok(self.step < MAX_LOT16_STEPS)
    }

    pub fn run_to_end(&mut self) -> Result<(), ScenarioError> {
        while self.step()? {}
        Ok(())
    }

    pub fn mode(&self) -> Lot16DataIncestMode {
        self.mode
    }

    pub fn sorted_track_means(&self, out: &mut [[f64; 4]; MAX_LOT16_TRACKS]) -> usize {
        let n = self.track_count;
        out[..n].copy_from_slice(&self.track_buf[..n]);
        n
    }

    pub fn viewer_snapshot(&self) -> Result<ViewerSnapshot, ScenarioError> {
        let step = self.step;
        let last_truth = LOT16_NUM_STEPS.saturating_sub(1);
        let idx = if step == 0 {
            0
        } else {
            step.saturating_sub(1).min(last_truth)
        };
        let truth = LOT16_TRUTH
            .get(idx)
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        let mut sorted = [[0.0; 4]; MAX_LOT16_TRACKS];
        let n = self.sorted_track_means(&mut sorted);
        let estimate_xy = if n > 0 {
            (sorted[0][0], sorted[0][2])
        } else {
            (0.0, 0.0)
        };
        let estimate_b_xy = if n > 1 {
            Some((sorted[1][0], sorted[1][2]))
        } else {
            None
        };
        let t0 = truth[0];
        let t1 = truth[1];
        let trace = self.current_mean_cov_trace();
        let rmse = if trace.is_nan() { 0.0 } else { trace };
        Ok(ViewerSnapshot {
            step,
            max_steps: MAX_LOT16_STEPS,
            truth_xy: (t0[0], t0[2]),
            estimate_xy,
            measurement_xy: estimate_xy,
            bearing_range: None,
            ellipse_semi: None,
            ellipse_angle_rad: 0.0,
            rmse,
            finished: step >= MAX_LOT16_STEPS,
            det_count: 0,
            det_xy: empty_det_xy(),
            det_is_true: empty_det_flags(),
            associated_xy: None,
            truth_b_xy: Some((t1[0], t1[2])),
            estimate_b_xy,
            resample_case: u8::MAX,
            resample_nparts: 0,
            resample_xy: empty_resample_xy(),
        })
    }
}
