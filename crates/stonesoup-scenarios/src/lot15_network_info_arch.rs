#[allow(dead_code, clippy::excessive_precision, clippy::approx_constant)]
mod fixtures {
    include!("lot15_network_info_arch_fixtures.rs");
}
use fixtures::*;

use crate::registry::ScenarioError;
use crate::scenario::{empty_det_flags, empty_det_xy, empty_resample_xy, ViewerSnapshot};

pub const MAX_LOT15_TRACKS: usize = LOT15_NUM_TRUTHS;
/// One initial step plus one frame per exported architecture timestep.
pub const MAX_LOT15_STEPS: usize = LOT15_NUM_STEPS + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lot15ArchitectureMode {
    Network,
    Information,
    NetworkEdgeRemoved,
    InformationEdgeRemoved,
}

pub struct Lot15NetworkInfoArchScenario {
    mode: Lot15ArchitectureMode,
    step: usize,
    track_buf: [[f64; 4]; MAX_LOT15_TRACKS],
    track_count: usize,
}

fn mode_tables(mode: Lot15ArchitectureMode) -> (&'static [usize], &'static [[f64; 4]]) {
    match mode {
        Lot15ArchitectureMode::Network => {
            (&LOT15_NETWORK_TRACK_OFFSETS, &LOT15_NETWORK_TRACK_MEANS)
        }
        Lot15ArchitectureMode::Information => (
            &LOT15_INFORMATION_TRACK_OFFSETS,
            &LOT15_INFORMATION_TRACK_MEANS,
        ),
        Lot15ArchitectureMode::NetworkEdgeRemoved => (
            &LOT15_NETWORK_RM_TRACK_OFFSETS,
            &LOT15_NETWORK_RM_TRACK_MEANS,
        ),
        Lot15ArchitectureMode::InformationEdgeRemoved => (
            &LOT15_INFORMATION_RM_TRACK_OFFSETS,
            &LOT15_INFORMATION_RM_TRACK_MEANS,
        ),
    }
}

impl Lot15NetworkInfoArchScenario {
    pub fn tutorial_defaults(mode: Lot15ArchitectureMode) -> Result<Self, ScenarioError> {
        Ok(Self {
            mode,
            step: 0,
            track_buf: [[0.0; 4]; MAX_LOT15_TRACKS],
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
        self.track_count = means.len().min(MAX_LOT15_TRACKS);
        for (i, row) in means.iter().enumerate().take(MAX_LOT15_TRACKS) {
            self.track_buf[i] = *row;
        }
        Ok(())
    }

    pub fn reference_track_means_at(
        step: usize,
        mode: Lot15ArchitectureMode,
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

    pub fn step(&mut self) -> Result<bool, ScenarioError> {
        if self.step >= MAX_LOT15_STEPS {
            return Ok(false);
        }
        if self.step == 0 {
            self.step = 1;
            self.load_tracks_after_step()?;
            return Ok(self.step < MAX_LOT15_STEPS);
        }
        if self.step >= MAX_LOT15_STEPS {
            return Ok(false);
        }
        self.step += 1;
        self.load_tracks_after_step()?;
        Ok(self.step < MAX_LOT15_STEPS)
    }

    pub fn run_to_end(&mut self) -> Result<(), ScenarioError> {
        while self.step()? {}
        Ok(())
    }

    pub fn mode(&self) -> Lot15ArchitectureMode {
        self.mode
    }

    pub fn sorted_track_means(&self, out: &mut [[f64; 4]; MAX_LOT15_TRACKS]) -> usize {
        let n = self.track_count;
        out[..n].copy_from_slice(&self.track_buf[..n]);
        n
    }

    pub fn viewer_snapshot(&self) -> Result<ViewerSnapshot, ScenarioError> {
        let step = self.step;
        let idx = if step == 0 { 0 } else { step.saturating_sub(1) };
        let truth = LOT15_TRUTH
            .get(idx)
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        let mut sorted = [[0.0; 4]; MAX_LOT15_TRACKS];
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
        Ok(ViewerSnapshot {
            step,
            max_steps: MAX_LOT15_STEPS,
            truth_xy: (t0[0], t0[2]),
            estimate_xy,
            measurement_xy: estimate_xy,
            bearing_range: None,
            ellipse_semi: None,
            ellipse_angle_rad: 0.0,
            rmse: 0.0,
            finished: step >= MAX_LOT15_STEPS,
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
