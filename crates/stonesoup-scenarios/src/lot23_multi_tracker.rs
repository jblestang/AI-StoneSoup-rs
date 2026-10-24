#[allow(dead_code, clippy::excessive_precision, clippy::approx_constant)]
mod fixtures {
    include!("lot23_multi_tracker_fixtures.rs");
}
use fixtures::*;

use crate::lot3_clutter::MAX_DETS_PER_STEP;
use crate::registry::ScenarioError;
use crate::scenario::{empty_det_flags, empty_det_xy, empty_resample_xy, ViewerSnapshot};

pub const MAX_LOT23_TRACKS: usize = LOT23_NUM_TRUTHS;
pub const MAX_LOT23_STEPS: usize = LOT23_NUM_STEPS + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lot23MultiTrackerMode {
    Ekf,
    Ukf,
    Pf,
    Esif,
}

pub struct Lot23MultiTrackerScenario {
    mode: Lot23MultiTrackerMode,
    step: usize,
    track_buf: [[f64; 4]; MAX_LOT23_TRACKS],
    track_count: usize,
}

fn mode_tables(mode: Lot23MultiTrackerMode) -> (&'static [usize], &'static [[f64; 4]]) {
    match mode {
        Lot23MultiTrackerMode::Ekf => (&LOT23_EKF_TRACK_OFFSETS, &LOT23_EKF_TRACK_MEANS),
        Lot23MultiTrackerMode::Ukf => (&LOT23_UKF_TRACK_OFFSETS, &LOT23_UKF_TRACK_MEANS),
        Lot23MultiTrackerMode::Pf => (&LOT23_PF_TRACK_OFFSETS, &LOT23_PF_TRACK_MEANS),
        Lot23MultiTrackerMode::Esif => (&LOT23_ESIF_TRACK_OFFSETS, &LOT23_ESIF_TRACK_MEANS),
    }
}

fn lot23_bearing_range_to_xy(bearing: f64, range: f64) -> (f64, f64) {
    (
        LOT23_SENSOR_X + range * libm::cos(bearing),
        LOT23_SENSOR_Y + range * libm::sin(bearing),
    )
}

impl Lot23MultiTrackerScenario {
    pub fn tutorial_defaults(mode: Lot23MultiTrackerMode) -> Result<Self, ScenarioError> {
        Ok(Self {
            mode,
            step: 0,
            track_buf: [[0.0; 4]; MAX_LOT23_TRACKS],
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
        self.track_count = means.len().min(MAX_LOT23_TRACKS);
        for (i, row) in means.iter().enumerate().take(MAX_LOT23_TRACKS) {
            self.track_buf[i] = *row;
        }
        Ok(())
    }

    pub fn reference_track_means_at(
        step: usize,
        mode: Lot23MultiTrackerMode,
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
        let end = offsets.get(idx + 1).copied().unwrap_or(means.len());
        means
            .get(start..end)
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))
    }

    pub fn step(&mut self) -> Result<bool, ScenarioError> {
        if self.step >= MAX_LOT23_STEPS {
            return Ok(false);
        }
        if self.step == 0 {
            self.step = 1;
            self.load_tracks_after_step()?;
            return Ok(self.step < MAX_LOT23_STEPS);
        }
        self.step += 1;
        self.load_tracks_after_step()?;
        Ok(self.step < MAX_LOT23_STEPS)
    }

    pub fn run_to_end(&mut self) -> Result<(), ScenarioError> {
        while self.step()? {}
        Ok(())
    }

    pub fn mode(&self) -> Lot23MultiTrackerMode {
        self.mode
    }

    pub fn sorted_track_means(&self, out: &mut [[f64; 4]; MAX_LOT23_TRACKS]) -> usize {
        let n = self.track_count;
        out[..n].copy_from_slice(&self.track_buf[..n]);
        n
    }

    pub fn viewer_snapshot(&self) -> Result<ViewerSnapshot, ScenarioError> {
        let step = self.step;
        let last_truth = LOT23_NUM_STEPS.saturating_sub(1);
        let idx = if step == 0 {
            0
        } else {
            step.saturating_sub(1).min(last_truth)
        };
        let truth = LOT23_TRUTH
            .get(idx)
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        let mut sorted = [[0.0; 4]; MAX_LOT23_TRACKS];
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
        let mut det_xy = empty_det_xy();
        let mut det_is_true = empty_det_flags();
        let det_count = if step == 0 {
            0
        } else {
            let start = LOT23_DET_OFFSETS
                .get(idx)
                .copied()
                .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
            let end = LOT23_DET_OFFSETS
                .get(idx + 1)
                .copied()
                .unwrap_or(LOT23_DETECTIONS.len());
            let count = (end - start).min(MAX_DETS_PER_STEP);
            for j in 0..count {
                let br = LOT23_DETECTIONS[start + j];
                let xy = lot23_bearing_range_to_xy(br[0], br[1]);
                det_xy[j] = [xy.0, xy.1];
                det_is_true[j] = true;
            }
            count
        };
        let (measurement_xy, bearing_range) = if det_count > 0 {
            let br = LOT23_DETECTIONS[LOT23_DET_OFFSETS[idx]];
            let xy = lot23_bearing_range_to_xy(br[0], br[1]);
            (xy, Some((br[0], br[1])))
        } else {
            (estimate_xy, None)
        };
        Ok(ViewerSnapshot {
            step,
            max_steps: MAX_LOT23_STEPS,
            truth_xy: (t0[0], t0[2]),
            estimate_xy,
            measurement_xy,
            bearing_range,
            ellipse_semi: None,
            ellipse_angle_rad: 0.0,
            rmse: 0.0,
            finished: step >= MAX_LOT23_STEPS,
            det_count: det_count as u8,
            det_xy,
            det_is_true,
            associated_xy: None,
            truth_b_xy: Some((t1[0], t1[2])),
            estimate_b_xy,
            resample_case: u8::MAX,
            resample_nparts: 0,
            resample_xy: empty_resample_xy(),
        })
    }
}
