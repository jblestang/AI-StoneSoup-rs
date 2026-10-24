//! Stone Soup GM-PHD filter tutorial (`GMPHDTutorial.py`, seed 42).

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

#[path = "lot7_gm_phd_fixtures.rs"]
#[allow(dead_code, clippy::large_const_arrays, clippy::excessive_precision)]
mod lot7_gm_phd_fixtures;
use lot7_gm_phd_fixtures::*;
pub use lot7_gm_phd_fixtures::{LOT7_NUM_STEPS, LOT7_REDUCED};

use stonesoup_core::{
    gm_phd_append_detection_update, gm_phd_append_missed_update, gm_phd_collect_missed,
    gm_phd_hypotheses_for_detection, gm_phd_reduce, ConstantVelocity2d, Detection, Duration,
    GmPhdConfig, KalmanPredictor, KalmanUpdater, KalmanWorkspace, LinearGaussianMeasurement, Mat,
    PhdComponent, PhdHypothesis, PhdTag, SensorId, Timestamp, MAX_GMPHD_COMPONENTS,
    MAX_GMPHD_DETECTIONS, MAX_GMPHD_UPDATED,
};

use crate::lot3_clutter::MAX_DETS_PER_STEP;
use crate::registry::ScenarioError;
use crate::scenario::{empty_det_flags, empty_det_xy, ViewerSnapshot};

pub const MAX_LOT7_STEPS: usize = LOT7_NUM_STEPS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lot7StepOutcome {
    Stepped,
    Finished,
}

pub struct Lot7GmPhdScenario {
    predictor: KalmanPredictor,
    updater: KalmanUpdater,
    ws: KalmanWorkspace,
    config: GmPhdConfig,
    components: [PhdComponent; MAX_GMPHD_COMPONENTS],
    n_components: usize,
    birth: PhdComponent,
    step: usize,
    finished: bool,
    /// Reused each step; large arrays are heap-backed for WASM stack limits.
    step_bufs: Box<Lot7GmPhdStepBuffers>,
}

struct Lot7GmPhdStepBuffers {
    current: [PhdComponent; MAX_GMPHD_COMPONENTS],
    missed: [PhdHypothesis; MAX_GMPHD_COMPONENTS],
    group: [PhdHypothesis; MAX_GMPHD_COMPONENTS],
    updated: Vec<PhdComponent>,
    merge_pool: Vec<PhdComponent>,
}

impl Lot7GmPhdScenario {
    pub fn tutorial_defaults() -> Result<Self, ScenarioError> {
        let transition = ConstantVelocity2d::new(0.3, 0.3).map_err(ScenarioError::Core)?;
        let predictor = KalmanPredictor::new(transition);
        let mut r = Mat::<2, 2>::zeros();
        r.set(0, 0, 0.75).map_err(ScenarioError::Core)?;
        r.set(1, 1, 0.75).map_err(ScenarioError::Core)?;
        let measurement = LinearGaussianMeasurement::position_2d(r).map_err(ScenarioError::Core)?;
        let updater = KalmanUpdater::new(measurement);
        let config = GmPhdConfig {
            prob_detection: LOT7_PROB_DETECTION,
            prob_survival: 1.0 - LOT7_DEATH_PROB,
            clutter_spatial_density: LOT7_CLUTTER_DENSITY,
            missed_distance: LOT7_MISSED_DISTANCE,
            merge_threshold_sq: LOT7_MERGE_THRESHOLD,
            prune_threshold: LOT7_PRUNE_THRESHOLD,
        };
        let mut components = [PhdComponent {
            state: stonesoup_core::GaussianState::new_from_array(
                [0.0; 4],
                Mat::zeros(),
                Timestamp::ZERO,
            )
            .map_err(ScenarioError::Core)?,
            weight: 0.0,
            tag: PhdTag::Untagged,
        }; MAX_GMPHD_COMPONENTS];
        let mut n_components = 0usize;
        for truth in LOT7_START_TRUTHS {
            let mut cov = Mat::<4, 4>::zeros();
            for (i, &v) in LOT7_INIT_COV_DIAG.iter().enumerate() {
                cov.set(i, i, v).map_err(ScenarioError::Core)?;
            }
            let state = stonesoup_core::GaussianState::new_from_array(truth, cov, Timestamp::ZERO)
                .map_err(ScenarioError::Core)?;
            components[n_components] = PhdComponent {
                state,
                weight: LOT7_INIT_WEIGHT,
                tag: PhdTag::Birth,
            };
            n_components += 1;
        }
        let mut birth_cov = Mat::<4, 4>::zeros();
        for (i, &v) in LOT7_BIRTH_COV_DIAG.iter().enumerate() {
            birth_cov.set(i, i, v).map_err(ScenarioError::Core)?;
        }
        let birth_state =
            stonesoup_core::GaussianState::new_from_array([0.0; 4], birth_cov, Timestamp::ZERO)
                .map_err(ScenarioError::Core)?;
        let birth = PhdComponent {
            state: birth_state,
            weight: LOT7_BIRTH_WEIGHT,
            tag: PhdTag::Birth,
        };
        let placeholder = PhdComponent {
            state: birth_state,
            weight: 0.0,
            tag: PhdTag::Untagged,
        };
        Ok(Self {
            predictor,
            updater,
            ws: KalmanWorkspace {
                f: Mat::zeros(),
                q: Mat::zeros(),
                h: Mat::zeros(),
                r: Mat::zeros(),
            },
            config,
            components,
            n_components,
            birth,
            step: 0,
            finished: false,
            step_bufs: Box::new(Lot7GmPhdStepBuffers {
                current: [placeholder; MAX_GMPHD_COMPONENTS],
                missed: [PhdHypothesis {
                    component: placeholder,
                    detection_index: None,
                }; MAX_GMPHD_COMPONENTS],
                group: [PhdHypothesis {
                    component: placeholder,
                    detection_index: None,
                }; MAX_GMPHD_COMPONENTS],
                updated: vec![placeholder; MAX_GMPHD_UPDATED],
                merge_pool: vec![placeholder; MAX_GMPHD_UPDATED],
            }),
        })
    }

    fn timestamp_for(step: usize) -> Timestamp {
        Timestamp::from_nanos((step as i64) * Duration::SEC.0)
    }

    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        *self = Self::tutorial_defaults()?;
        Ok(())
    }

    fn cmp_component(a: &PhdComponent, b: &PhdComponent) -> core::cmp::Ordering {
        match b
            .weight
            .partial_cmp(&a.weight)
            .unwrap_or(core::cmp::Ordering::Equal)
        {
            core::cmp::Ordering::Equal => match a.state.mean.0[0]
                .partial_cmp(&b.state.mean.0[0])
                .unwrap_or(core::cmp::Ordering::Equal)
            {
                core::cmp::Ordering::Equal => a.state.mean.0[2]
                    .partial_cmp(&b.state.mean.0[2])
                    .unwrap_or(core::cmp::Ordering::Equal),
                o => o,
            },
            o => o,
        }
    }

    fn sort_components(components: &mut [PhdComponent], n: usize) {
        for i in 1..n {
            let mut j = i;
            while j > 0
                && Self::cmp_component(&components[j - 1], &components[j])
                    == core::cmp::Ordering::Greater
            {
                components.swap(j, j - 1);
                j -= 1;
            }
        }
    }

    pub fn step(&mut self) -> Result<Lot7StepOutcome, ScenarioError> {
        if self.finished || self.step >= LOT7_NUM_STEPS {
            self.finished = true;
            return Ok(Lot7StepOutcome::Finished);
        }
        let ts = Self::timestamp_for(self.step);
        let meas = &LOT7_MEASUREMENTS[self.step];
        let mut detections = [Detection::new([0.0, 0.0], ts, SensorId(0))
            .map_err(ScenarioError::Core)?; MAX_GMPHD_DETECTIONS];
        for (i, xy) in meas.xy.iter().take(meas.n).enumerate() {
            detections[i] = Detection::new(*xy, ts, SensorId(0)).map_err(ScenarioError::Core)?;
        }
        let bufs = &mut self.step_bufs;
        bufs.current[..self.n_components].copy_from_slice(&self.components[..self.n_components]);
        let mut n_current = self.n_components;
        if n_current >= MAX_GMPHD_COMPONENTS {
            // Drop lowest-weight component to make room for the birth term (Stone Soup adds birth each step).
            let mut min_i = 0usize;
            for i in 1..n_current {
                if bufs.current[i].weight < bufs.current[min_i].weight {
                    min_i = i;
                }
            }
            bufs.current[min_i] = bufs.current[n_current - 1];
            n_current -= 1;
        }
        self.birth.state.timestamp = ts;
        bufs.current[n_current] = self.birth;
        n_current += 1;

        let mut missed_n = 0usize;
        gm_phd_collect_missed(
            &bufs.current,
            n_current,
            &detections,
            meas.n,
            ts,
            self.config.missed_distance,
            &self.predictor,
            &self.updater,
            &mut self.ws,
            &mut bufs.missed,
            &mut missed_n,
        )
        .map_err(ScenarioError::Core)?;

        let mut n_updated = 0usize;
        for di in 0..meas.n {
            let mut group_n = 0usize;
            gm_phd_hypotheses_for_detection(
                &bufs.current,
                n_current,
                &detections,
                meas.n,
                di,
                ts,
                self.config.missed_distance,
                &self.predictor,
                &self.updater,
                &mut self.ws,
                &mut bufs.group,
                &mut group_n,
            )
            .map_err(ScenarioError::Core)?;
            gm_phd_append_detection_update(
                &detections[di],
                &bufs.group,
                group_n,
                &self.config,
                &self.updater,
                &mut self.ws,
                &mut bufs.updated,
                &mut n_updated,
            )
            .map_err(ScenarioError::Core)?;
        }
        gm_phd_append_missed_update(
            &bufs.missed,
            missed_n,
            &self.config,
            &mut bufs.updated,
            &mut n_updated,
        )
        .map_err(ScenarioError::Core)?;

        let mut n_reduced = n_updated;
        gm_phd_reduce(
            &mut bufs.updated,
            &mut n_reduced,
            &self.config,
            &mut bufs.merge_pool,
        )
        .map_err(ScenarioError::Core)?;
        if n_reduced > MAX_GMPHD_COMPONENTS {
            return Err(ScenarioError::Core(stonesoup_core::Error::CapacityExceeded));
        }
        Self::sort_components(&mut bufs.updated, n_reduced);
        self.components[..n_reduced].copy_from_slice(&bufs.updated[..n_reduced]);
        self.n_components = n_reduced;
        self.step += 1;
        if self.step >= LOT7_NUM_STEPS {
            self.finished = true;
            Ok(Lot7StepOutcome::Finished)
        } else {
            Ok(Lot7StepOutcome::Stepped)
        }
    }

    pub fn snapshot_components(&self) -> (&[PhdComponent], usize) {
        (&self.components[..], self.n_components)
    }

    pub fn viewer_snapshot(&self) -> Result<ViewerSnapshot, ScenarioError> {
        let idx = if self.step == 0 { 0 } else { self.step - 1 };
        let meas = &LOT7_MEASUREMENTS[idx];
        let mut det_xy = empty_det_xy();
        let mut det_is_true = empty_det_flags();
        let det_count = meas.n.min(MAX_DETS_PER_STEP) as u8;
        for i in 0..det_count as usize {
            det_xy[i] = meas.xy[i];
            det_is_true[i] = false;
        }
        let (estimate_xy, ellipse_semi) = if self.n_components > 0 {
            let best = self.components[0];
            let (ex, ey) = best.state.position_xy().map_err(ScenarioError::Core)?;
            let (c00, c01, c11) = best
                .state
                .position_cov_block()
                .map_err(ScenarioError::Core)?;
            let (sx, sy) = stonesoup_core::linalg::ellipse_from_cov2(c00, c01, c11, 2.0)
                .map_err(ScenarioError::Core)?;
            ((ex, ey), Some((sx, sy)))
        } else {
            ((0.0, 0.0), None)
        };
        Ok(ViewerSnapshot {
            step: self.step,
            max_steps: LOT7_NUM_STEPS,
            truth_xy: estimate_xy,
            estimate_xy,
            measurement_xy: if meas.n > 0 {
                (meas.xy[0][0], meas.xy[0][1])
            } else {
                (0.0, 0.0)
            },
            bearing_range: None,
            ellipse_semi,
            ellipse_angle_rad: 0.0,
            rmse: 0.0,
            finished: self.finished,
            det_count,
            det_xy,
            det_is_true,
            associated_xy: None,
            truth_b_xy: None,
            estimate_b_xy: None,
            resample_case: u8::MAX,
            resample_nparts: 0,
            resample_xy: crate::scenario::empty_resample_xy(),
        })
    }
}
