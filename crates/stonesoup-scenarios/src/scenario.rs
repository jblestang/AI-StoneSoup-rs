#![allow(clippy::large_enum_variant)]

use crate::lot1_kalman::{Lot1KalmanScenario, Lot1StepOutcome, MAX_LOT1_STEPS};
use crate::lot2_ekf::{
    bearing_range_to_xy, Lot2EkfScenario, Lot2Snapshot, Lot2StepOutcome, MAX_LOT2_STEPS,
};
use crate::lot2_ukf::Lot2UkfScenario;
use crate::lot3_clutter::{
    Lot3ClutterScenario, Lot3StepOutcome, MAX_DETS_PER_STEP, MAX_LOT3_STEPS,
};
use crate::particle_lot2::Lot2ParticleScenario;
use crate::registry::{ScenarioError, ScenarioId};

/// Normalized frame for egui viewer (x–y plot + optional covariance ellipse).
#[derive(Debug, Clone, Copy)]
pub struct ViewerSnapshot {
    pub step: usize,
    pub max_steps: usize,
    pub truth_xy: (f64, f64),
    pub estimate_xy: (f64, f64),
    pub measurement_xy: (f64, f64),
    /// Bearing (rad) and range (m) when the tutorial uses a bearing–range sensor.
    pub bearing_range: Option<(f64, f64)>,
    pub ellipse_semi: Option<(f64, f64)>,
    pub ellipse_angle_rad: f64,
    pub rmse: f64,
    pub finished: bool,
    /// Tutorial 05: all detections at the current step (`det_count` valid entries).
    pub det_count: u8,
    pub det_xy: [[f64; 2]; MAX_DETS_PER_STEP],
    pub det_is_true: [bool; MAX_DETS_PER_STEP],
    /// Tutorial 05: nearest-neighbour pick, if any.
    pub associated_xy: Option<(f64, f64)>,
}

const fn empty_det_xy() -> [[f64; 2]; MAX_DETS_PER_STEP] {
    [[0.0, 0.0]; MAX_DETS_PER_STEP]
}

const fn empty_det_flags() -> [bool; MAX_DETS_PER_STEP] {
    [false; MAX_DETS_PER_STEP]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioStepOutcome {
    Stepped,
    Finished,
}

/// Steppable scenario (strict core); rendering uses [`ViewerSnapshot`] only.
pub trait Scenario {
    fn reset(&mut self) -> Result<(), ScenarioError>;
    fn step(&mut self) -> Result<ScenarioStepOutcome, ScenarioError>;
    fn viewer_snapshot(&self) -> Result<ViewerSnapshot, ScenarioError>;

    fn run_to_end(&mut self) -> Result<(), ScenarioError> {
        while let ScenarioStepOutcome::Stepped = self.step()? {}
        Ok(())
    }
}

pub enum AnyScenario {
    Lot1(Lot1KalmanScenario),
    Lot2Ekf(Lot2EkfScenario),
    Lot2Ukf(Lot2UkfScenario),
    Lot2Pf(Lot2ParticleScenario),
    Lot3(Lot3ClutterScenario),
}

impl AnyScenario {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn spawn(id: ScenarioId) -> Result<Self, ScenarioError> {
        match id {
            ScenarioId::Lot1KalmanTutorial => {
                Ok(Self::Lot1(Lot1KalmanScenario::tutorial_defaults()?))
            }
            ScenarioId::Lot2EkfTutorial => Ok(Self::Lot2Ekf(Lot2EkfScenario::tutorial_defaults()?)),
            ScenarioId::Lot2UkfTutorial => Ok(Self::Lot2Ukf(Lot2UkfScenario::tutorial_defaults()?)),
            ScenarioId::Lot2ParticleTutorial => {
                Ok(Self::Lot2Pf(Lot2ParticleScenario::tutorial_defaults()?))
            }
            ScenarioId::Lot3ClutterTutorial => {
                Ok(Self::Lot3(Lot3ClutterScenario::tutorial_defaults()?))
            }
        }
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn scenario_id(&self) -> ScenarioId {
        match self {
            Self::Lot1(_) => ScenarioId::Lot1KalmanTutorial,
            Self::Lot2Ekf(_) => ScenarioId::Lot2EkfTutorial,
            Self::Lot2Ukf(_) => ScenarioId::Lot2UkfTutorial,
            Self::Lot2Pf(_) => ScenarioId::Lot2ParticleTutorial,
            Self::Lot3(_) => ScenarioId::Lot3ClutterTutorial,
        }
    }
}

impl Scenario for AnyScenario {
    fn reset(&mut self) -> Result<(), ScenarioError> {
        match self {
            Self::Lot1(s) => s.reset(),
            Self::Lot2Ekf(s) => s.reset(),
            Self::Lot2Ukf(s) => s.reset(),
            Self::Lot2Pf(s) => s.reset(),
            Self::Lot3(s) => s.reset(),
        }
    }

    fn step(&mut self) -> Result<ScenarioStepOutcome, ScenarioError> {
        let outcome = match self {
            Self::Lot1(s) => match s.step()? {
                Lot1StepOutcome::Stepped => ScenarioStepOutcome::Stepped,
                Lot1StepOutcome::Finished => ScenarioStepOutcome::Finished,
            },
            Self::Lot2Ekf(s) => match s.step()? {
                Lot2StepOutcome::Stepped => ScenarioStepOutcome::Stepped,
                Lot2StepOutcome::Finished => ScenarioStepOutcome::Finished,
            },
            Self::Lot2Ukf(s) => match s.step()? {
                Lot2StepOutcome::Stepped => ScenarioStepOutcome::Stepped,
                Lot2StepOutcome::Finished => ScenarioStepOutcome::Finished,
            },
            Self::Lot2Pf(s) => match s.step()? {
                Lot2StepOutcome::Stepped => ScenarioStepOutcome::Stepped,
                Lot2StepOutcome::Finished => ScenarioStepOutcome::Finished,
            },
            Self::Lot3(s) => match s.step()? {
                Lot3StepOutcome::Stepped => ScenarioStepOutcome::Stepped,
                Lot3StepOutcome::Finished => ScenarioStepOutcome::Finished,
            },
        };
        Ok(outcome)
    }

    fn viewer_snapshot(&self) -> Result<ViewerSnapshot, ScenarioError> {
        match self {
            Self::Lot1(s) => {
                let snap = s.snapshot()?;
                Ok(ViewerSnapshot {
                    step: snap.step,
                    max_steps: MAX_LOT1_STEPS,
                    truth_xy: snap.truth_xy,
                    estimate_xy: snap.estimate_xy,
                    measurement_xy: snap.measurement_xy,
                    bearing_range: None,
                    ellipse_semi: Some(snap.ellipse_semi),
                    ellipse_angle_rad: snap.ellipse_angle_rad,
                    rmse: snap.rmse,
                    finished: snap.finished,
                    det_count: 0,
                    det_xy: empty_det_xy(),
                    det_is_true: empty_det_flags(),
                    associated_xy: None,
                })
            }
            Self::Lot2Ekf(s) => lot2_viewer_snapshot(s.snapshot()?),
            Self::Lot2Ukf(s) => lot2_viewer_snapshot(s.snapshot()?),
            Self::Lot2Pf(s) => {
                let mut frame = lot2_viewer_snapshot(s.snapshot()?)?;
                frame.ellipse_semi = None;
                Ok(frame)
            }
            Self::Lot3(s) => lot3_viewer_snapshot(s.snapshot()?),
        }
    }
}

fn lot2_viewer_snapshot(snap: Lot2Snapshot) -> Result<ViewerSnapshot, ScenarioError> {
    Ok(ViewerSnapshot {
        step: snap.step,
        max_steps: MAX_LOT2_STEPS,
        truth_xy: snap.truth_xy,
        estimate_xy: snap.estimate_xy,
        measurement_xy: bearing_range_to_xy(snap.measurement_br.0, snap.measurement_br.1),
        bearing_range: Some(snap.measurement_br),
        ellipse_semi: Some(snap.ellipse_semi),
        ellipse_angle_rad: snap.ellipse_angle_rad,
        rmse: snap.rmse,
        finished: snap.finished,
        det_count: 0,
        det_xy: empty_det_xy(),
        det_is_true: empty_det_flags(),
        associated_xy: None,
    })
}

fn lot3_viewer_snapshot(
    snap: crate::lot3_clutter::Lot3Snapshot,
) -> Result<ViewerSnapshot, ScenarioError> {
    let mut det_xy = empty_det_xy();
    let mut det_is_true = empty_det_flags();
    for i in 0..snap.measurement_count.min(MAX_DETS_PER_STEP) {
        det_xy[i] = [snap.measurements[i].xy.0, snap.measurements[i].xy.1];
        det_is_true[i] = snap.measurements[i].is_true;
    }
    let measurement_xy = snap
        .associated_xy
        .or_else(|| snap.measurements.first().map(|m| m.xy))
        .unwrap_or((0.0, 0.0));
    Ok(ViewerSnapshot {
        step: snap.step,
        max_steps: MAX_LOT3_STEPS,
        truth_xy: snap.truth_xy,
        estimate_xy: snap.estimate_xy,
        measurement_xy,
        bearing_range: None,
        ellipse_semi: Some(snap.ellipse_semi),
        ellipse_angle_rad: snap.ellipse_angle_rad,
        rmse: snap.rmse,
        finished: snap.finished,
        det_count: snap.measurement_count.min(MAX_DETS_PER_STEP) as u8,
        det_xy,
        det_is_true,
        associated_xy: snap.associated_xy,
    })
}
