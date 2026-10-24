#![allow(clippy::large_enum_variant)]

use crate::lot1_kalman::{Lot1KalmanScenario, Lot1StepOutcome, MAX_LOT1_STEPS};
use crate::lot2_ekf::{
    bearing_range_to_xy, Lot2EkfScenario, Lot2Snapshot, Lot2StepOutcome, MAX_LOT2_STEPS,
};
use crate::lot2_ukf::Lot2UkfScenario;
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
        }
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn scenario_id(&self) -> ScenarioId {
        match self {
            Self::Lot1(_) => ScenarioId::Lot1KalmanTutorial,
            Self::Lot2Ekf(_) => ScenarioId::Lot2EkfTutorial,
            Self::Lot2Ukf(_) => ScenarioId::Lot2UkfTutorial,
            Self::Lot2Pf(_) => ScenarioId::Lot2ParticleTutorial,
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
                })
            }
            Self::Lot2Ekf(s) => lot2_viewer_snapshot(s.snapshot()?),
            Self::Lot2Ukf(s) => lot2_viewer_snapshot(s.snapshot()?),
            Self::Lot2Pf(s) => {
                let mut frame = lot2_viewer_snapshot(s.snapshot()?)?;
                frame.ellipse_semi = None;
                Ok(frame)
            }
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
    })
}
