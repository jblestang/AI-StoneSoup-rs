use crate::lot1_kalman::{Lot1KalmanScenario, Lot1Snapshot, Lot1StepOutcome};
use crate::registry::{ScenarioError, ScenarioId};

/// Steppable scenario (strict core); rendering uses snapshots only.
pub trait Scenario {
    fn reset(&mut self) -> Result<(), ScenarioError>;
    fn step(&mut self) -> Result<Lot1StepOutcome, ScenarioError>;
    fn snapshot(&self) -> Result<Lot1Snapshot, ScenarioError>;

    fn run_to_end(&mut self) -> Result<(), ScenarioError> {
        while let Lot1StepOutcome::Stepped = self.step()? {}
        Ok(())
    }
}

pub enum AnyScenario {
    Lot1(Lot1KalmanScenario),
}

impl AnyScenario {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn spawn(id: ScenarioId) -> Result<Self, ScenarioError> {
        match id {
            ScenarioId::Lot1KalmanTutorial => {
                Ok(Self::Lot1(Lot1KalmanScenario::tutorial_defaults()?))
            }
        }
    }
}

impl Scenario for AnyScenario {
    fn reset(&mut self) -> Result<(), ScenarioError> {
        match self {
            Self::Lot1(s) => s.reset(),
        }
    }

    fn step(&mut self) -> Result<Lot1StepOutcome, ScenarioError> {
        match self {
            Self::Lot1(s) => s.step(),
        }
    }

    fn snapshot(&self) -> Result<Lot1Snapshot, ScenarioError> {
        match self {
            Self::Lot1(s) => s.snapshot(),
        }
    }
}
