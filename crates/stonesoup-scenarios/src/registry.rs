use crate::lot1_kalman::Lot1KalmanScenario;
use crate::lot2_ekf::Lot2EkfScenario;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioId {
    Lot1KalmanTutorial,
    Lot2EkfTutorial,
}

#[derive(Debug, Clone, Copy)]
pub struct ScenarioDescriptor {
    pub id: ScenarioId,
    pub title: &'static str,
    pub upstream_path: &'static str,
    pub upstream_sha: &'static str,
    pub family: &'static str,
}

pub struct ScenarioRegistry;

impl ScenarioRegistry {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn all() -> &'static [ScenarioDescriptor] {
        const SHA: &str = "a4336b920a799cfe0a77ecb05867c5deeb371c7a";
        static LIST: [ScenarioDescriptor; 2] = [
            ScenarioDescriptor {
                id: ScenarioId::Lot1KalmanTutorial,
                title: "01 — Kalman filter introduction",
                upstream_path: "docs/tutorials/01_KalmanFilterTutorial.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot2EkfTutorial,
                title: "02 — Extended Kalman filter",
                upstream_path: "docs/tutorials/02_ExtendedKalmanFilterTutorial.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
        ];
        &LIST
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn spawn(id: ScenarioId) -> Result<Lot1KalmanScenario, ScenarioError> {
        Self::spawn_lot1(id)
    }

    pub fn spawn_lot1(id: ScenarioId) -> Result<Lot1KalmanScenario, ScenarioError> {
        match id {
            ScenarioId::Lot1KalmanTutorial => Lot1KalmanScenario::tutorial_defaults(),
            ScenarioId::Lot2EkfTutorial => {
                Err(ScenarioError::Core(stonesoup_core::Error::InvalidParameter))
            }
        }
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn spawn_lot2(id: ScenarioId) -> Result<Lot2EkfScenario, ScenarioError> {
        match id {
            ScenarioId::Lot2EkfTutorial => Lot2EkfScenario::tutorial_defaults(),
            ScenarioId::Lot1KalmanTutorial => {
                Err(ScenarioError::Core(stonesoup_core::Error::InvalidParameter))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioError {
    Core(stonesoup_core::Error),
    Finished,
    NotStarted,
}

impl From<stonesoup_core::Error> for ScenarioError {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    fn from(e: stonesoup_core::Error) -> Self {
        Self::Core(e)
    }
}
