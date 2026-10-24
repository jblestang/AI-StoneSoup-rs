use crate::lot1_kalman::Lot1KalmanScenario;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioId {
    Lot1KalmanTutorial,
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
    pub fn all() -> &'static [ScenarioDescriptor] {
        const SHA: &str = "a4336b920a799cfe0a77ecb05867c5deeb371c7a";
        static LIST: [ScenarioDescriptor; 1] = [ScenarioDescriptor {
            id: ScenarioId::Lot1KalmanTutorial,
            title: "01 — Kalman filter introduction",
            upstream_path: "docs/tutorials/01_KalmanFilterTutorial.py",
            upstream_sha: SHA,
            family: "tutorial",
        }];
        &LIST
    }

    pub fn spawn(id: ScenarioId) -> Result<Lot1KalmanScenario, ScenarioError> {
        match id {
            ScenarioId::Lot1KalmanTutorial => Lot1KalmanScenario::tutorial_defaults(),
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
    fn from(e: stonesoup_core::Error) -> Self {
        Self::Core(e)
    }
}
