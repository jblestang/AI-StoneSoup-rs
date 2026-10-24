use crate::lot1_kalman::Lot1KalmanScenario;
use crate::lot2_ekf::Lot2EkfScenario;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioId {
    Lot1KalmanTutorial,
    Lot2EkfTutorial,
    Lot2UkfTutorial,
    Lot2ParticleTutorial,
    Lot3ClutterTutorial,
    Lot3MultiTargetTutorial,
    Lot3PdaTutorial,
    Lot3JpdaTutorial,
    Lot4InitiatorsTutorial,
    Lot4SimulationTutorial,
    Lot5ResamplingTutorial,
    Lot6InformationFilterTutorial,
    Lot7GmphdTutorial,
    Lot8AsdTutorial,
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
        static LIST: [ScenarioDescriptor; 14] = [
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
            ScenarioDescriptor {
                id: ScenarioId::Lot2UkfTutorial,
                title: "03 — Unscented Kalman filter",
                upstream_path: "docs/tutorials/03_UnscentedKalmanFilterTutorial.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot2ParticleTutorial,
                title: "04 — Particle filter",
                upstream_path: "docs/tutorials/04_ParticleFilter.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot3ClutterTutorial,
                title: "05 — Data association (clutter)",
                upstream_path: "docs/tutorials/05_DataAssociation-Clutter.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot3MultiTargetTutorial,
                title: "06 — Multi-target (GNN)",
                upstream_path: "docs/tutorials/06_DataAssociation-MultiTargetTutorial.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot3PdaTutorial,
                title: "07 — Probabilistic data association",
                upstream_path: "docs/tutorials/07_PDATutorial.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot3JpdaTutorial,
                title: "08 — Joint PDA",
                upstream_path: "docs/tutorials/08_JPDATutorial.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot4InitiatorsTutorial,
                title: "09 — Initiators & deleters",
                upstream_path: "docs/tutorials/09_Initiators_&_Deleters.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot4SimulationTutorial,
                title: "10 — Simulation & tracking stack",
                upstream_path: "docs/tutorials/10_Simulation_&_Tracking_Components.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot5ResamplingTutorial,
                title: "Sampling — Particle resamplers",
                upstream_path: "docs/tutorials/sampling/ResamplingTutorial.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot6InformationFilterTutorial,
                title: "12 — Information filter",
                upstream_path: "docs/tutorials/filters/InformationFilterTutorial.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot7GmphdTutorial,
                title: "13 — Gaussian mixture PHD",
                upstream_path: "docs/tutorials/filters/GMPHDTutorial.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot8AsdTutorial,
                title: "14 — Accumulated state densities (ASD)",
                upstream_path: "docs/tutorials/filters/ASDFilter.py",
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
            ScenarioId::Lot2EkfTutorial
            | ScenarioId::Lot2UkfTutorial
            | ScenarioId::Lot2ParticleTutorial
            | ScenarioId::Lot3ClutterTutorial
            | ScenarioId::Lot3MultiTargetTutorial
            | ScenarioId::Lot3PdaTutorial
            | ScenarioId::Lot3JpdaTutorial
            | ScenarioId::Lot4InitiatorsTutorial
            | ScenarioId::Lot4SimulationTutorial
            | ScenarioId::Lot5ResamplingTutorial
            | ScenarioId::Lot6InformationFilterTutorial
            | ScenarioId::Lot7GmphdTutorial
            | ScenarioId::Lot8AsdTutorial => {
                Err(ScenarioError::Core(stonesoup_core::Error::InvalidParameter))
            }
        }
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn spawn_lot2(id: ScenarioId) -> Result<Lot2EkfScenario, ScenarioError> {
        match id {
            ScenarioId::Lot2EkfTutorial => Lot2EkfScenario::tutorial_defaults(),
            ScenarioId::Lot1KalmanTutorial
            | ScenarioId::Lot2UkfTutorial
            | ScenarioId::Lot2ParticleTutorial
            | ScenarioId::Lot3ClutterTutorial
            | ScenarioId::Lot3MultiTargetTutorial
            | ScenarioId::Lot3PdaTutorial
            | ScenarioId::Lot3JpdaTutorial
            | ScenarioId::Lot4InitiatorsTutorial
            | ScenarioId::Lot4SimulationTutorial
            | ScenarioId::Lot5ResamplingTutorial
            | ScenarioId::Lot6InformationFilterTutorial
            | ScenarioId::Lot7GmphdTutorial
            | ScenarioId::Lot8AsdTutorial => {
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
