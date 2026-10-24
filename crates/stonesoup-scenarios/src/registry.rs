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
    Lot10SingleSensorTutorial,
    Lot11KdtreeTutorial,
    Lot12AkkfTutorial,
    Lot13MultiSensorTutorial,
    Lot14OptimisedSensorTutorial,
    Lot15NetworkInfoArchTutorial,
    Lot16DataIncestTutorial,
    Lot17ArchIntroTutorial,
    Lot18SimpleOosmExample,
    Lot19KalmanOosmExample,
    Lot20PfOosmExample,
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
        static LIST: [ScenarioDescriptor; 25] = [
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
            ScenarioDescriptor {
                id: ScenarioId::Lot11KdtreeTutorial,
                title: "15 — KD-tree GNN data association",
                upstream_path: "docs/tutorials/dataassociation/KDTree_Tutorial.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot10SingleSensorTutorial,
                title: "16 — Single sensor management",
                upstream_path: "docs/tutorials/sensormanagement/01_SingleSensorManagement.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot12AkkfTutorial,
                title: "17 — Adaptive kernel Kalman filter (AKKF)",
                upstream_path: "docs/tutorials/filters/AKKF.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot13MultiSensorTutorial,
                title: "18 — Multi-sensor management",
                upstream_path: "docs/tutorials/sensormanagement/02_MultiSensorManagement.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot14OptimisedSensorTutorial,
                title: "19 — Optimised sensor management",
                upstream_path: "docs/tutorials/sensormanagement/03_OptimisedSensorManagement.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot15NetworkInfoArchTutorial,
                title: "20 — Information vs network architectures",
                upstream_path:
                    "docs/tutorials/architecture/02_Information_and_Network_Architectures.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot16DataIncestTutorial,
                title: "21 — Avoiding data incest",
                upstream_path: "docs/tutorials/architecture/03_Avoiding_Data_Incest.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot17ArchIntroTutorial,
                title: "22 — Introduction to architectures",
                upstream_path: "docs/tutorials/architecture/01_Introduction_to_Architectures.py",
                upstream_sha: SHA,
                family: "tutorial",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot18SimpleOosmExample,
                title: "23 — Simple OOSM (fixed lag)",
                upstream_path: "docs/examples/oosm/example_simple_oosm.py",
                upstream_sha: SHA,
                family: "example",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot19KalmanOosmExample,
                title: "24 — Kalman filter OOSM",
                upstream_path: "docs/examples/oosm/KalmanFilterOOSMExample.py",
                upstream_sha: SHA,
                family: "example",
            },
            ScenarioDescriptor {
                id: ScenarioId::Lot20PfOosmExample,
                title: "25 — Particle filter OOSM",
                upstream_path: "docs/examples/oosm/PF_OOSM_example.py",
                upstream_sha: SHA,
                family: "example",
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
            | ScenarioId::Lot8AsdTutorial
            | ScenarioId::Lot10SingleSensorTutorial
            | ScenarioId::Lot11KdtreeTutorial
            | ScenarioId::Lot12AkkfTutorial
            | ScenarioId::Lot13MultiSensorTutorial
            | ScenarioId::Lot14OptimisedSensorTutorial
            | ScenarioId::Lot15NetworkInfoArchTutorial
            | ScenarioId::Lot16DataIncestTutorial
            | ScenarioId::Lot17ArchIntroTutorial
            | ScenarioId::Lot18SimpleOosmExample
            | ScenarioId::Lot19KalmanOosmExample
            | ScenarioId::Lot20PfOosmExample => {
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
            | ScenarioId::Lot8AsdTutorial
            | ScenarioId::Lot10SingleSensorTutorial
            | ScenarioId::Lot11KdtreeTutorial
            | ScenarioId::Lot12AkkfTutorial
            | ScenarioId::Lot13MultiSensorTutorial
            | ScenarioId::Lot14OptimisedSensorTutorial
            | ScenarioId::Lot15NetworkInfoArchTutorial
            | ScenarioId::Lot16DataIncestTutorial
            | ScenarioId::Lot17ArchIntroTutorial
            | ScenarioId::Lot18SimpleOosmExample
            | ScenarioId::Lot19KalmanOosmExample
            | ScenarioId::Lot20PfOosmExample => {
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
