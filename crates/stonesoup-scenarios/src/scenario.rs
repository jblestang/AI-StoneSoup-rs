#![allow(clippy::large_enum_variant)]

use crate::lot10_single_sensor::{Lot10SensorManagerMode, Lot10SingleSensorScenario};
use crate::lot11_kdtree::Lot11KdtreeScenario;
use crate::lot13_multi_sensor::Lot13MultiSensorScenario;
use crate::lot14_optimised_sensor::Lot14OptimisedSensorScenario;
use crate::lot15_network_info_arch::Lot15NetworkInfoArchScenario;
use crate::lot16_data_incest::Lot16DataIncestScenario;
use crate::lot17_arch_intro::Lot17ArchIntroScenario;
use crate::lot18_simple_oosm::Lot18SimpleOosmScenario;
use crate::lot19_kalman_oosm::Lot19KalmanOosmScenario;
use crate::lot1_kalman::{Lot1KalmanScenario, Lot1StepOutcome, MAX_LOT1_STEPS};
use crate::lot2_ekf::{
    bearing_range_to_xy, Lot2EkfScenario, Lot2Snapshot, Lot2StepOutcome, MAX_LOT2_STEPS,
};
use crate::lot2_ukf::Lot2UkfScenario;
use crate::lot3_clutter::{
    Lot3ClutterScenario, Lot3StepOutcome, MAX_DETS_PER_STEP, MAX_LOT3_STEPS,
};
use crate::lot3_jpda::Lot3JpdaScenario;
use crate::lot3_jpda::MAX_LOT3_JPDA_STEPS;
use crate::lot3_meas::fill_step_detection_view;
use crate::lot3_multitarget::Lot3MultiTargetScenario;
use crate::lot3_multitarget::MAX_LOT3_MT_STEPS;
use crate::lot3_pda::Lot3PdaScenario;
use crate::lot3_pda::MAX_LOT3_PDA_STEPS;
use crate::lot4_tracker::MAX_LOT4_ACTIVE_TRACKS;
use crate::lot4_tutorial09::Lot4Tutorial09Scenario;
use crate::lot4_tutorial10::Lot4Tutorial10Scenario;
use crate::lot5_resampling::Lot5ResamplingScenario;
use crate::lot6_information::{Lot6InformationScenario, Lot6StepOutcome, MAX_LOT6_STEPS};
use crate::lot7_gm_phd::{Lot7GmPhdScenario, Lot7StepOutcome};
use crate::lot8_asd::{Lot8AsdScenario, Lot8StepOutcome, MAX_LOT8_STEPS};
use crate::lot9_akkf::{Lot9AkkfScenario, Lot9StepOutcome, MAX_LOT9_STEPS};
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
    /// Tutorials 06/08: second target truth and estimate (x–y).
    pub truth_b_xy: Option<(f64, f64)>,
    pub estimate_b_xy: Option<(f64, f64)>,
    /// Lot 5 resampling tutorial (`u8::MAX` when not active).
    pub resample_case: u8,
    pub resample_nparts: u8,
    /// Particle positions as `(index, state)` for up to [`MAX_VIEWER_RESAMPLE_PARTS`].
    pub resample_xy: [[f64; 2]; MAX_VIEWER_RESAMPLE_PARTS],
}

pub const MAX_VIEWER_RESAMPLE_PARTS: usize = 16;

pub(crate) const fn empty_det_xy() -> [[f64; 2]; MAX_DETS_PER_STEP] {
    [[0.0, 0.0]; MAX_DETS_PER_STEP]
}

pub(crate) const fn empty_det_flags() -> [bool; MAX_DETS_PER_STEP] {
    [false; MAX_DETS_PER_STEP]
}

pub(crate) const fn empty_resample_xy() -> [[f64; 2]; MAX_VIEWER_RESAMPLE_PARTS] {
    [[0.0, 0.0]; MAX_VIEWER_RESAMPLE_PARTS]
}

const fn no_resample_view() -> (u8, u8, [[f64; 2]; MAX_VIEWER_RESAMPLE_PARTS]) {
    (u8::MAX, 0, empty_resample_xy())
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
    Lot3Pda(Lot3PdaScenario),
    Lot3Mt(Lot3MultiTargetScenario),
    Lot3Jpda(Lot3JpdaScenario),
    Lot4T09(Lot4Tutorial09Scenario),
    Lot4T10(Lot4Tutorial10Scenario),
    Lot5Resampling(Lot5ResamplingScenario),
    Lot6Information(Lot6InformationScenario),
    Lot7Gmphd(Lot7GmPhdScenario),
    Lot8Asd(Lot8AsdScenario),
    Lot10SingleSensor(Lot10SingleSensorScenario),
    Lot11Kdtree(Lot11KdtreeScenario),
    Lot12Akkf(Lot9AkkfScenario),
    Lot13MultiSensor(Lot13MultiSensorScenario),
    Lot14OptimisedSensor(Lot14OptimisedSensorScenario),
    Lot15NetworkInfoArch(Lot15NetworkInfoArchScenario),
    Lot16DataIncest(Lot16DataIncestScenario),
    Lot17ArchIntro(Lot17ArchIntroScenario),
    Lot18SimpleOosm(Lot18SimpleOosmScenario),
    Lot19KalmanOosm(Lot19KalmanOosmScenario),
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
            ScenarioId::Lot3PdaTutorial => Ok(Self::Lot3Pda(Lot3PdaScenario::tutorial_defaults()?)),
            ScenarioId::Lot3MultiTargetTutorial => {
                Ok(Self::Lot3Mt(Lot3MultiTargetScenario::tutorial_defaults()?))
            }
            ScenarioId::Lot3JpdaTutorial => {
                Ok(Self::Lot3Jpda(Lot3JpdaScenario::tutorial_defaults()?))
            }
            ScenarioId::Lot4InitiatorsTutorial => {
                Ok(Self::Lot4T09(Lot4Tutorial09Scenario::tutorial_defaults()?))
            }
            ScenarioId::Lot4SimulationTutorial => {
                Ok(Self::Lot4T10(Lot4Tutorial10Scenario::tutorial_defaults()?))
            }
            ScenarioId::Lot5ResamplingTutorial => Ok(Self::Lot5Resampling(
                Lot5ResamplingScenario::tutorial_defaults()?,
            )),
            ScenarioId::Lot6InformationFilterTutorial => Ok(Self::Lot6Information(
                Lot6InformationScenario::tutorial_defaults()?,
            )),
            ScenarioId::Lot7GmphdTutorial => {
                Ok(Self::Lot7Gmphd(Lot7GmPhdScenario::tutorial_defaults()?))
            }
            ScenarioId::Lot8AsdTutorial => Ok(Self::Lot8Asd(Lot8AsdScenario::tutorial_defaults()?)),
            ScenarioId::Lot11KdtreeTutorial => {
                Ok(Self::Lot11Kdtree(Lot11KdtreeScenario::tutorial_defaults()?))
            }
            ScenarioId::Lot10SingleSensorTutorial => Ok(Self::Lot10SingleSensor(
                Lot10SingleSensorScenario::tutorial_defaults(Lot10SensorManagerMode::BruteForce)?,
            )),
            ScenarioId::Lot12AkkfTutorial => {
                Ok(Self::Lot12Akkf(Lot9AkkfScenario::tutorial_defaults()?))
            }
            ScenarioId::Lot13MultiSensorTutorial => Ok(Self::Lot13MultiSensor(
                Lot13MultiSensorScenario::tutorial_defaults(
                    crate::lot13_multi_sensor::Lot13SensorManagerMode::BruteForce,
                )?,
            )),
            ScenarioId::Lot14OptimisedSensorTutorial => Ok(Self::Lot14OptimisedSensor(
                Lot14OptimisedSensorScenario::tutorial_defaults(
                    crate::lot14_optimised_sensor::Lot14OptimisedSensorMode::OptimizeBrute,
                )?,
            )),
            ScenarioId::Lot15NetworkInfoArchTutorial => Ok(Self::Lot15NetworkInfoArch(
                Lot15NetworkInfoArchScenario::tutorial_defaults(
                    crate::lot15_network_info_arch::Lot15ArchitectureMode::Network,
                )?,
            )),
            ScenarioId::Lot16DataIncestTutorial => Ok(Self::Lot16DataIncest(
                Lot16DataIncestScenario::tutorial_defaults(
                    crate::lot16_data_incest::Lot16DataIncestMode::NonHierarchical,
                )?,
            )),
            ScenarioId::Lot17ArchIntroTutorial => Ok(Self::Lot17ArchIntro(
                Lot17ArchIntroScenario::tutorial_defaults()?,
            )),
            ScenarioId::Lot18SimpleOosmExample => Ok(Self::Lot18SimpleOosm(
                Lot18SimpleOosmScenario::tutorial_defaults(
                    crate::lot18_simple_oosm::Lot18OosmMode::FixedLag,
                )?,
            )),
            ScenarioId::Lot19KalmanOosmExample => Ok(Self::Lot19KalmanOosm(
                Lot19KalmanOosmScenario::tutorial_defaults(
                    crate::lot19_kalman_oosm::Lot19KalmanOosmMode::Tracker2,
                )?,
            )),
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
            Self::Lot3Pda(_) => ScenarioId::Lot3PdaTutorial,
            Self::Lot3Mt(_) => ScenarioId::Lot3MultiTargetTutorial,
            Self::Lot3Jpda(_) => ScenarioId::Lot3JpdaTutorial,
            Self::Lot4T09(_) => ScenarioId::Lot4InitiatorsTutorial,
            Self::Lot4T10(_) => ScenarioId::Lot4SimulationTutorial,
            Self::Lot5Resampling(_) => ScenarioId::Lot5ResamplingTutorial,
            Self::Lot6Information(_) => ScenarioId::Lot6InformationFilterTutorial,
            Self::Lot7Gmphd(_) => ScenarioId::Lot7GmphdTutorial,
            Self::Lot8Asd(_) => ScenarioId::Lot8AsdTutorial,
            Self::Lot10SingleSensor(_) => ScenarioId::Lot10SingleSensorTutorial,
            Self::Lot11Kdtree(_) => ScenarioId::Lot11KdtreeTutorial,
            Self::Lot12Akkf(_) => ScenarioId::Lot12AkkfTutorial,
            Self::Lot13MultiSensor(_) => ScenarioId::Lot13MultiSensorTutorial,
            Self::Lot14OptimisedSensor(_) => ScenarioId::Lot14OptimisedSensorTutorial,
            Self::Lot15NetworkInfoArch(_) => ScenarioId::Lot15NetworkInfoArchTutorial,
            Self::Lot16DataIncest(_) => ScenarioId::Lot16DataIncestTutorial,
            Self::Lot17ArchIntro(_) => ScenarioId::Lot17ArchIntroTutorial,
            Self::Lot18SimpleOosm(_) => ScenarioId::Lot18SimpleOosmExample,
            Self::Lot19KalmanOosm(_) => ScenarioId::Lot19KalmanOosmExample,
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
            Self::Lot3Pda(s) => s.reset(),
            Self::Lot3Mt(s) => s.reset(),
            Self::Lot3Jpda(s) => s.reset(),
            Self::Lot4T09(s) => s.reset(),
            Self::Lot4T10(s) => s.reset(),
            Self::Lot5Resampling(s) => s.reset(),
            Self::Lot6Information(s) => s.reset(),
            Self::Lot7Gmphd(s) => s.reset(),
            Self::Lot8Asd(s) => s.reset(),
            Self::Lot10SingleSensor(s) => s.reset(),
            Self::Lot11Kdtree(s) => s.reset(),
            Self::Lot12Akkf(s) => s.reset(),
            Self::Lot13MultiSensor(s) => s.reset(),
            Self::Lot14OptimisedSensor(s) => s.reset(),
            Self::Lot15NetworkInfoArch(s) => s.reset(),
            Self::Lot16DataIncest(s) => s.reset(),
            Self::Lot17ArchIntro(s) => s.reset(),
            Self::Lot18SimpleOosm(s) => s.reset(),
            Self::Lot19KalmanOosm(s) => s.reset(),
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
            Self::Lot3Pda(s) => {
                if s.step()? {
                    ScenarioStepOutcome::Stepped
                } else {
                    ScenarioStepOutcome::Finished
                }
            }
            Self::Lot3Mt(s) => {
                if s.step()? {
                    ScenarioStepOutcome::Stepped
                } else {
                    ScenarioStepOutcome::Finished
                }
            }
            Self::Lot3Jpda(s) => {
                if s.step()? {
                    ScenarioStepOutcome::Stepped
                } else {
                    ScenarioStepOutcome::Finished
                }
            }
            Self::Lot4T09(s) => {
                if s.step()? {
                    ScenarioStepOutcome::Stepped
                } else {
                    ScenarioStepOutcome::Finished
                }
            }
            Self::Lot4T10(s) => {
                if s.step()? {
                    ScenarioStepOutcome::Stepped
                } else {
                    ScenarioStepOutcome::Finished
                }
            }
            Self::Lot5Resampling(s) => {
                if s.step()? {
                    ScenarioStepOutcome::Stepped
                } else {
                    ScenarioStepOutcome::Finished
                }
            }
            Self::Lot6Information(s) => match s.step()? {
                Lot6StepOutcome::Stepped => ScenarioStepOutcome::Stepped,
                Lot6StepOutcome::Finished => ScenarioStepOutcome::Finished,
            },
            Self::Lot7Gmphd(s) => match s.step()? {
                Lot7StepOutcome::Stepped => ScenarioStepOutcome::Stepped,
                Lot7StepOutcome::Finished => ScenarioStepOutcome::Finished,
            },
            Self::Lot8Asd(s) => match s.step()? {
                Lot8StepOutcome::Stepped => ScenarioStepOutcome::Stepped,
                Lot8StepOutcome::Finished => ScenarioStepOutcome::Finished,
            },
            Self::Lot10SingleSensor(s) => {
                if s.step()? {
                    ScenarioStepOutcome::Stepped
                } else {
                    ScenarioStepOutcome::Finished
                }
            }
            Self::Lot11Kdtree(s) => {
                if s.step()? {
                    ScenarioStepOutcome::Stepped
                } else {
                    ScenarioStepOutcome::Finished
                }
            }
            Self::Lot12Akkf(s) => match s.step()? {
                Lot9StepOutcome::Stepped => ScenarioStepOutcome::Stepped,
                Lot9StepOutcome::Finished => ScenarioStepOutcome::Finished,
            },
            Self::Lot13MultiSensor(s) => {
                if s.step()? {
                    ScenarioStepOutcome::Stepped
                } else {
                    ScenarioStepOutcome::Finished
                }
            }
            Self::Lot14OptimisedSensor(s) => {
                if s.step()? {
                    ScenarioStepOutcome::Stepped
                } else {
                    ScenarioStepOutcome::Finished
                }
            }
            Self::Lot15NetworkInfoArch(s) => {
                if s.step()? {
                    ScenarioStepOutcome::Stepped
                } else {
                    ScenarioStepOutcome::Finished
                }
            }
            Self::Lot16DataIncest(s) => {
                if s.step()? {
                    ScenarioStepOutcome::Stepped
                } else {
                    ScenarioStepOutcome::Finished
                }
            }
            Self::Lot17ArchIntro(s) => {
                if s.step()? {
                    ScenarioStepOutcome::Stepped
                } else {
                    ScenarioStepOutcome::Finished
                }
            }
            Self::Lot18SimpleOosm(s) => {
                if s.step()? {
                    ScenarioStepOutcome::Stepped
                } else {
                    ScenarioStepOutcome::Finished
                }
            }
            Self::Lot19KalmanOosm(s) => {
                if s.step()? {
                    ScenarioStepOutcome::Stepped
                } else {
                    ScenarioStepOutcome::Finished
                }
            }
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
                    truth_b_xy: None,
                    estimate_b_xy: None,
                    resample_case: no_resample_view().0,
                    resample_nparts: no_resample_view().1,
                    resample_xy: no_resample_view().2,
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
            Self::Lot3Pda(s) => lot3_pda_viewer_snapshot(s),
            Self::Lot3Mt(s) => lot3_mt_viewer_snapshot(s),
            Self::Lot3Jpda(s) => lot3_jpda_viewer_snapshot(s),
            Self::Lot4T09(s) => s.viewer_snapshot(),
            Self::Lot4T10(s) => s.viewer_snapshot(),
            Self::Lot5Resampling(s) => s.viewer_snapshot(),
            Self::Lot6Information(s) => {
                let snap = s.snapshot()?;
                Ok(ViewerSnapshot {
                    step: snap.step,
                    max_steps: MAX_LOT6_STEPS,
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
                    truth_b_xy: None,
                    estimate_b_xy: None,
                    resample_case: no_resample_view().0,
                    resample_nparts: no_resample_view().1,
                    resample_xy: no_resample_view().2,
                })
            }
            Self::Lot7Gmphd(s) => s.viewer_snapshot(),
            Self::Lot8Asd(s) => {
                let snap = s.snapshot()?;
                Ok(ViewerSnapshot {
                    step: snap.step,
                    max_steps: MAX_LOT8_STEPS,
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
                    truth_b_xy: None,
                    estimate_b_xy: None,
                    resample_case: no_resample_view().0,
                    resample_nparts: no_resample_view().1,
                    resample_xy: no_resample_view().2,
                })
            }
            Self::Lot10SingleSensor(s) => s.viewer_snapshot(),
            Self::Lot11Kdtree(s) => s.viewer_snapshot(),
            Self::Lot12Akkf(s) => {
                let snap = s.snapshot()?;
                Ok(ViewerSnapshot {
                    step: snap.step,
                    max_steps: MAX_LOT9_STEPS,
                    truth_xy: snap.truth_xy,
                    estimate_xy: snap.estimate_xy,
                    measurement_xy: snap.truth_xy,
                    bearing_range: Some((snap.bearing_rad, 0.0)),
                    ellipse_semi: Some(snap.ellipse_semi),
                    ellipse_angle_rad: snap.ellipse_angle_rad,
                    rmse: snap.rmse,
                    finished: snap.finished,
                    det_count: 0,
                    det_xy: empty_det_xy(),
                    det_is_true: empty_det_flags(),
                    associated_xy: None,
                    truth_b_xy: None,
                    estimate_b_xy: None,
                    resample_case: no_resample_view().0,
                    resample_nparts: no_resample_view().1,
                    resample_xy: no_resample_view().2,
                })
            }
            Self::Lot13MultiSensor(s) => s.viewer_snapshot(),
            Self::Lot14OptimisedSensor(s) => s.viewer_snapshot(),
            Self::Lot15NetworkInfoArch(s) => s.viewer_snapshot(),
            Self::Lot16DataIncest(s) => s.viewer_snapshot(),
            Self::Lot17ArchIntro(s) => s.viewer_snapshot(),
            Self::Lot18SimpleOosm(s) => s.viewer_snapshot(),
            Self::Lot19KalmanOosm(s) => s.viewer_snapshot(),
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
        truth_b_xy: None,
        estimate_b_xy: None,
        resample_case: no_resample_view().0,
        resample_nparts: no_resample_view().1,
        resample_xy: no_resample_view().2,
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
        truth_b_xy: None,
        estimate_b_xy: None,
        resample_case: no_resample_view().0,
        resample_nparts: no_resample_view().1,
        resample_xy: no_resample_view().2,
    })
}

fn lot3_pda_viewer_snapshot(s: &Lot3PdaScenario) -> Result<ViewerSnapshot, ScenarioError> {
    let step = s.step_index();
    let idx = if step == 0 { 0 } else { step.saturating_sub(1) };
    let truth = crate::lot3_pda::TRUTH
        .get(idx)
        .copied()
        .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
    let est = s.estimate_state();
    let mut det_xy = empty_det_xy();
    let mut det_is_true = empty_det_flags();
    let det_count = if step == 0 {
        0
    } else {
        fill_step_detection_view(
            &crate::lot3_pda::MEAS_OFFSETS,
            &crate::lot3_pda::MEAS_XY,
            &crate::lot3_pda::_MEAS_IS_TRUE,
            idx,
            &mut det_xy,
            &mut det_is_true,
            MAX_DETS_PER_STEP,
        )?
    };
    let measurement_xy = if det_count > 0 {
        (det_xy[0][0], det_xy[0][1])
    } else {
        (0.0, 0.0)
    };
    Ok(ViewerSnapshot {
        step,
        max_steps: MAX_LOT3_PDA_STEPS,
        truth_xy: (truth[0], truth[2]),
        estimate_xy: (est.mean.0[0], est.mean.0[2]),
        measurement_xy,
        bearing_range: None,
        ellipse_semi: None,
        ellipse_angle_rad: 0.0,
        rmse: s.rmse_value(),
        finished: step >= MAX_LOT3_PDA_STEPS,
        det_count: det_count as u8,
        det_xy,
        det_is_true,
        associated_xy: None,
        truth_b_xy: None,
        estimate_b_xy: None,
        resample_case: no_resample_view().0,
        resample_nparts: no_resample_view().1,
        resample_xy: no_resample_view().2,
    })
}

fn lot3_mt_viewer_snapshot(s: &Lot3MultiTargetScenario) -> Result<ViewerSnapshot, ScenarioError> {
    let step = s.step_index();
    let idx = if step == 0 { 0 } else { step.saturating_sub(1) };
    let ta = crate::lot3_multitarget::TRUTH_A
        .get(idx)
        .copied()
        .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
    let tb = crate::lot3_multitarget::TRUTH_B
        .get(idx)
        .copied()
        .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
    let a = s
        .track_state(0)
        .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
    let b = s
        .track_state(1)
        .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
    let mut det_xy = empty_det_xy();
    let mut det_is_true = empty_det_flags();
    let det_count = if step == 0 {
        0
    } else {
        fill_step_detection_view(
            &crate::lot3_multitarget::MEAS_OFFSETS,
            &crate::lot3_multitarget::MEAS_XY,
            &crate::lot3_multitarget::_MEAS_IS_TRUE,
            idx,
            &mut det_xy,
            &mut det_is_true,
            MAX_DETS_PER_STEP,
        )?
    };
    let measurement_xy = if det_count > 0 {
        (det_xy[0][0], det_xy[0][1])
    } else {
        (0.0, 0.0)
    };
    Ok(ViewerSnapshot {
        step,
        max_steps: MAX_LOT3_MT_STEPS,
        truth_xy: (ta[0], ta[2]),
        estimate_xy: (a.mean.0[0], a.mean.0[2]),
        measurement_xy,
        bearing_range: None,
        ellipse_semi: None,
        ellipse_angle_rad: 0.0,
        rmse: s.rmse_value(),
        finished: step >= MAX_LOT3_MT_STEPS,
        det_count: det_count as u8,
        det_xy,
        det_is_true,
        associated_xy: None,
        truth_b_xy: Some((tb[0], tb[2])),
        estimate_b_xy: Some((b.mean.0[0], b.mean.0[2])),
        resample_case: no_resample_view().0,
        resample_nparts: no_resample_view().1,
        resample_xy: no_resample_view().2,
    })
}

pub(crate) fn lot4_viewer_from(
    step: usize,
    max_steps: usize,
    s: &impl Lot4ViewerTracks,
    meas_offsets: &[usize],
    meas_xy: &[[f64; 2]],
) -> Result<ViewerSnapshot, ScenarioError> {
    let idx = if step == 0 { 0 } else { step.saturating_sub(1) };
    let mut det_xy = empty_det_xy();
    let mut det_is_true = empty_det_flags();
    let det_count = if step == 0 {
        0
    } else {
        fill_step_detection_view(
            meas_offsets,
            meas_xy,
            &[],
            idx,
            &mut det_xy,
            &mut det_is_true,
            MAX_DETS_PER_STEP,
        )?
    };
    let mut means = [[0.0; 4]; MAX_LOT4_ACTIVE_TRACKS];
    let n = s.snapshot_track_means(&mut means);
    let estimate_xy = if n > 0 {
        (means[0][0], means[0][2])
    } else {
        (0.0, 0.0)
    };
    let estimate_b_xy = if n > 1 {
        Some((means[1][0], means[1][2]))
    } else {
        None
    };
    let measurement_xy = if det_count > 0 {
        (det_xy[0][0], det_xy[0][1])
    } else {
        estimate_xy
    };
    Ok(ViewerSnapshot {
        step,
        max_steps,
        truth_xy: estimate_xy,
        estimate_xy,
        measurement_xy,
        bearing_range: None,
        ellipse_semi: None,
        ellipse_angle_rad: 0.0,
        rmse: 0.0,
        finished: step >= max_steps,
        det_count: det_count as u8,
        det_xy,
        det_is_true,
        associated_xy: None,
        truth_b_xy: estimate_b_xy,
        estimate_b_xy,
        resample_case: no_resample_view().0,
        resample_nparts: no_resample_view().1,
        resample_xy: no_resample_view().2,
    })
}

pub(crate) trait Lot4ViewerTracks {
    fn snapshot_track_means(&self, out: &mut [[f64; 4]; MAX_LOT4_ACTIVE_TRACKS]) -> usize;
}

impl Lot4ViewerTracks for Lot4Tutorial09Scenario {
    fn snapshot_track_means(&self, out: &mut [[f64; 4]; MAX_LOT4_ACTIVE_TRACKS]) -> usize {
        Lot4Tutorial09Scenario::snapshot_track_means(self, out)
    }
}

impl Lot4ViewerTracks for Lot4Tutorial10Scenario {
    fn snapshot_track_means(&self, out: &mut [[f64; 4]; MAX_LOT4_ACTIVE_TRACKS]) -> usize {
        Lot4Tutorial10Scenario::snapshot_track_means(self, out)
    }
}

impl Lot4ViewerTracks for Lot11KdtreeScenario {
    fn snapshot_track_means(&self, out: &mut [[f64; 4]; MAX_LOT4_ACTIVE_TRACKS]) -> usize {
        Lot11KdtreeScenario::snapshot_track_means(self, out)
    }
}

fn lot3_jpda_viewer_snapshot(s: &Lot3JpdaScenario) -> Result<ViewerSnapshot, ScenarioError> {
    let step = s.step_index();
    let idx = if step == 0 { 0 } else { step.saturating_sub(1) };
    let ta = crate::lot3_jpda::TRUTH_A
        .get(idx)
        .copied()
        .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
    let tb = crate::lot3_jpda::TRUTH_B
        .get(idx)
        .copied()
        .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
    let a = s
        .track_state(0)
        .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
    let b = s
        .track_state(1)
        .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
    let mut det_xy = empty_det_xy();
    let mut det_is_true = empty_det_flags();
    let det_count = if step == 0 {
        0
    } else {
        fill_step_detection_view(
            &crate::lot3_jpda::MEAS_OFFSETS,
            &crate::lot3_jpda::MEAS_XY,
            &crate::lot3_jpda::_MEAS_IS_TRUE,
            idx,
            &mut det_xy,
            &mut det_is_true,
            MAX_DETS_PER_STEP,
        )?
    };
    let measurement_xy = if det_count > 0 {
        (det_xy[0][0], det_xy[0][1])
    } else {
        (0.0, 0.0)
    };
    Ok(ViewerSnapshot {
        step,
        max_steps: MAX_LOT3_JPDA_STEPS,
        truth_xy: (ta[0], ta[2]),
        estimate_xy: (a.mean.0[0], a.mean.0[2]),
        measurement_xy,
        bearing_range: None,
        ellipse_semi: None,
        ellipse_angle_rad: 0.0,
        rmse: s.rmse_value(),
        finished: step >= MAX_LOT3_JPDA_STEPS,
        det_count: det_count as u8,
        det_xy,
        det_is_true,
        associated_xy: None,
        truth_b_xy: Some((tb[0], tb[2])),
        estimate_b_xy: Some((b.mean.0[0], b.mean.0[2])),
        resample_case: no_resample_view().0,
        resample_nparts: no_resample_view().1,
        resample_xy: no_resample_view().2,
    })
}
