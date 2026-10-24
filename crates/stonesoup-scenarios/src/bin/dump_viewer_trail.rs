//! Emit JSON viewer trails after stepping a catalogue tutorial to completion.
//! Used by `tools/python/capture_python_tutorial_screenshots.py`.

use std::io::{self, Write};
use std::process::ExitCode;

use stonesoup_scenarios::scenario::{Scenario, ScenarioStepOutcome, ViewerSnapshot};
use stonesoup_scenarios::{AnyScenario, ScenarioId, TUTORIAL02_SENSOR_XY};

struct TrailState {
    truth_trail: Vec<[f64; 2]>,
    meas_trail: Vec<[f64; 2]>,
    est_trail: Vec<[f64; 2]>,
    truth_b_trail: Vec<[f64; 2]>,
    est_b_trail: Vec<[f64; 2]>,
    step_clutter: Vec<[f64; 2]>,
    step_true_dets: Vec<[f64; 2]>,
    step_associated: Option<[f64; 2]>,
    last_ellipse: Option<EllipseOverlay>,
    resample_plot: Vec<[f64; 2]>,
}

#[derive(Clone, Copy)]
struct EllipseOverlay {
    center: [f64; 2],
    semi: (f64, f64),
    angle_rad: f64,
}

impl TrailState {
    fn new() -> Self {
        Self {
            truth_trail: Vec::new(),
            meas_trail: Vec::new(),
            est_trail: Vec::new(),
            truth_b_trail: Vec::new(),
            est_b_trail: Vec::new(),
            step_clutter: Vec::new(),
            step_true_dets: Vec::new(),
            step_associated: None,
            last_ellipse: None,
            resample_plot: Vec::new(),
        }
    }

    fn apply(&mut self, snap: ViewerSnapshot) {
        if snap.resample_case != u8::MAX {
            self.resample_plot.clear();
            for i in 0..snap.resample_nparts as usize {
                self.resample_plot.push(snap.resample_xy[i]);
            }
            return;
        }
        self.truth_trail.push([snap.truth_xy.0, snap.truth_xy.1]);
        self.step_clutter.clear();
        self.step_true_dets.clear();
        for i in 0..snap.det_count as usize {
            let p = [snap.det_xy[i][0], snap.det_xy[i][1]];
            if snap.det_is_true[i] {
                self.step_true_dets.push(p);
            } else {
                self.step_clutter.push(p);
            }
        }
        self.step_associated = snap.associated_xy.map(|(x, y)| [x, y]);
        if let Some(p) = self.step_associated {
            self.meas_trail.push(p);
        } else if snap.det_count == 0 {
            self.meas_trail
                .push([snap.measurement_xy.0, snap.measurement_xy.1]);
        }
        self.est_trail
            .push([snap.estimate_xy.0, snap.estimate_xy.1]);
        if let Some(t) = snap.truth_b_xy {
            self.truth_b_trail.push([t.0, t.1]);
        }
        if let Some(e) = snap.estimate_b_xy {
            self.est_b_trail.push([e.0, e.1]);
        }
        self.last_ellipse = snap.ellipse_semi.map(|semi| EllipseOverlay {
            center: [snap.estimate_xy.0, snap.estimate_xy.1],
            semi,
            angle_rad: snap.ellipse_angle_rad,
        });
    }
}

fn tutorial_to_id(code: &str) -> Option<ScenarioId> {
    match code {
        "01" | "1" => Some(ScenarioId::Lot1KalmanTutorial),
        "02" | "2" => Some(ScenarioId::Lot2EkfTutorial),
        "03" | "3" => Some(ScenarioId::Lot2UkfTutorial),
        "04" | "4" => Some(ScenarioId::Lot2ParticleTutorial),
        "05" | "5" => Some(ScenarioId::Lot3ClutterTutorial),
        "06" | "6" => Some(ScenarioId::Lot3MultiTargetTutorial),
        "07" | "7" => Some(ScenarioId::Lot3PdaTutorial),
        "08" | "8" => Some(ScenarioId::Lot3JpdaTutorial),
        "09" | "9" => Some(ScenarioId::Lot4InitiatorsTutorial),
        "10" => Some(ScenarioId::Lot4SimulationTutorial),
        "11" | "resampling" => Some(ScenarioId::Lot5ResamplingTutorial),
        "12" | "information" => Some(ScenarioId::Lot6InformationFilterTutorial),
        "13" | "gmphd" => Some(ScenarioId::Lot7GmphdTutorial),
        "14" | "asd" => Some(ScenarioId::Lot8AsdTutorial),
        "15" | "kdtree" => Some(ScenarioId::Lot11KdtreeTutorial),
        "16" | "sensor" => Some(ScenarioId::Lot10SingleSensorTutorial),
        "17" | "akkf" => Some(ScenarioId::Lot12AkkfTutorial),
        "18" | "multisensor" => Some(ScenarioId::Lot13MultiSensorTutorial),
        "19" | "optimised" => Some(ScenarioId::Lot14OptimisedSensorTutorial),
        "20" | "architecture" => Some(ScenarioId::Lot15NetworkInfoArchTutorial),
        "21" | "data-incest" => Some(ScenarioId::Lot16DataIncestTutorial),
        "22" | "arch-intro" => Some(ScenarioId::Lot17ArchIntroTutorial),
        "23" | "oosm" => Some(ScenarioId::Lot18SimpleOosmExample),
        "24" | "kalman-oosm" => Some(ScenarioId::Lot19KalmanOosmExample),
        "25" | "pf-oosm" => Some(ScenarioId::Lot20PfOosmExample),
        "26" | "algorithm-c" => Some(ScenarioId::Lot21OosmAlgorithmExample),
        "27" | "bearing-only" => Some(ScenarioId::Lot22BearingOnlyExample),
        "28" | "multi-tracker" => Some(ScenarioId::Lot23MultiTrackerExample),
        "29" | "ensemble" => Some(ScenarioId::Lot24EnsembleFilterExample),
        "30" | "particle-flow" => Some(ScenarioId::Lot25ParticleFlowExample),
        _ => None,
    }
}

fn is_lot2(id: ScenarioId) -> bool {
    matches!(
        id,
        ScenarioId::Lot2EkfTutorial
            | ScenarioId::Lot2UkfTutorial
            | ScenarioId::Lot2ParticleTutorial
    )
}

fn is_lot3_meas(id: ScenarioId) -> bool {
    matches!(
        id,
        ScenarioId::Lot3ClutterTutorial
            | ScenarioId::Lot3MultiTargetTutorial
            | ScenarioId::Lot3PdaTutorial
            | ScenarioId::Lot3JpdaTutorial
            | ScenarioId::Lot4InitiatorsTutorial
            | ScenarioId::Lot4SimulationTutorial
            | ScenarioId::Lot10SingleSensorTutorial
            | ScenarioId::Lot11KdtreeTutorial
            | ScenarioId::Lot13MultiSensorTutorial
            | ScenarioId::Lot14OptimisedSensorTutorial
            | ScenarioId::Lot15NetworkInfoArchTutorial
            | ScenarioId::Lot16DataIncestTutorial
            | ScenarioId::Lot17ArchIntroTutorial
            | ScenarioId::Lot23MultiTrackerExample
    )
}

fn is_lot3_nn(id: ScenarioId) -> bool {
    matches!(id, ScenarioId::Lot3ClutterTutorial)
}

fn is_lot5(id: ScenarioId) -> bool {
    matches!(id, ScenarioId::Lot5ResamplingTutorial)
}

fn run_tutorial(code: &str) -> Result<serde_json::Value, String> {
    let id = tutorial_to_id(code).ok_or_else(|| format!("unknown tutorial code: {code}"))?;
    let mut scenario = AnyScenario::spawn(id).map_err(|e| format!("spawn: {e:?}"))?;
    let mut trails = TrailState::new();
    loop {
        let outcome = scenario.step().map_err(|e| format!("step: {e:?}"))?;
        let snap = scenario
            .viewer_snapshot()
            .map_err(|e| format!("snapshot: {e:?}"))?;
        trails.apply(snap);
        if outcome == ScenarioStepOutcome::Finished {
            break;
        }
    }
    let snap = scenario
        .viewer_snapshot()
        .map_err(|e| format!("final snapshot: {e:?}"))?;

    let ellipse = trails.last_ellipse.map(|el| {
        serde_json::json!({
            "center": el.center,
            "semi_axes": [el.semi.0, el.semi.1],
            "angle_rad": el.angle_rad,
        })
    });

    Ok(serde_json::json!({
        "tutorial": code,
        "scenario_id": format!("{:?}", id),
        "is_lot2": is_lot2(id),
        "is_lot3_meas": is_lot3_meas(id),
        "is_lot3_nn": is_lot3_nn(id),
        "is_lot5": is_lot5(id),
        "sensor_xy": if is_lot2(id) { Some([TUTORIAL02_SENSOR_XY.0, TUTORIAL02_SENSOR_XY.1]) } else { None },
        "step": snap.step,
        "max_steps": snap.max_steps,
        "rmse": snap.rmse,
        "finished": snap.finished,
        "truth_trail": trails.truth_trail,
        "meas_trail": trails.meas_trail,
        "est_trail": trails.est_trail,
        "truth_b_trail": trails.truth_b_trail,
        "est_b_trail": trails.est_b_trail,
        "step_clutter": trails.step_clutter,
        "step_true_dets": trails.step_true_dets,
        "step_associated": trails.step_associated,
        "last_ellipse": ellipse,
        "resample_plot": trails.resample_plot,
    }))
}

fn main() -> ExitCode {
    let Some(code) = std::env::args().nth(1) else {
        eprintln!("usage: dump_viewer_trail <tutorial-code 01-30>");
        return ExitCode::from(2);
    };
    match run_tutorial(&code) {
        Ok(v) => {
            if let Err(e) = writeln!(io::stdout(), "{v}") {
                eprintln!("write: {e}");
                return ExitCode::from(1);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}
