//! Run dual radar sim and write a compact JSON snapshot for x–y tracking plots.

use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::ExitCode;

use stonesoup_scenarios::{DualRadarClutterSim, ScenarioError};

fn main() -> ExitCode {
    match run() {
        Ok(path) => {
            eprintln!("wrote {path}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("dual_radar_clutter_viz: {e:?}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<String, ScenarioError> {
    let mut sim = DualRadarClutterSim::new()?;
    let stats = sim.run()?;
    let export = sim.build_viz_export(&stats);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../verification/reports");
    fs::create_dir_all(&root).map_err(scenario_io)?;
    let path = root.join("dual_radar_viz.json");
    let text = serde_json::to_string_pretty(&export).map_err(|_| scenario_io(io::Error::other("json")))?;
    fs::write(&path, text).map_err(scenario_io)?;
    Ok(path.display().to_string())
}

fn scenario_io(_e: io::Error) -> ScenarioError {
    ScenarioError::Core(stonesoup_core::Error::InvalidParameter)
}
