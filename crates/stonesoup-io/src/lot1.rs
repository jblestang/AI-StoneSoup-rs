use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Lot1Fixture {
    pub upstream_sha: String,
    pub seed: u64,
    pub q_x: f64,
    pub q_y: f64,
    pub dt_seconds: f64,
    pub prior_mean: [f64; 4],
    pub prior_cov_diag: [f64; 4],
    pub measurement_noise_diag: [f64; 2],
    pub truth: Vec<[f64; 4]>,
    pub measurements: Vec<[f64; 2]>,
    pub track: Vec<[f64; 4]>,
}

impl Lot1Fixture {
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path.as_ref())?;
        Ok(serde_json::from_str(&text)?)
    }
}
