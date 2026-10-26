//! Two co-located rotating bearing–range radars, clutter, fused multi-target EKF tracker.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use stonesoup_core::linalg::Mat;
use stonesoup_core::{
    assess_track_quality, associate_tracks_to_truths, continuous_tracks_per_truth,
    diagnose_identity_switches_from_assoc, mod_bearing, study_radar_a_coast_relocks,
    CartesianToBearingRange,
    CoastRelockImmStudy, study_coast_relock_imm, SwitchDiagnosisSummary,
    TrackQualityReport,
    ConstantVelocity2d, ContinuousTruthTrack, Detection, Duration, Error, ExtendedKalmanUpdater,
    GaussianState, JpdaHypothesisSet, KalmanPredictor, KalmanWorkspace, ProbabilityHypothesis,
    RadarRevisitTrackToTruth, SensorId, Timestamp, TrackPath, TrackToTruthConfig,
    TrackTruthAssociation, TrackTruthPositionGate,
    TruthPath,
    MAX_GNN_ASSIGN_DETS, MAX_JPDA_EKF_TRACKS, MAX_PDA_HYPOTHESES,
    jpda_associate_ekf_det_models, jpda_detection_weights, ekf_pda_merged_update,
    GaussianMeasurementPrediction, PDA_GATE_CHI2_2D, squared_mahalanobis,
    HoldingDeleter, ImmEkfConfig, ImmEkfState, MultiMeasurementInitiator, IMM_MODEL_COUNT,
};
use stonesoup_core::linalg::VecN;
use stonesoup_core::GroundTruthState;
use stonesoup_sim::{SplitMix64, TruthStepConfig};

use crate::registry::ScenarioError;

/// Maximum **target** (true) detections one radar may output per 360° rotation; clutter is extra.
pub const TARGETS_PER_ROTATION: usize = 1000;
pub const PLOTS_PER_ROTATION: usize = TARGETS_PER_ROTATION;
pub const PLOTS_PER_RADAR: usize = TARGETS_PER_ROTATION;
pub const CLUTTER_FRACTION_TARGET: f64 = 0.10;
pub const CLUTTER_LAMBDA: f64 = 0.10;
pub const OVERLAP_AREA_FRACTION: f64 = 0.20;
pub const RANGE_A_M: f64 = 400_000.0;
/// Co-located disks: overlap area fraction = (R_B / R_A)² ⇒ R_B = R_A √0.2.
pub const RANGE_B_M: f64 = RANGE_A_M * 0.447_213_595_499_957_9;
pub const PERIOD_A_S: f64 = 6.0;
pub const PERIOD_B_S: f64 = 3.0;
pub const FOV_DEG: f64 = 90.0;
/// Measurement noise σ (bearing relative to boresight, degrees).
pub const RADAR_BEARING_SIGMA_DEG: f64 = 0.2;
/// Measurement noise σ (slant range, metres).
pub const RADAR_RANGE_SIGMA_M: f64 = 30.0;
pub const DT_S: f64 = 0.25;
/// Wall-clock simulation length (plot budget resets each radar rotation).
pub const SIM_DURATION_S: f64 = 60.0;
/// Distinct ground-truth targets in the scenario (each may produce plots every rotation).
pub const NUM_TRUTHS: usize = 1000;
const TRUTH_MIN_RANGE_M: f64 = 50_000.0;
const TRUTH_MAX_RANGE_M: f64 = RANGE_A_M * 0.98;
pub const JPDA_PROB_DETECT: f64 = 0.9;
const CLUTTER_RANGE_EXTENT_FRAC: f64 = 0.95;

/// Uniform clutter in bearing (rad) × range (m): `CLUTTER_LAMBDA / V_surveillance`.
#[inline]
pub fn clutter_spatial_density(max_range_m: f64) -> f64 {
    let fov_rad = FOV_DEG * core::f64::consts::PI / 180.0;
    CLUTTER_LAMBDA / (fov_rad * max_range_m * CLUTTER_RANGE_EXTENT_FRAC)
}

/// Default for radar A (`clutter_spatial_density(RANGE_A_M)`).
#[allow(dead_code)]
pub const JPDA_CLUTTER_DENSITY: f64 = {
    const FOV: f64 = FOV_DEG * core::f64::consts::PI / 180.0;
    CLUTTER_LAMBDA / (FOV * RANGE_A_M * CLUTTER_RANGE_EXTENT_FRAC)
};
const JPDA_DET_ASSOCIATED_THRESH: f64 = 0.35;
const JPDA_MEAS_UPDATE_THRESH: f64 = 0.35;
/// Stricter JPDA mass threshold before a tentative track earns an associated-update point.
const INITIATOR_MEAS_UPDATE_THRESH: f64 = 0.5;
/// Stone Soup ``MultiMeasurementInitiator.min_points`` (associated updates, ``updates_only``).
/// ``from_init`` sets ``update_count = 1``; one further corroborating JPDA update before confirm.
pub const INITIATOR_MIN_POINTS: u8 = 2;
/// Holding coast must exceed one radar-A rotation (6 s @ 0.25 s) so tentatives survive to the next plot.
pub const HOLDING_TIME_STEPS_DELETER: usize = 40;

fn holding_deleter() -> HoldingDeleter {
    HoldingDeleter::update_time_steps(HOLDING_TIME_STEPS_DELETER)
}
/// ~2.5× radar A rotation: coast through out-of-FOV gaps between beam passes.
pub const COAST_SECONDS: f64 = 60.0;
/// Planar cross-range at max truth range for reporting (≈ bearing σ·R at 392 km).
pub const TRUTH_ASSOC_MAX_RANGE_SIGMA_M: f64 = {
    const BEARING_SIGMA_RAD: f64 = RADAR_BEARING_SIGMA_DEG * core::f64::consts::PI / 180.0;
    BEARING_SIGMA_RAD * TRUTH_MAX_RANGE_M
};
/// Avoid nearest-neighbour truth label flicker in the 1000-target field.
pub const TRACK_TO_TRUTH_CONSEC_PAIRS: u8 = 3;
/// Legacy 1 s @ 0.25 s steps; prefer [`track_to_truth_consec_misses_for_period`].
pub const TRACK_TO_TRUTH_CONSEC_MISSES_LEGACY: u8 = 4;
/// Require one χ² unit better score before TrackToTruth steals the label from the current truth.
pub const TRACK_TO_TRUTH_SWITCH_SCORE_MARGIN: f64 = 1.0;
/// Track–truth metrics gate: ≈4σ in bearing at long range (χ²=16), tightening with range via R.
pub const TRACK_TO_TRUTH_CHI2_LIMIT: f64 = 16.0;
/// Wider BR gate to keep an established label through IMM coast between A revisits (~6σ at long range).
pub const TRACK_TO_TRUTH_CHI2_SUSTAIN: f64 = 36.0;

/// Bearing–range Mahalanobis gate for TrackToTruth / quality metrics (not JPDA detection gating).
pub fn dual_radar_track_truth_gate() -> TrackTruthPositionGate {
    TrackTruthPositionGate::BearingRangeChi2 {
        sensor_x: 0.0,
        sensor_y: 0.0,
        bearing_sigma_rad: RADAR_BEARING_SIGMA_DEG * core::f64::consts::PI / 180.0,
        range_sigma_m: RADAR_RANGE_SIGMA_M,
        chi2_limit: TRACK_TO_TRUTH_CHI2_LIMIT,
        chi2_sustain_limit: TRACK_TO_TRUTH_CHI2_SUSTAIN,
    }
}

/// TrackToTruth segment timing for radar A (bearing–range gate sensor): end label after ~one rotation.
pub fn radar_a_track_to_truth_revisit() -> RadarRevisitTrackToTruth {
    RadarRevisitTrackToTruth::new(PERIOD_A_S, DT_S, TRACK_TO_TRUTH_CONSEC_PAIRS)
}

/// Reference timing for radar B (faster rotation); metrics gate is still A-centric today.
pub fn radar_b_track_to_truth_revisit() -> RadarRevisitTrackToTruth {
    RadarRevisitTrackToTruth::new(PERIOD_B_S, DT_S, TRACK_TO_TRUTH_CONSEC_PAIRS)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn track_to_truth_consec_misses_for_period(period_s: f64, dt_s: f64) -> u8 {
    RadarRevisitTrackToTruth::new(period_s, dt_s, TRACK_TO_TRUTH_CONSEC_PAIRS).consec_misses_end()
}

pub fn dual_radar_track_to_truth_config() -> TrackToTruthConfig {
    TrackToTruthConfig::from_radar_revisit(
        dual_radar_track_truth_gate(),
        radar_a_track_to_truth_revisit(),
        TRACK_TO_TRUTH_SWITCH_SCORE_MARGIN,
    )
}
/// Coprime stride to rotate truth scan order each 360° (fair plot budget at cap).
pub const PLOT_SCAN_CURSOR_STRIDE: usize = 127;
pub const SEED: u64 = 42;

/// Sea-level speed of sound for Mach tiers (m/s).
pub const TRUTH_SPEED_OF_SOUND_MPS: f64 = 343.0;
pub const KNOTS_TO_MPS: f64 = 1852.0 / 3600.0;

pub const TRUTH_SPEED_MACH2_MPS: f64 = 2.0 * TRUTH_SPEED_OF_SOUND_MPS;
pub const TRUTH_SPEED_MACH1_MPS: f64 = TRUTH_SPEED_OF_SOUND_MPS;
pub const TRUTH_SPEED_800_KNOTS_MPS: f64 = 800.0 * KNOTS_TO_MPS;
pub const TRUTH_SPEED_400_KNOTS_MPS: f64 = 400.0 * KNOTS_TO_MPS;
pub const TRUTH_SPEED_250_KNOTS_MPS: f64 = 250.0 * KNOTS_TO_MPS;

/// Target speed mix: 10% Mach 2, 20% Mach 1, 30% 800 kn, 20% 400 kn, 20% 250 kn.
pub const TRUTH_SPEED_TIER_MPS: [f64; 5] = [
    TRUTH_SPEED_MACH2_MPS,
    TRUTH_SPEED_MACH1_MPS,
    TRUTH_SPEED_800_KNOTS_MPS,
    TRUTH_SPEED_400_KNOTS_MPS,
    TRUTH_SPEED_250_KNOTS_MPS,
];
pub const TRUTH_SPEED_TIER_FRACTION: [f64; 5] = [0.10, 0.20, 0.30, 0.20, 0.20];

pub const TRUTH_GRAVITY_MPS2: f64 = 9.806_65;
/// Peak manoeuvre magnitude for ground-truth dynamics (CA tangential / CT centripetal).
pub const TRUTH_MAX_MANEUVER_G: f64 = 9.0;
pub const TRUTH_MAX_MANEUVER_MPS2: f64 = TRUTH_MAX_MANEUVER_G * TRUTH_GRAVITY_MPS2;

const TRUTH_MODE_MIN_STEPS: u16 = 8;
const TRUTH_MODE_MAX_STEPS: u16 = 48;

/// Concurrent fused tracks; [`MAX_GNN_ASSIGN_DETS`] matches this per-step detection budget.
pub const MAX_ACTIVE: usize = 4096;
const MAX_HOLDING: usize = 65_536;
const MAX_TRUE_PLOT_TRACKS: usize = 2_000_000;

#[derive(Debug, Clone, Copy)]
struct RadarConfig {
    rpm: f64,
    max_range_m: f64,
    sensor: SensorId,
}

#[derive(Debug, Clone)]
struct RadarRuntime {
    dwell_rad: f64,
    last_ts: Option<Timestamp>,
    /// Distinct truths reported this 360° rotation (≤ [`TARGETS_PER_ROTATION`]).
    unique_targets_this_rotation: usize,
    rotation_count: usize,
    max_targets_in_rotation: usize,
    /// Round-robin offset when applying the per-rotation plot cap (realistic scan scheduling).
    plot_scan_cursor: usize,
    /// At most one true plot per truth index per rotation (beam dwell may revisit FOV many steps).
    rotation_plotted: Vec<bool>,
}

#[derive(Debug, Clone)]
struct ActiveTrack {
    id: u32,
    imm: ImmEkfState,
    last_meas_step: usize,
    birth_step: usize,
    sensors_meas_mask: u8,
}

/// How many **simulated truths** received ≥1 true plot from each radar (detection layer).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TruthSensorPlotCounts {
    pub radar_a_only: usize,
    pub radar_b_only: usize,
    pub both_radars: usize,
    pub no_plots: usize,
}

fn truth_sensor_plot_counts(seen_a: &[bool], seen_b: &[bool]) -> TruthSensorPlotCounts {
    let mut out = TruthSensorPlotCounts::default();
    for (&a, &b) in seen_a.iter().zip(seen_b.iter()) {
        match (a, b) {
            (true, true) => out.both_radars += 1,
            (true, false) => out.radar_a_only += 1,
            (false, true) => out.radar_b_only += 1,
            (false, false) => out.no_plots += 1,
        }
    }
    out
}

fn continuous_tracks_by_fused_sensor_mask(
    continuous: &[ContinuousTruthTrack],
    id_to_mask: &[(u32, u8)],
) -> TruthSensorPlotCounts {
    let mut out = TruthSensorPlotCounts::default();
    for ct in continuous {
        let mut mask = 0u8;
        for id in &ct.linked_filter_track_ids {
            for (fid, m) in id_to_mask {
                if fid == id {
                    mask |= m;
                }
            }
        }
        match mask {
            0b01 => out.radar_a_only += 1,
            0b10 => out.radar_b_only += 1,
            0b11 => out.both_radars += 1,
            _ => {}
        }
    }
    out
}

fn continuous_tracks_by_plot_visibility(
    continuous: &[ContinuousTruthTrack],
    seen_a: &[bool],
    seen_b: &[bool],
) -> TruthSensorPlotCounts {
    let mut out = TruthSensorPlotCounts::default();
    for ct in continuous {
        let i = ct.truth_id as usize;
        if i >= seen_a.len() {
            continue;
        }
        let a = seen_a[i];
        let b = seen_b[i];
        match (a, b) {
            (true, true) => out.both_radars += 1,
            (true, false) => out.radar_a_only += 1,
            (false, true) => out.radar_b_only += 1,
            (false, false) => {}
        }
    }
    out
}

#[derive(Debug, Clone, Copy)]
pub struct CompletedTrack {
    pub filter_track_id: u32,
    pub birth_step: usize,
    pub death_step: usize,
    pub end_xy: (f64, f64),
    pub truth_id: Option<u16>,
    /// Bit ``i``: fused track received JPDA meas update on radar ``i`` (0=A, 1=B).
    pub sensors_meas_mask: u8,
}

#[derive(Debug, Clone)]
pub struct DualRadarStats {
    pub radar_a_plots: usize,
    pub radar_b_plots: usize,
    pub radar_a_rotations: usize,
    pub radar_b_rotations: usize,
    pub radar_a_max_targets_in_rotation: usize,
    pub radar_b_max_targets_in_rotation: usize,
    pub total_true: usize,
    pub total_clutter: usize,
    pub simulation_steps: usize,
    pub truths_with_any_plot: usize,
    pub truths_with_labeled_fused_track: usize,
    pub continuous_truth_track_count: usize,
    /// Fused filter: completed segments after coast (few per truth).
    pub fused_completed: Vec<CompletedTrack>,
    /// One STT-style track row per **true** plot (~450 at 10% clutter on 500 plots).
    pub true_plot_tracks: Vec<CompletedTrack>,
    /// True plots whose range falls inside radar B max range (both sensors’ overlap disk).
    pub true_plots_in_overlap: usize,
    pub track_to_truth: Vec<TrackTruthAssociation>,
    pub continuous_truth_tracks: Vec<ContinuousTruthTrack>,
    pub truth_plot_sensor_counts: TruthSensorPlotCounts,
    pub continuous_track_sensor_counts: TruthSensorPlotCounts,
    /// Continuous tracks classified by **fused filter** measurement updates (not plots alone).
    pub continuous_track_fused_sensor_counts: TruthSensorPlotCounts,
    pub track_quality: TrackQualityReport,
    pub switch_diagnosis: SwitchDiagnosisSummary,
    pub coast_relock_imm: CoastRelockImmStudy,
    pub sensor_fusion: FusedSensorDiagnostics,
    pub radar_a_coast_relock: stonesoup_core::RadarACoastRelockStudy,
    pub jpda_a_histogram: crate::jpda_a_histogram::JpdaAHistogramReport,
}

/// Why fused tracks show A-only vs both-sensor JPDA updates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JpdaPassCounters {
    pub passes_with_dets_and_actives: u64,
    pub track_meas_above_thresh: u64,
    /// Association mass on a measurement, but below ``JPDA_MEAS_UPDATE_THRESH``.
    pub track_meas_below_thresh: u64,
    pub track_meas_zero: u64,
}

#[derive(Debug, Clone, Default)]
pub struct FusedSensorDiagnostics {
    pub truths_spawn_outside_b_range: u32,
    pub truths_spawn_inside_b_range: u32,
    /// Inside B disk at spawn but never received a true B plot in 60 s.
    pub inside_b_never_b_plotted: u32,
    /// B plotted the truth at least once.
    pub inside_b_b_plotted: u32,
    /// Continuous track + B plotted + fused mask includes B.
    pub inside_b_plotted_fused_b_updated: u32,
    /// B plotted but fused track never got B JPDA meas update (A-only mask).
    pub inside_b_plotted_fused_a_only: u32,
    pub inside_b_plotted_no_continuous_track: u32,
    pub jpda_radar_a: JpdaPassCounters,
    pub jpda_radar_b: JpdaPassCounters,
    pub b_plots_total: usize,
    pub a_plots_total: usize,
}

fn half_fov_rad() -> f64 {
    FOV_DEG * core::f64::consts::PI / 180.0 / 2.0
}

fn radar_model(dwell_rad: f64) -> Result<CartesianToBearingRange, ScenarioError> {
    let bearing_sigma_rad = RADAR_BEARING_SIGMA_DEG * core::f64::consts::PI / 180.0;
    let mut r = Mat::<2, 2>::zeros();
    r.set(0, 0, bearing_sigma_rad * bearing_sigma_rad)
        .map_err(ScenarioError::Core)?;
    r.set(1, 1, RADAR_RANGE_SIGMA_M * RADAR_RANGE_SIGMA_M)
        .map_err(ScenarioError::Core)?;
    Ok(CartesianToBearingRange {
        sensor_x: 0.0,
        sensor_y: 0.0,
        noise_covar: r,
        antenna_heading_rad: dwell_rad,
    })
}

fn advance_dwell(cfg: RadarConfig, rt: &mut RadarRuntime, ts: Timestamp) -> Result<f64, ScenarioError> {
    let Some(prev_ts) = rt.last_ts else {
        rt.last_ts = Some(ts);
        return Ok(rt.dwell_rad);
    };
    let dt = ts
        .checked_sub(prev_ts)
        .map_err(ScenarioError::Core)?
        .as_seconds();
    let rps = cfg.rpm / 60.0;
    let prev_dwell = rt.dwell_rad;
    let angle = prev_dwell + dt * rps * 2.0 * core::f64::consts::PI;
    rt.dwell_rad = mod_bearing(angle).map_err(ScenarioError::Core)?;
    if rt.dwell_rad + 1e-12 < prev_dwell {
        if rt.unique_targets_this_rotation > rt.max_targets_in_rotation {
            rt.max_targets_in_rotation = rt.unique_targets_this_rotation;
        }
        rt.unique_targets_this_rotation = 0;
        rt.rotation_plotted.fill(false);
        rt.rotation_count += 1;
        rt.plot_scan_cursor = (rt.plot_scan_cursor + PLOT_SCAN_CURSOR_STRIDE) % NUM_TRUTHS;
    }
    rt.last_ts = Some(ts);
    Ok(rt.dwell_rad)
}

fn poisson(lambda: f64, rng: &mut SplitMix64) -> Result<usize, ScenarioError> {
    if lambda <= 0.0 {
        return Ok(0);
    }
    let limit = libm::exp(-lambda);
    let mut k = 0usize;
    let mut p = 1.0;
    loop {
        k += 1;
        p *= rng.uniform01();
        if p <= limit {
            return Ok(k - 1);
        }
        if k > 64 {
            return Ok(k);
        }
    }
}

fn is_in_fov(model: &CartesianToBearingRange, state: [f64; 4], max_range: f64) -> Result<bool, ScenarioError> {
    let z = model.h(state).map_err(ScenarioError::Core)?;
    let half = half_fov_rad();
    Ok(z[1] <= max_range && z[0] >= -half && z[0] <= half)
}

fn measure_truth(
    model: &CartesianToBearingRange,
    gt: GroundTruthState,
    sensor: SensorId,
    rng: &mut SplitMix64,
) -> Result<Detection, ScenarioError> {
    let mut z = model.h(gt.state).map_err(ScenarioError::Core)?;
    let n0 = rng.standard_normal().map_err(ScenarioError::Core)?;
    let n1 = rng.standard_normal().map_err(ScenarioError::Core)?;
    let r = model.noise_covar;
    let r00 = r.get(0, 0).map_err(ScenarioError::Core)?;
    let r11 = r.get(1, 1).map_err(ScenarioError::Core)?;
    z[0] += libm::sqrt(r00) * n0;
    z[1] += libm::sqrt(r11) * n1;
    z[0] = mod_bearing(z[0]).map_err(ScenarioError::Core)?;
    Detection::new(z, gt.timestamp, sensor).map_err(ScenarioError::Core)
}

fn clutter_detection(
    max_range: f64,
    ts: Timestamp,
    sensor: SensorId,
    rng: &mut SplitMix64,
) -> Result<Detection, ScenarioError> {
    let half = half_fov_rad();
    let bearing = (rng.uniform01() * 2.0 - 1.0) * half;
    let range = rng.uniform01() * max_range * 0.95;
    Detection::new([bearing, range], ts, sensor).map_err(ScenarioError::Core)
}

fn large_prior(ts: Timestamp) -> Result<GaussianState, ScenarioError> {
    let mut cov = Mat::<4, 4>::zeros();
    let v_var = TRUTH_SPEED_MACH2_MPS * TRUTH_SPEED_MACH2_MPS;
    cov.set(0, 0, 1e6).map_err(ScenarioError::Core)?;
    cov.set(1, 1, v_var).map_err(ScenarioError::Core)?;
    cov.set(2, 2, 1e6).map_err(ScenarioError::Core)?;
    cov.set(3, 3, v_var).map_err(ScenarioError::Core)?;
    GaussianState::new_from_array([0.0; 4], cov, ts).map_err(ScenarioError::Core)
}

fn initiate_bearing_range(
    prior: &GaussianState,
    det: &Detection,
    model: CartesianToBearingRange,
) -> Result<GaussianState, ScenarioError> {
    let b = det.vector.0[0];
    let r = det.vector.0[1];
    let world_b = mod_bearing(b + model.antenna_heading_rad).map_err(ScenarioError::Core)?;
    let x = model.sensor_x + r * libm::cos(world_b);
    let y = model.sensor_y + r * libm::sin(world_b);
    let mut mean = prior.mean.0;
    mean[0] = x;
    mean[2] = y;
    let mut cov = prior.cov;
    for &d in &[0usize, 2] {
        for j in 0..4 {
            cov.set(d, j, 0.0).map_err(ScenarioError::Core)?;
            cov.set(j, d, 0.0).map_err(ScenarioError::Core)?;
        }
    }
    let r00 = model.noise_covar.get(0, 0).map_err(ScenarioError::Core)?;
    let r11 = model.noise_covar.get(1, 1).map_err(ScenarioError::Core)?;
    // Map bearing variance to x/y at this range (linearised).
    cov.set(0, 0, cov.get(0, 0)? + r11).map_err(ScenarioError::Core)?;
    cov.set(2, 2, cov.get(2, 2)? + r11).map_err(ScenarioError::Core)?;
    let _ = r00;
    GaussianState::new_from_array(mean, cov, det.timestamp).map_err(ScenarioError::Core)
}

fn truth_speed_tier_counts() -> [usize; 5] {
    let mut counts = [0usize; 5];
    let mut assigned = 0usize;
    for (i, &frac) in TRUTH_SPEED_TIER_FRACTION.iter().enumerate() {
        let n = if i + 1 == TRUTH_SPEED_TIER_FRACTION.len() {
            NUM_TRUTHS - assigned
        } else {
            libm::round(frac * NUM_TRUTHS as f64) as usize
        };
        counts[i] = n;
        assigned += n;
    }
    counts
}

fn shuffle_speeds(speeds: &mut [f64], rng: &mut SplitMix64) {
    for i in (1..speeds.len()).rev() {
        let j = (rng.uniform01() * (i + 1) as f64) as usize;
        speeds.swap(i, j);
    }
}

fn init_truth_speeds_mps(rng: &mut SplitMix64) -> alloc::vec::Vec<f64> {
    let tier_counts = truth_speed_tier_counts();
    let mut speeds = alloc::vec::Vec::with_capacity(NUM_TRUTHS);
    for (tier, &count) in TRUTH_SPEED_TIER_MPS.iter().zip(tier_counts.iter()) {
        for _ in 0..count {
            speeds.push(*tier);
        }
    }
    debug_assert_eq!(speeds.len(), NUM_TRUTHS);
    shuffle_speeds(&mut speeds, rng);
    speeds
}

fn init_truths(
    ts0: Timestamp,
    speed_mps: &[f64],
    rng: &mut SplitMix64,
) -> Result<Vec<GroundTruthState>, ScenarioError> {
    let mut out = Vec::with_capacity(NUM_TRUTHS);
    let r_min2 = TRUTH_MIN_RANGE_M * TRUTH_MIN_RANGE_M;
    let r_max2 = TRUTH_MAX_RANGE_M * TRUTH_MAX_RANGE_M;
    for &speed in speed_mps {
        let bearing = rng.uniform01() * 2.0 * core::f64::consts::PI;
        let u = rng.uniform01();
        let range = libm::sqrt(u * (r_max2 - r_min2) + r_min2);
        let heading = rng.uniform01() * 2.0 * core::f64::consts::PI;
        let vx = speed * libm::cos(heading);
        let vy = speed * libm::sin(heading);
        let state = [
            range * libm::cos(bearing),
            vx,
            range * libm::sin(bearing),
            vy,
        ];
        out.push(GroundTruthState::new(state, ts0).map_err(ScenarioError::Core)?);
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TruthMotionMode {
    Cv,
    Ca,
    Ct,
}

#[derive(Debug, Clone, Copy)]
struct TruthMotionProfile {
    mode: TruthMotionMode,
    steps_left: u16,
    ax: f64,
    ay: f64,
    omega_rad_s: f64,
}

fn init_truth_motion_profiles(rng: &mut SplitMix64) -> alloc::vec::Vec<TruthMotionProfile> {
    let mut out = alloc::vec::Vec::with_capacity(NUM_TRUTHS);
    for _ in 0..NUM_TRUTHS {
        out.push(sample_truth_motion(rng, TruthMotionMode::Cv, 0.0));
    }
    out
}

fn sample_mode_dwell(rng: &mut SplitMix64) -> u16 {
    let span = (TRUTH_MODE_MAX_STEPS - TRUTH_MODE_MIN_STEPS) as f64;
    TRUTH_MODE_MIN_STEPS + (rng.uniform01() * (span + 1.0)) as u16
}

fn sample_truth_motion(
    rng: &mut SplitMix64,
    mode: TruthMotionMode,
    speed_mps: f64,
) -> TruthMotionProfile {
    let steps_left = sample_mode_dwell(rng);
    match mode {
        TruthMotionMode::Cv => TruthMotionProfile {
            mode,
            steps_left,
            ax: 0.0,
            ay: 0.0,
            omega_rad_s: 0.0,
        },
        TruthMotionMode::Ca => {
            let heading = rng.uniform01() * 2.0 * core::f64::consts::PI;
            let frac = 0.15 + 0.85 * rng.uniform01();
            let mag = frac * TRUTH_MAX_MANEUVER_MPS2;
            TruthMotionProfile {
                mode,
                steps_left,
                ax: mag * libm::cos(heading),
                ay: mag * libm::sin(heading),
                omega_rad_s: 0.0,
            }
        }
        TruthMotionMode::Ct => {
            let v = speed_mps.max(30.0);
            let omega_max = TRUTH_MAX_MANEUVER_MPS2 / v;
            let sign = if rng.uniform01() < 0.5 { -1.0 } else { 1.0 };
            let frac = 0.2 + 0.8 * rng.uniform01();
            TruthMotionProfile {
                mode,
                steps_left,
                ax: 0.0,
                ay: 0.0,
                omega_rad_s: sign * frac * omega_max,
            }
        }
    }
}

fn pick_next_truth_mode(rng: &mut SplitMix64, speed_mps: f64) -> TruthMotionProfile {
    let u = rng.uniform01();
    let mode = if u < 0.35 {
        TruthMotionMode::Cv
    } else if u < 0.65 {
        TruthMotionMode::Ca
    } else {
        TruthMotionMode::Ct
    };
    sample_truth_motion(rng, mode, speed_mps)
}

fn clamp_truth_speed_cap(state: &mut [f64; 4], max_speed_mps: f64) {
    let vx = state[1];
    let vy = state[3];
    let mag = libm::sqrt(vx * vx + vy * vy);
    if mag > max_speed_mps && mag > 1e-6 {
        let s = max_speed_mps / mag;
        state[1] = vx * s;
        state[3] = vy * s;
    }
}

fn count_unique_truth_labels(completed: &[CompletedTrack]) -> usize {
    let mut seen = alloc::vec![false; NUM_TRUTHS];
    for c in completed {
        if let Some(tid) = c.truth_id {
            let i = tid as usize;
            if i < NUM_TRUTHS {
                seen[i] = true;
            }
        }
    }
    seen.iter().filter(|&&x| x).count()
}

/// Dominant truth id on a filter path from TrackToTruth segment durations.
fn primary_truth_for_filter_track(
    assoc: &[TrackTruthAssociation],
    filter_track_id: u32,
) -> Option<u16> {
    let mut covered_ns: BTreeMap<u16, i64> = BTreeMap::new();
    for a in assoc {
        if a.track_id != filter_track_id {
            continue;
        }
        let ns = a.range.end.0.saturating_sub(a.range.start.0);
        *covered_ns.entry(a.truth_id).or_default() += ns;
    }
    covered_ns
        .into_iter()
        .max_by_key(|(_, ns)| *ns)
        .map(|(tid, _)| tid)
}

fn advance_truth(
    gt: GroundTruthState,
    speed_mps: f64,
    motion: &mut TruthMotionProfile,
    dt_s: f64,
    rng: &mut SplitMix64,
) -> Result<GroundTruthState, ScenarioError> {
    let ts = gt
        .timestamp
        .checked_add(Duration::from_seconds(dt_s).map_err(ScenarioError::Core)?)
        .map_err(ScenarioError::Core)?;
    if motion.steps_left == 0 {
        *motion = pick_next_truth_mode(rng, speed_mps);
    }
    motion.steps_left = motion.steps_left.saturating_sub(1);

    let mut state = gt.state;
    let dt2 = dt_s * dt_s;

    match motion.mode {
        TruthMotionMode::Cv => {
            state[0] += state[1] * dt_s;
            state[2] += state[3] * dt_s;
        }
        TruthMotionMode::Ca => {
            state[0] += state[1] * dt_s + 0.5 * motion.ax * dt2;
            state[1] += motion.ax * dt_s;
            state[2] += state[3] * dt_s + 0.5 * motion.ay * dt2;
            state[3] += motion.ay * dt_s;
        }
        TruthMotionMode::Ct => {
            let vx = state[1];
            let vy = state[3];
            let speed = libm::sqrt(vx * vx + vy * vy).max(1.0);
            let heading = libm::atan2(vy, vx);
            let w = motion.omega_rad_s;
            let heading_new = heading + w * dt_s;
            state[1] = speed * libm::cos(heading_new);
            state[3] = speed * libm::sin(heading_new);
            state[0] += state[1] * dt_s;
            state[2] += state[3] * dt_s;
        }
    }

    clamp_truth_speed_cap(&mut state, speed_mps);
    GroundTruthState::new(state, ts).map_err(ScenarioError::Core)
}

fn truth_mode_byte(mode: TruthMotionMode) -> u8 {
    match mode {
        TruthMotionMode::Cv => 0,
        TruthMotionMode::Ca => 1,
        TruthMotionMode::Ct => 2,
    }
}

fn euclidean_xy(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    libm::sqrt(dx * dx + dy * dy)
}

pub struct DualRadarClutterSim {
    predictor: KalmanPredictor,
    /// JPDA gating on marginalized 4D combined state (CV spectral density, not IMM sub-model F).
    imm_gating_predictor: KalmanPredictor,
    imm_cfg: ImmEkfConfig,
    updater: ExtendedKalmanUpdater,
    ws: KalmanWorkspace,
    prior: GaussianState,
    truths: Vec<GroundTruthState>,
    truth_cfg: TruthStepConfig,
    radars: [RadarConfig; 2],
    radar_rt: [RadarRuntime; 2],
    plot_counts: [usize; 2],
    total_true: usize,
    total_clutter: usize,
    truths_plot_seen: Vec<bool>,
    truths_plot_seen_a: Vec<bool>,
    truths_plot_seen_b: Vec<bool>,
    step: usize,
    active: Vec<ActiveTrack>,
    tentative: MultiMeasurementInitiator,
    completed: Vec<CompletedTrack>,
    coast_steps: usize,
    rng: SplitMix64,
    true_plots_in_overlap: usize,
    true_plot_tracks: Vec<CompletedTrack>,
    next_track_id: u32,
    filter_paths: Vec<TrackPath>,
    truth_log: Vec<Vec<(Timestamp, [f64; 4])>>,
    /// Ground-truth motion mode per step: 0 = CV, 1 = CA, 2 = CT.
    truth_mode_log: Vec<Vec<(Timestamp, u8)>>,
    truth_speed_mps: Vec<f64>,
    truth_motion: Vec<TruthMotionProfile>,
    truth_birth_range_m: Vec<f64>,
    jpda_pass: [JpdaPassCounters; 2],
    truth_a_plot_steps: Vec<Vec<usize>>,
    filter_a_meas_steps: BTreeMap<u32, Vec<usize>>,
    a_meas_pass_samples: Vec<crate::jpda_a_histogram::AMeasPassSample>,
}

impl DualRadarClutterSim {
    pub fn new() -> Result<Self, ScenarioError> {
        let truth_transition = ConstantVelocity2d::new(1.0, 1.0).map_err(ScenarioError::Core)?;
        let predictor = KalmanPredictor::new(
            ConstantVelocity2d::new(80.0, 80.0).map_err(ScenarioError::Core)?,
        );
        let imm_gating_predictor = KalmanPredictor::new(
            ConstantVelocity2d::new(40.0, 40.0).map_err(ScenarioError::Core)?,
        );
        let imm_cfg = ImmEkfConfig::dual_radar_default().map_err(ScenarioError::Core)?;
        let updater = ExtendedKalmanUpdater::new(radar_model(0.0)?);
        let truth_cfg = TruthStepConfig {
            transition: truth_transition,
            dt: Duration::from_seconds(DT_S).map_err(ScenarioError::Core)?,
        };
        let ts0 = Timestamp::ZERO;
        let mut rng = SplitMix64::new(SEED);
        let truth_speed_mps = init_truth_speeds_mps(&mut rng);
        let truths = init_truths(ts0, &truth_speed_mps, &mut rng)?;
        let truth_birth_range_m = truths
            .iter()
            .map(|gt| libm::sqrt(gt.state[0] * gt.state[0] + gt.state[2] * gt.state[2]))
            .collect();
        let truth_motion = init_truth_motion_profiles(&mut rng);
        let truths_plot_seen = alloc::vec![false; NUM_TRUTHS];
        let truths_plot_seen_a = alloc::vec![false; NUM_TRUTHS];
        let truths_plot_seen_b = alloc::vec![false; NUM_TRUTHS];
        let truth_log = alloc::vec![Vec::new(); NUM_TRUTHS];
        let truth_mode_log = alloc::vec![Vec::new(); NUM_TRUTHS];
        let coast_steps = libm::ceil(COAST_SECONDS / DT_S) as usize;
        Ok(Self {
            predictor,
            imm_gating_predictor,
            imm_cfg,
            updater,
            ws: KalmanWorkspace {
                f: Mat::zeros(),
                q: Mat::zeros(),
                h: Mat::zeros(),
                r: Mat::zeros(),
            },
            prior: large_prior(ts0)?,
            truths,
            truth_cfg,
            radars: [
                RadarConfig {
                    rpm: 60.0 / PERIOD_A_S,
                    max_range_m: RANGE_A_M,
                    sensor: SensorId(0),
                },
                RadarConfig {
                    rpm: 60.0 / PERIOD_B_S,
                    max_range_m: RANGE_B_M,
                    sensor: SensorId(1),
                },
            ],
            radar_rt: [
                RadarRuntime {
                    dwell_rad: 0.0,
                    last_ts: None,
                    unique_targets_this_rotation: 0,
                    rotation_count: 0,
                    max_targets_in_rotation: 0,
                    plot_scan_cursor: 0,
                    rotation_plotted: alloc::vec![false; NUM_TRUTHS],
                },
                RadarRuntime {
                    dwell_rad: 0.0,
                    last_ts: None,
                    unique_targets_this_rotation: 0,
                    rotation_count: 0,
                    max_targets_in_rotation: 0,
                    plot_scan_cursor: 63,
                    rotation_plotted: alloc::vec![false; NUM_TRUTHS],
                },
            ],
            plot_counts: [0, 0],
            total_true: 0,
            total_clutter: 0,
            truths_plot_seen,
            truths_plot_seen_a,
            truths_plot_seen_b,
            step: 0,
            active: Vec::with_capacity(16),
            tentative: MultiMeasurementInitiator::new(
                INITIATOR_MIN_POINTS,
                true,
                MAX_HOLDING,
            ),
            completed: Vec::new(),
            coast_steps,
            rng: SplitMix64::new(SEED.wrapping_add(1)),
            true_plots_in_overlap: 0,
            true_plot_tracks: Vec::new(),
            next_track_id: 0,
            filter_paths: Vec::new(),
            truth_log,
            truth_mode_log,
            truth_speed_mps,
            truth_motion,
            truth_birth_range_m,
            jpda_pass: [JpdaPassCounters::default(), JpdaPassCounters::default()],
            truth_a_plot_steps: alloc::vec![Vec::new(); NUM_TRUTHS],
            filter_a_meas_steps: BTreeMap::new(),
            a_meas_pass_samples: Vec::new(),
        })
    }

    fn log_truths_at(&mut self, ts: Timestamp) {
        for (i, gt) in self.truths.iter().enumerate() {
            if self.truth_log[i].len() < 4096 {
                self.truth_log[i].push((ts, gt.state));
            }
            if self.truth_mode_log[i].len() < 4096 {
                self.truth_mode_log[i].push((ts, truth_mode_byte(self.truth_motion[i].mode)));
            }
        }
    }

    fn start_filter_path(&mut self, id: u32, ts: Timestamp, mean: [f64; 4]) {
        self.filter_paths.push(TrackPath {
            id,
            states: alloc::vec![(ts, mean)],
            imm_mu: alloc::vec![],
        });
    }

    fn append_filter_path(
        &mut self,
        id: u32,
        ts: Timestamp,
        mean: [f64; 4],
        mu: [f64; IMM_MODEL_COUNT],
    ) {
        if let Some(p) = self.filter_paths.iter_mut().find(|p| p.id == id) {
            if p.states.last().map(|(t, _)| *t != ts).unwrap_or(true) {
                p.states.push((ts, mean));
                p.imm_mu.push((ts, mu));
            }
        }
    }

    fn snapshot_active_paths(&mut self, ts: Timestamp) {
        let snaps: alloc::vec::Vec<(u32, [f64; 4], [f64; IMM_MODEL_COUNT])> = self
            .active
            .iter()
            .filter_map(|t| {
                t.imm
                    .combined()
                    .ok()
                    .map(|c| (t.id, c.mean.0, t.imm.mu))
            })
            .collect();
        for (id, mean, mu) in snaps {
            let _ = self.append_filter_path(id, ts, mean, mu);
        }
    }

    fn record_true_plot_track(
        &mut self,
        truth_idx: usize,
        state: [f64; 4],
        _ts: Timestamp,
    ) {
        if self.true_plot_tracks.len() >= MAX_TRUE_PLOT_TRACKS {
            return;
        }
        self.true_plot_tracks.push(CompletedTrack {
            filter_track_id: 0,
            birth_step: self.step,
            death_step: self.step,
            end_xy: (state[0], state[2]),
            truth_id: Some(truth_idx as u16),
            sensors_meas_mask: 0,
        });
    }

    fn ts(&self) -> Timestamp {
        Timestamp::from_nanos((self.step as i64) * truth_step_nanos())
    }

    fn advance_truths(&mut self) -> Result<(), ScenarioError> {
        let dt_s = self.truth_cfg.dt.as_seconds();
        for (i, gt) in self.truths.iter_mut().enumerate() {
            let speed = self
                .truth_speed_mps
                .get(i)
                .copied()
                .unwrap_or(TRUTH_SPEED_250_KNOTS_MPS);
            let motion = self
                .truth_motion
                .get_mut(i)
                .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
            *gt = advance_truth(*gt, speed, motion, dt_s, &mut self.rng)?;
        }
        Ok(())
    }

    fn generate_sensor_dets(
        &mut self,
        radar_idx: usize,
        ts: Timestamp,
        out: &mut [Detection],
    ) -> Result<usize, ScenarioError> {
        let cfg = self.radars[radar_idx];
        advance_dwell(cfg, &mut self.radar_rt[radar_idx], ts)?;
        let dwell = self.radar_rt[radar_idx].dwell_rad;
        let model = radar_model(dwell)?;
        let mut n = 0usize;
        let n_truths = self.truths.len();
        let cursor = self.radar_rt[radar_idx].plot_scan_cursor;
        for k in 0..n_truths {
            let ti = (cursor + k) % n_truths;
            let gt = self.truths[ti];
            if self.radar_rt[radar_idx]
                .rotation_plotted
                .get(ti)
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            if self.radar_rt[radar_idx].unique_targets_this_rotation >= TARGETS_PER_ROTATION {
                continue;
            }
            if is_in_fov(&model, gt.state, cfg.max_range_m)? {
                let det = measure_truth(&model, gt, cfg.sensor, &mut self.rng)?;
                if n < out.len() {
                    out[n] = det;
                    n += 1;
                    self.radar_rt[radar_idx].rotation_plotted[ti] = true;
                    self.bump_radar_target(radar_idx);
                    self.total_true += 1;
                    if ti < NUM_TRUTHS {
                        self.truths_plot_seen[ti] = true;
                        if radar_idx == 0 {
                            self.truths_plot_seen_a[ti] = true;
                            self.truth_a_plot_steps[ti].push(self.step);
                        } else {
                            self.truths_plot_seen_b[ti] = true;
                        }
                    }
                    if det.vector.0[1] <= RANGE_B_M {
                        self.true_plots_in_overlap += 1;
                    }
                    self.record_true_plot_track(ti, gt.state, gt.timestamp);
                }
            }
        }
        let n_clutter = poisson(CLUTTER_LAMBDA, &mut self.rng)?;
        for _ in 0..n_clutter {
            if n >= out.len() {
                break;
            }
            out[n] = clutter_detection(cfg.max_range_m, ts, cfg.sensor, &mut self.rng)?;
            n += 1;
            self.bump_radar_clutter(radar_idx);
            self.total_clutter += 1;
        }
        Ok(n)
    }

    fn bump_radar_target(&mut self, radar_idx: usize) {
        self.plot_counts[radar_idx] += 1;
        let rt = &mut self.radar_rt[radar_idx];
        rt.unique_targets_this_rotation += 1;
        if rt.unique_targets_this_rotation > rt.max_targets_in_rotation {
            rt.max_targets_in_rotation = rt.unique_targets_this_rotation;
        }
    }

    fn bump_radar_clutter(&mut self, radar_idx: usize) {
        self.plot_counts[radar_idx] += 1;
    }

    fn insert_active(&mut self, tail: GaussianState, step: usize) -> Result<(), ScenarioError> {
        if self.active.len() >= MAX_ACTIVE {
            return Err(ScenarioError::Core(Error::CapacityExceeded));
        }
        let id = self.next_track_id;
        self.next_track_id = self.next_track_id.saturating_add(1);
        let ts = self.ts();
        self.start_filter_path(id, ts, tail.mean.0);
        self.active.push(ActiveTrack {
            id,
            imm: ImmEkfState::from_gaussian(tail).map_err(ScenarioError::Core)?,
            last_meas_step: step,
            birth_step: step,
            sensors_meas_mask: 0,
        });
        Ok(())
    }

    fn promote_confirmed_tentatives(
        &mut self,
        confirmed: Vec<GaussianState>,
    ) -> Result<(), ScenarioError> {
        for tail in confirmed {
            self.insert_active(tail, self.step)?;
        }
        Ok(())
    }

    fn finish_track(&mut self, t: ActiveTrack, step: usize) -> Result<(), ScenarioError> {
        let m = t.imm.combined().map_err(ScenarioError::Core)?.mean.0;
        self.completed.push(CompletedTrack {
            filter_track_id: t.id,
            birth_step: t.birth_step,
            death_step: step,
            end_xy: (m[0], m[2]),
            truth_id: None,
            sensors_meas_mask: t.sensors_meas_mask,
        });
        Ok(())
    }

    fn delete_stale_active(&mut self) -> Result<(), ScenarioError> {
        let step = self.step;
        let coast = self.coast_steps;
        let mut stale = Vec::new();
        self.active.retain(|t| {
            if step.saturating_sub(t.last_meas_step) > coast {
                stale.push(t.clone());
                false
            } else {
                true
            }
        });
        for t in stale {
            self.finish_track(t, step)?;
        }
        Ok(())
    }

    fn empty_jpda_set(template: GaussianState) -> JpdaHypothesisSet {
        JpdaHypothesisSet {
            hypotheses: [ProbabilityHypothesis {
                prediction: template,
                detection_index: None,
                probability: 0.0,
            }; MAX_PDA_HYPOTHESES],
            count: 0,
        }
    }

    fn jpda_apply_sets(
        &mut self,
        tails: &mut [GaussianState],
        sets: &[JpdaHypothesisSet],
        detections: &[Detection],
        det_models: &[CartesianToBearingRange],
    ) -> Result<(), ScenarioError> {
        for (tail, set) in tails.iter_mut().zip(sets.iter()) {
            *tail = ekf_pda_merged_update(
                &set.hypotheses,
                set.count,
                detections,
                det_models,
                &mut self.updater,
                &mut self.ws,
            )
            .map_err(ScenarioError::Core)?;
        }
        Ok(())
    }

    fn meas_association_prob(set: &JpdaHypothesisSet) -> f64 {
        set.hypotheses
            .iter()
            .take(set.count)
            .filter(|h| h.detection_index.is_some())
            .map(|h| h.probability)
            .sum()
    }

    /// ``MultiMeasurementInitiator`` + JPDA on all step plots.
    fn update_tentative_initiator(
        &mut self,
        detections: &[Detection],
        det_models: &[CartesianToBearingRange],
        clutter_density: f64,
    ) -> Result<(), ScenarioError> {
        let ts = self.ts();
        let n_hold = self.tentative.holding_len();
        if n_hold == 0 || detections.is_empty() || n_hold > MAX_JPDA_EKF_TRACKS {
            return Ok(());
        }
        let mut hold_tails: Vec<GaussianState> = self
            .tentative
            .holding_tracks()
            .iter()
            .map(|h| h.tail)
            .collect();
        let template = hold_tails[0];
        let mut sets = alloc::vec![Self::empty_jpda_set(template); n_hold];
        jpda_associate_ekf_det_models(
            &hold_tails,
            ts,
            detections,
            JPDA_PROB_DETECT,
            clutter_density,
            &self.predictor,
            &self.updater,
            det_models,
            &mut self.ws,
            &mut sets,
        )
        .map_err(ScenarioError::Core)?;
        self.jpda_apply_sets(&mut hold_tails, &sets, detections, det_models)?;
        let updates_only = self.tentative.updates_only;
        for (ht, (tail, set)) in self
            .tentative
            .holding_tracks_mut()
            .iter_mut()
            .zip(hold_tails.iter().zip(sets.iter()))
        {
            let got_update = Self::meas_association_prob(set) >= INITIATOR_MEAS_UPDATE_THRESH;
            ht.finish_scan(*tail, got_update, updates_only);
        }
        let confirmed = self
            .tentative
            .confirm_and_prune(&holding_deleter())
            .map_err(ScenarioError::Core)?;
        self.promote_confirmed_tentatives(confirmed)
    }

    fn detection_gated_by_existing_track(
        &mut self,
        det: &Detection,
        model: &CartesianToBearingRange,
    ) -> Result<bool, ScenarioError> {
        for track in &self.active {
            let tail = track.imm.combined().map_err(ScenarioError::Core)?;
            let pred = self
                .predictor
                .predict(&tail, det.timestamp, &mut self.ws)
                .map_err(ScenarioError::Core)?;
            self.updater.measurement = *model;
            let meas = match self.updater.predict_measurement(&pred, &mut self.ws) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let innov = match CartesianToBearingRange::innovation(det.vector.0, meas.mean.0) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let sq = match squared_mahalanobis(
                VecN(innov),
                &GaussianMeasurementPrediction {
                    mean: VecN([0.0, 0.0]),
                    cov: meas.cov,
                },
            ) {
                Ok(s) => s,
                Err(_) => continue,
            };
            if sq <= PDA_GATE_CHI2_2D {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn initiate_tentatives_from_unassoc(
        &mut self,
        detections: &[Detection],
        det_models: &[CartesianToBearingRange],
    ) -> Result<(), ScenarioError> {
        for (i, det) in detections.iter().enumerate() {
            if self.detection_gated_by_existing_track(det, &det_models[i])? {
                continue;
            }
            let init = initiate_bearing_range(&self.prior, det, det_models[i])?;
            self.tentative
                .try_add_holding(init)
                .map_err(ScenarioError::Core)?;
        }
        Ok(())
    }

    /// IMM interaction + per-model predict for every active track at the simulation step time.
    ///
    /// Runs before any sensor pass so filter states coast through radar rotation gaps and
    /// zero-detection steps with growing process noise, not a frozen posterior.
    fn imm_coast_predict_actives(&mut self, ts: Timestamp) -> Result<(), ScenarioError> {
        for track in &mut self.active {
            track
                .imm
                .interaction_and_predict(ts, &self.imm_cfg, &mut self.ws)
                .map_err(ScenarioError::Core)?;
        }
        Ok(())
    }

    /// JPDA + EKF on the current active set for one sensor's detections (no initiation).
    fn associate_update_actives(
        &mut self,
        detections: &[Detection],
        det_models: &[CartesianToBearingRange],
        clutter_density: f64,
        radar_idx: usize,
    ) -> Result<(Vec<Detection>, Vec<CartesianToBearingRange>), ScenarioError> {
        let ts = self.ts();
        let n_dets = detections.len();
        let mut det_weights = alloc::vec![0.0; n_dets];
        let n_act = self.active.len();
        if n_act > 0 && !detections.is_empty() && n_act <= MAX_JPDA_EKF_TRACKS {
            let act_tails: alloc::vec::Vec<GaussianState> = self
                .active
                .iter()
                .map(|t| t.imm.combined().map_err(ScenarioError::Core))
                .collect::<Result<_, _>>()?;
            let template = act_tails[0];
            let mut sets = alloc::vec![Self::empty_jpda_set(template); n_act];
            // Tails are already at ``ts`` from ``imm_coast_predict_actives``; JPDA ``predict`` is a no-op.
            jpda_associate_ekf_det_models(
                &act_tails,
                ts,
                detections,
                JPDA_PROB_DETECT,
                clutter_density,
                &self.imm_gating_predictor,
                &self.updater,
                det_models,
                &mut self.ws,
                &mut sets,
            )
            .map_err(ScenarioError::Core)?;
            let pass = &mut self.jpda_pass[radar_idx];
            pass.passes_with_dets_and_actives += 1;
            for (track, set) in self.active.iter_mut().zip(sets.iter()) {
                let meas_p = Self::meas_association_prob(set);
                if radar_idx == 0 {
                    self.a_meas_pass_samples.push(
                        crate::jpda_a_histogram::AMeasPassSample {
                            track_id: track.id,
                            step: self.step.min(u16::MAX as usize) as u16,
                            meas_p: meas_p as f32,
                        },
                    );
                }
                if meas_p >= JPDA_MEAS_UPDATE_THRESH {
                    pass.track_meas_above_thresh += 1;
                    track.last_meas_step = self.step;
                    if radar_idx < 2 {
                        track.sensors_meas_mask |= 1 << radar_idx;
                    }
                    if radar_idx == 0 {
                        self.filter_a_meas_steps
                            .entry(track.id)
                            .or_default()
                            .push(self.step);
                    }
                } else if meas_p > 1e-8 {
                    pass.track_meas_below_thresh += 1;
                } else {
                    pass.track_meas_zero += 1;
                }
            }
            for (track, set) in self.active.iter_mut().zip(sets.iter()) {
                track
                    .imm
                    .jpda_update(
                        &set.hypotheses,
                        set.count,
                        detections,
                        det_models,
                        ts,
                        &mut self.updater,
                        &self.imm_cfg,
                        &mut self.ws,
                    )
                    .map_err(ScenarioError::Core)?;
            }
            det_weights = jpda_detection_weights(&sets, n_dets);
        }
        let mut unassoc = Vec::new();
        let mut un_models = Vec::new();
        for (i, det) in detections.iter().enumerate() {
            if det_weights.get(i).copied().unwrap_or(0.0) < JPDA_DET_ASSOCIATED_THRESH {
                unassoc.push(*det);
                un_models.push(det_models[i]);
            }
        }
        Ok((unassoc, un_models))
    }

    fn finish_tracker_step(
        &mut self,
        step_dets: &[Detection],
        step_models: &[CartesianToBearingRange],
        unassoc: &[Detection],
        un_models: &[CartesianToBearingRange],
        clutter_density: f64,
    ) -> Result<(), ScenarioError> {
        let ts = self.ts();
        self.delete_stale_active()?;
        if !step_dets.is_empty() {
            self.update_tentative_initiator(step_dets, step_models, clutter_density)?;
        }
        if !unassoc.is_empty() {
            self.initiate_tentatives_from_unassoc(unassoc, un_models)?;
        }
        self.snapshot_active_paths(ts);
        Ok(())
    }

    /// Advance one simulation time step. Returns `Ok(true)` while more steps remain.
    pub fn run_single_step(&mut self) -> Result<bool, ScenarioError> {
        let max_steps = max_simulation_steps();
        if self.step >= max_steps {
            return Ok(false);
        }
        if self.step > 0 {
            self.advance_truths()?;
        }
        let ts = self.ts();
        self.log_truths_at(ts);
        self.prior.timestamp = ts;
        if self.step > 0 {
            self.imm_coast_predict_actives(ts)?;
        }
        let mut step_unassoc = Vec::with_capacity(16);
        let mut step_unassoc_models = Vec::with_capacity(16);
        let mut step_all_dets = Vec::with_capacity(32);
        let mut step_all_models = Vec::with_capacity(32);
        let mut chunk = alloc::vec![
            Detection {
                vector: stonesoup_core::VecN([0.0, 0.0]),
                timestamp: ts,
                sensor: SensorId(0),
            };
            MAX_GNN_ASSIGN_DETS
        ];
        let mut any_dets = false;
        for ri in 0..2 {
            let got = self.generate_sensor_dets(ri, ts, &mut chunk)?;
            if got == 0 {
                continue;
            }
            any_dets = true;
            let model = radar_model(self.radar_rt[ri].dwell_rad)?;
            let dets = &chunk[..got];
            let models: alloc::vec::Vec<CartesianToBearingRange> = alloc::vec![model; got];
            step_all_dets.extend_from_slice(dets);
            step_all_models.extend(models.iter().copied());
            let clutter_rho = clutter_spatial_density(self.radars[ri].max_range_m);
            let (mut unassoc, mut un_models) =
                self.associate_update_actives(dets, &models, clutter_rho, ri)?;
            step_unassoc.append(&mut unassoc);
            step_unassoc_models.append(&mut un_models);
        }
        if any_dets {
            let pooled_clutter = 0.5
                * (clutter_spatial_density(self.radars[0].max_range_m)
                    + clutter_spatial_density(self.radars[1].max_range_m));
            self.finish_tracker_step(
                &step_all_dets,
                &step_all_models,
                &step_unassoc,
                &step_unassoc_models,
                pooled_clutter,
            )?;
        } else {
            self.delete_stale_active()?;
            let confirmed = self
                .tentative
                .predict_only_step(&holding_deleter())
                .map_err(ScenarioError::Core)?;
            self.promote_confirmed_tentatives(confirmed)?;
            self.snapshot_active_paths(ts);
        }
        self.step += 1;
        Ok(self.step < max_steps)
    }

    pub fn run(&mut self) -> Result<DualRadarStats, ScenarioError> {
        while self.run_single_step()? {}
        let step = self.step;
        for t in core::mem::take(&mut self.active) {
            self.finish_track(t, step)?;
        }
        let mut completed = Vec::with_capacity(self.completed.len());
        let end_label_gate = dual_radar_track_truth_gate();
        for mut c in core::mem::take(&mut self.completed) {
            let track_state = [c.end_xy.0, 0.0, c.end_xy.1, 0.0];
            let mut best_score = f64::INFINITY;
            let mut best_id = None;
            for (ti, gt) in self.truths.iter().enumerate() {
                let Some(score) = end_label_gate
                    .score(track_state, gt.state)
                    .map_err(ScenarioError::Core)?
                else {
                    continue;
                };
                if score < best_score {
                    best_score = score;
                    best_id = Some(ti as u16);
                }
            }
            c.truth_id = best_id;
            completed.push(c);
        }
        let truths_with_any_plot = self.truths_plot_seen.iter().filter(|&&x| x).count();

        let truth_paths: Vec<TruthPath> = self
            .truth_log
            .iter()
            .enumerate()
            .map(|(i, states)| TruthPath {
                id: i as u16,
                states: states.clone(),
            })
            .collect();
        let tt_config = dual_radar_track_to_truth_config();
        let track_to_truth = associate_tracks_to_truths(&self.filter_paths, &truth_paths, &tt_config)
            .map_err(ScenarioError::Core)?;
        for c in &mut completed {
            if let Some(tid) = primary_truth_for_filter_track(&track_to_truth, c.filter_track_id) {
                c.truth_id = Some(tid);
            }
        }
        let truths_with_labeled = count_unique_truth_labels(&completed);
        let continuous_truth_tracks =
            continuous_tracks_per_truth(&track_to_truth, NUM_TRUTHS as u16);
        let track_quality = assess_track_quality(
            &self.filter_paths,
            &truth_paths,
            &continuous_truth_tracks,
            &tt_config,
        )
        .map_err(ScenarioError::Core)?;
        let switch_diagnosis = diagnose_identity_switches_from_assoc(
            &self.filter_paths,
            &truth_paths,
            &track_to_truth,
            &tt_config,
            40,
        )
        .map_err(ScenarioError::Core)?;
        let coast_relock_imm = study_coast_relock_imm(
            &self.filter_paths,
            &truth_paths,
            &track_to_truth,
            &tt_config,
            &self.truth_mode_log,
            PERIOD_A_S,
        )
        .map_err(ScenarioError::Core)?;
        let truth_plot_sensor_counts = truth_sensor_plot_counts(
            &self.truths_plot_seen_a,
            &self.truths_plot_seen_b,
        );
        let continuous_track_sensor_counts = continuous_tracks_by_plot_visibility(
            &continuous_truth_tracks,
            &self.truths_plot_seen_a,
            &self.truths_plot_seen_b,
        );
        let id_to_mask: alloc::vec::Vec<(u32, u8)> = completed
            .iter()
            .map(|c| (c.filter_track_id, c.sensors_meas_mask))
            .collect();
        let continuous_track_fused_sensor_counts =
            continuous_tracks_by_fused_sensor_mask(&continuous_truth_tracks, &id_to_mask);
        let sensor_fusion = compute_fused_sensor_diagnostics(
            &self.truth_birth_range_m,
            &self.truths_plot_seen_b,
            &completed,
            &continuous_truth_tracks,
        );
        let mut sensor_fusion = sensor_fusion;
        sensor_fusion.jpda_radar_a = self.jpda_pass[0];
        sensor_fusion.jpda_radar_b = self.jpda_pass[1];
        sensor_fusion.a_plots_total = self.plot_counts[0];
        sensor_fusion.b_plots_total = self.plot_counts[1];

        let radar_a_coast_relock = study_radar_a_coast_relocks(
            &self.filter_paths,
            &truth_paths,
            &track_to_truth,
            &tt_config,
            &self.filter_a_meas_steps,
            &self.truth_a_plot_steps,
            DT_S,
            PERIOD_A_S,
        )
        .map_err(ScenarioError::Core)?;

        let jpda_a_histogram = crate::jpda_a_histogram::build_jpda_a_histogram_report(
            &self.a_meas_pass_samples,
            &self.filter_paths,
            &truth_paths,
            &track_to_truth,
            &tt_config,
            &self.truth_a_plot_steps,
            DT_S,
            PERIOD_A_S,
        )
        .map_err(ScenarioError::Core)?;

        for ri in 0..2 {
            let rt = &mut self.radar_rt[ri];
            if rt.unique_targets_this_rotation > rt.max_targets_in_rotation {
                rt.max_targets_in_rotation = rt.unique_targets_this_rotation;
            }
        }

        Ok(DualRadarStats {
            radar_a_plots: self.plot_counts[0],
            radar_b_plots: self.plot_counts[1],
            radar_a_rotations: self.radar_rt[0].rotation_count,
            radar_b_rotations: self.radar_rt[1].rotation_count,
            radar_a_max_targets_in_rotation: self.radar_rt[0].max_targets_in_rotation,
            radar_b_max_targets_in_rotation: self.radar_rt[1].max_targets_in_rotation,
            total_true: self.total_true,
            total_clutter: self.total_clutter,
            simulation_steps: self.step,
            truths_with_any_plot,
            truths_with_labeled_fused_track: truths_with_labeled,
            continuous_truth_track_count: continuous_truth_tracks.len(),
            fused_completed: completed,
            true_plot_tracks: self.true_plot_tracks.clone(),
            true_plots_in_overlap: self.true_plots_in_overlap,
            track_to_truth,
            continuous_truth_tracks,
            truth_plot_sensor_counts,
            continuous_track_sensor_counts,
            continuous_track_fused_sensor_counts,
            track_quality,
            switch_diagnosis,
            coast_relock_imm,
            sensor_fusion,
            radar_a_coast_relock,
            jpda_a_histogram,
        })
    }
}

fn compute_fused_sensor_diagnostics(
    birth_range_m: &[f64],
    seen_b: &[bool],
    completed: &[CompletedTrack],
    continuous: &[ContinuousTruthTrack],
) -> FusedSensorDiagnostics {
    let mut out = FusedSensorDiagnostics::default();
    let mut fused_mask_by_truth = [0u8; NUM_TRUTHS];
    for c in completed {
        if let Some(tid) = c.truth_id {
            let i = tid as usize;
            if i < NUM_TRUTHS {
                fused_mask_by_truth[i] |= c.sensors_meas_mask;
            }
        }
    }
    let mut has_continuous = [false; NUM_TRUTHS];
    for ct in continuous {
        let i = ct.truth_id as usize;
        if i < NUM_TRUTHS {
            has_continuous[i] = true;
        }
    }
    for tid in 0..NUM_TRUTHS {
        let in_b = birth_range_m
            .get(tid)
            .copied()
            .unwrap_or(f64::INFINITY)
            <= RANGE_B_M;
        if !in_b {
            out.truths_spawn_outside_b_range += 1;
            continue;
        }
        out.truths_spawn_inside_b_range += 1;
        if !seen_b.get(tid).copied().unwrap_or(false) {
            out.inside_b_never_b_plotted += 1;
            continue;
        }
        out.inside_b_b_plotted += 1;
        if !has_continuous[tid] {
            out.inside_b_plotted_no_continuous_track += 1;
            continue;
        }
        let mask = fused_mask_by_truth[tid];
        if mask & 0b10 != 0 {
            out.inside_b_plotted_fused_b_updated += 1;
        } else if mask & 0b01 != 0 {
            out.inside_b_plotted_fused_a_only += 1;
        }
    }
    out
}

fn truth_step_nanos() -> i64 {
    (DT_S * 1e9) as i64
}

pub fn max_simulation_steps() -> usize {
    libm::ceil(SIM_DURATION_S / DT_S) as usize
}

const VIZ_TRUTH_PATH_STRIDE: usize = 2;
const VIZ_FILTER_PATH_STRIDE: usize = 1;

fn states_to_points_km(states: &[(Timestamp, [f64; 4])], stride: usize) -> alloc::vec::Vec<[f64; 2]> {
    if states.is_empty() {
        return alloc::vec::Vec::new();
    }
    let last = states.len() - 1;
    states
        .iter()
        .enumerate()
        .filter(|(i, _)| i % stride == 0 || *i == last)
        .map(|(_, (_, mean))| [mean[0] / 1000.0, mean[2] / 1000.0])
        .collect()
}

/// Lightweight snapshot for x–y tracking plots (see `dual_radar_clutter_viz` bin).
#[derive(Debug, Clone, serde::Serialize)]
pub struct DualRadarVizExport {
    pub range_a_km: f64,
    pub range_b_km: f64,
    pub fov_deg: f64,
    /// Antenna boresight bearing from north, degrees clockwise (at end of sim).
    pub radar_beam_deg: [f64; 2],
    pub num_truths: usize,
    pub truths_with_any_plot: usize,
    pub continuous_track_count: usize,
    pub plotted_without_continuous_count: usize,
    /// Arrow length in plot = velocity × this horizon (seconds).
    pub speed_vector_horizon_s: f64,
    pub track_quality_mean_rmse_m: Option<f64>,
    pub track_quality_identity_switches: u32,
    pub track_quality_mean_score: f64,
    pub truths: alloc::vec::Vec<TruthVizPoint>,
    pub truth_trajectories: alloc::vec::Vec<TruthTrajectoryViz>,
    pub filter_tracks: alloc::vec::Vec<FilterTrackViz>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TruthVizPoint {
    pub id: u16,
    pub x_km: f64,
    pub y_km: f64,
    /// `none` | `a_only` | `b_only` | `both`
    pub plot_visibility: &'static str,
    pub has_continuous_track: bool,
    /// Received ≥1 plot but no continuous track-to-truth segment.
    pub plotted_untracked: bool,
    pub speed_mps: f64,
    pub vx_mps: f64,
    pub vy_mps: f64,
    pub quality_score: Option<f64>,
    pub rmse_m: Option<f64>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TruthTrajectoryViz {
    pub id: u16,
    pub points_km: alloc::vec::Vec<[f64; 2]>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FilterTrackViz {
    pub track_id: u32,
    pub truth_id: Option<u16>,
    pub continuous: bool,
    pub is_false_track: bool,
    pub points_km: alloc::vec::Vec<[f64; 2]>,
    pub end_vx_mps: f64,
    pub end_vy_mps: f64,
    /// 0–1 heuristic (accuracy, coverage, stability).
    pub quality_score: Option<f64>,
    pub rmse_m: Option<f64>,
    pub identity_switch_count: u32,
}

impl DualRadarClutterSim {
    pub fn build_viz_export(&self, stats: &DualRadarStats) -> DualRadarVizExport {
        let mut continuous_truth = alloc::vec![false; NUM_TRUTHS];
        let mut continuous_track_ids = alloc::vec::Vec::new();
        for ct in &stats.continuous_truth_tracks {
            let i = ct.truth_id as usize;
            if i < NUM_TRUTHS {
                continuous_truth[i] = true;
            }
            for fid in &ct.linked_filter_track_ids {
                continuous_track_ids.push(*fid);
            }
        }
        continuous_track_ids.sort_unstable();
        continuous_track_ids.dedup();

        let mut assoc_truth_by_track = alloc::vec::Vec::new();
        for assoc in &stats.track_to_truth {
            assoc_truth_by_track.push((assoc.track_id, assoc.truth_id));
        }
        assoc_truth_by_track.sort_unstable_by_key(|(id, _)| *id);

        let mut false_track_ids = alloc::vec::Vec::new();
        for c in &stats.fused_completed {
            if c.truth_id.is_none() {
                false_track_ids.push(c.filter_track_id);
            }
        }
        false_track_ids.sort_unstable();
        false_track_ids.dedup();

        let path_quality: BTreeMap<u32, _> = stats
            .track_quality
            .filter_paths
            .iter()
            .map(|q| (q.track_id, q))
            .collect();
        let truth_quality: BTreeMap<u16, _> = stats
            .track_quality
            .continuous_truth
            .iter()
            .map(|q| (q.truth_id, q))
            .collect();
        let mut score_sum = 0.0f64;
        let mut score_n = 0u32;
        for q in &stats.track_quality.continuous_truth {
            score_sum += q.quality_score;
            score_n += 1;
        }
        let track_quality_mean_score = if score_n > 0 {
            score_sum / score_n as f64
        } else {
            0.0
        };

        let mut plotted_without_continuous = 0usize;
        let truths = self
            .truths
            .iter()
            .enumerate()
            .map(|(i, gt)| {
                let seen_a = self.truths_plot_seen_a.get(i).copied().unwrap_or(false);
                let seen_b = self.truths_plot_seen_b.get(i).copied().unwrap_or(false);
                let vis = match (seen_a, seen_b) {
                    (true, true) => "both",
                    (true, false) => "a_only",
                    (false, true) => "b_only",
                    _ => "none",
                };
                let has_cont = continuous_truth.get(i).copied().unwrap_or(false);
                let plotted = seen_a || seen_b;
                let plotted_untracked = plotted && !has_cont;
                if plotted_untracked {
                    plotted_without_continuous += 1;
                }
                let tq = truth_quality.get(&(i as u16));
                let speed_mps = self
                    .truth_speed_mps
                    .get(i)
                    .copied()
                    .unwrap_or(TRUTH_SPEED_250_KNOTS_MPS);
                TruthVizPoint {
                    id: i as u16,
                    x_km: gt.state[0] / 1000.0,
                    y_km: gt.state[2] / 1000.0,
                    plot_visibility: vis,
                    has_continuous_track: has_cont,
                    plotted_untracked,
                    speed_mps,
                    vx_mps: gt.state[1],
                    vy_mps: gt.state[3],
                    quality_score: tq.map(|q| q.quality_score),
                    rmse_m: tq.and_then(|q| q.rmse_m),
                }
            })
            .collect();

        let truth_trajectories = self
            .truth_log
            .iter()
            .enumerate()
            .map(|(i, log)| TruthTrajectoryViz {
                id: i as u16,
                points_km: states_to_points_km(log, VIZ_TRUTH_PATH_STRIDE),
            })
            .collect();

        let filter_tracks = self
            .filter_paths
            .iter()
            .filter_map(|path| {
                let points_km = states_to_points_km(&path.states, VIZ_FILTER_PATH_STRIDE);
                if points_km.len() < 2 {
                    return None;
                }
                let continuous = continuous_track_ids
                    .binary_search(&path.id)
                    .is_ok();
                let truth_id = assoc_truth_by_track
                    .iter()
                    .find(|(tid, _)| *tid == path.id)
                    .map(|(_, truth)| *truth);
                let is_false_track = false_track_ids.binary_search(&path.id).is_ok();
                if is_false_track && !continuous {
                    return None;
                }
                let (_, last_mean) = path.states.last().copied().unwrap_or((Timestamp::ZERO, [0.0; 4]));
                let pq = path_quality.get(&path.id);
                Some(FilterTrackViz {
                    track_id: path.id,
                    truth_id,
                    continuous,
                    is_false_track,
                    points_km,
                    end_vx_mps: last_mean[1],
                    end_vy_mps: last_mean[3],
                    quality_score: pq.and_then(|_| {
                        truth_id.and_then(|tid| truth_quality.get(&tid).map(|q| q.quality_score))
                    }),
                    rmse_m: pq.and_then(|q| q.rmse_m),
                    identity_switch_count: pq.map(|q| q.identity_switch_count).unwrap_or(0),
                })
            })
            .collect();

        let rad_to_deg = 180.0 / core::f64::consts::PI;
        let radar_beam_deg = [
            self.radar_rt[0].dwell_rad * rad_to_deg,
            self.radar_rt[1].dwell_rad * rad_to_deg,
        ];

        DualRadarVizExport {
            range_a_km: RANGE_A_M / 1000.0,
            range_b_km: RANGE_B_M / 1000.0,
            fov_deg: FOV_DEG,
            radar_beam_deg,
            num_truths: NUM_TRUTHS,
            truths_with_any_plot: stats.truths_with_any_plot,
            continuous_track_count: stats.continuous_truth_track_count,
            plotted_without_continuous_count: plotted_without_continuous,
            speed_vector_horizon_s: 12.0,
            track_quality_mean_rmse_m: stats.track_quality.established_mean_rmse_m,
            track_quality_identity_switches: stats.track_quality.total_identity_switches,
            track_quality_mean_score: track_quality_mean_score,
            truths,
            truth_trajectories,
            filter_tracks,
        }
    }
}

impl DualRadarStats {
    pub fn clutter_fraction(&self) -> f64 {
        let total = self.total_true + self.total_clutter;
        if total == 0 {
            0.0
        } else {
            self.total_clutter as f64 / total as f64
        }
    }

    /// Primary “track count”: one row per true plot (≈450 when clutter is 10%).
    pub fn total_unique_tracks(&self) -> usize {
        self.true_plot_tracks.len()
    }

    pub fn fused_false_tracks(&self) -> usize {
        self.fused_completed
            .iter()
            .filter(|c| c.truth_id.is_none())
            .count()
    }

    pub fn fused_truth_tracks(&self) -> usize {
        self.fused_completed
            .iter()
            .filter(|c| c.truth_id.is_some())
            .count()
    }

    pub fn duration_s(&self, c: &CompletedTrack) -> f64 {
        (c.death_step.saturating_sub(c.birth_step)) as f64 * DT_S
    }

}
