//! Two co-located rotating bearing–range radars, clutter, fused multi-target EKF tracker.

extern crate alloc;

use alloc::vec::Vec;

use stonesoup_core::linalg::Mat;
use stonesoup_core::{
    associate_gnn_2d_ekf_det_models, associate_tracks_to_truths, continuous_tracks_per_truth,
    mod_bearing, CartesianToBearingRange, ConstantVelocity2d, ContinuousTruthTrack, Detection,
    Duration, Error, ExtendedKalmanUpdater, GaussianState, GnnAssignment, KalmanPredictor,
    KalmanWorkspace, MeasurementHypothesis, SensorId, Timestamp, TrackPath, TrackToTruthConfig,
    TrackTruthAssociation, TruthPath, MAX_GNN_ASSIGN_DETS,
};
use stonesoup_core::GroundTruthState;
use stonesoup_sim::{SplitMix64, TruthStepConfig};

use crate::registry::ScenarioError;

/// Maximum detections emitted by one radar during a single 360° rotation (budget resets each rev).
pub const PLOTS_PER_ROTATION: usize = 250;
pub const PLOTS_PER_RADAR: usize = PLOTS_PER_ROTATION;
pub const CLUTTER_FRACTION_TARGET: f64 = 0.10;
pub const CLUTTER_LAMBDA: f64 = 0.10;
pub const OVERLAP_AREA_FRACTION: f64 = 0.20;
pub const RANGE_A_M: f64 = 400_000.0;
/// Co-located disks: overlap area fraction = (R_B / R_A)² ⇒ R_B = R_A √0.2.
pub const RANGE_B_M: f64 = RANGE_A_M * 0.447_213_595_499_957_9;
pub const PERIOD_A_S: f64 = 6.0;
pub const PERIOD_B_S: f64 = 3.0;
pub const FOV_DEG: f64 = 90.0;
pub const DT_S: f64 = 0.25;
/// Wall-clock simulation length (plot budget resets each radar rotation).
pub const SIM_DURATION_S: f64 = 60.0;
pub const NUM_TRUTHS: usize = 4;
pub const GATE: f64 = 6.0;
pub const MIN_POINTS: u8 = 3;
pub const COAST_SECONDS: f64 = 30.0;
pub const TRUTH_ASSOC_METERS: f64 = 2_000.0;
pub const SEED: u64 = 42;

const MAX_ACTIVE: usize = 32;
const MAX_HOLDING: usize = 256;
const MAX_COMPLETED: usize = 128;
const MAX_TRUE_PLOT_TRACKS: usize = 32_768;

#[derive(Debug, Clone, Copy)]
struct RadarConfig {
    rpm: f64,
    max_range_m: f64,
    sensor: SensorId,
}

#[derive(Debug, Clone, Copy)]
struct RadarRuntime {
    dwell_rad: f64,
    last_ts: Option<Timestamp>,
    plots_this_rotation: usize,
    rotation_count: usize,
    max_plots_in_rotation: usize,
}

#[derive(Debug, Clone, Copy)]
struct HoldingTrack {
    tail: GaussianState,
    updates: u8,
}

#[derive(Debug, Clone, Copy)]
struct ActiveTrack {
    id: u32,
    tail: GaussianState,
    last_meas_step: usize,
    birth_step: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct CompletedTrack {
    pub birth_step: usize,
    pub death_step: usize,
    pub end_xy: (f64, f64),
    pub truth_id: Option<u8>,
}

#[derive(Debug, Clone)]
pub struct DualRadarStats {
    pub radar_a_plots: usize,
    pub radar_b_plots: usize,
    pub radar_a_rotations: usize,
    pub radar_b_rotations: usize,
    pub radar_a_max_plots_in_rotation: usize,
    pub radar_b_max_plots_in_rotation: usize,
    pub total_true: usize,
    pub total_clutter: usize,
    pub simulation_steps: usize,
    pub truths_with_plot: u8,
    pub truths_with_track: u8,
    /// Fused filter: completed segments after coast (few per truth).
    pub fused_completed: Vec<CompletedTrack>,
    /// One STT-style track row per **true** plot (~450 at 10% clutter on 500 plots).
    pub true_plot_tracks: Vec<CompletedTrack>,
    /// True plots whose range falls inside radar B max range (both sensors’ overlap disk).
    pub true_plots_in_overlap: usize,
    pub track_to_truth: Vec<TrackTruthAssociation>,
    pub continuous_truth_tracks: Vec<ContinuousTruthTrack>,
}

fn half_fov_rad() -> f64 {
    FOV_DEG * core::f64::consts::PI / 180.0 / 2.0
}

fn radar_model(dwell_rad: f64) -> Result<CartesianToBearingRange, ScenarioError> {
    let half_deg = 0.5 * core::f64::consts::PI / 180.0;
    let mut r = Mat::<2, 2>::zeros();
    r.set(0, 0, half_deg * half_deg)
        .map_err(ScenarioError::Core)?;
    r.set(1, 1, 80.0 * 80.0).map_err(ScenarioError::Core)?;
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
        if rt.plots_this_rotation > rt.max_plots_in_rotation {
            rt.max_plots_in_rotation = rt.plots_this_rotation;
        }
        rt.plots_this_rotation = 0;
        rt.rotation_count += 1;
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
    cov.set(0, 0, 1e6).map_err(ScenarioError::Core)?;
    cov.set(1, 1, 10.0).map_err(ScenarioError::Core)?;
    cov.set(2, 2, 1e6).map_err(ScenarioError::Core)?;
    cov.set(3, 3, 10.0).map_err(ScenarioError::Core)?;
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

fn init_truths(
    ts0: Timestamp,
    bearings: &[f64; NUM_TRUTHS],
    rng: &mut SplitMix64,
) -> Result<[GroundTruthState; NUM_TRUTHS], ScenarioError> {
    let p0 = truth_at(ts0, bearings[0], rng)?;
    let p1 = truth_at(ts0, bearings[1], rng)?;
    let p2 = truth_at(ts0, bearings[2], rng)?;
    let p3 = truth_at(ts0, bearings[3], rng)?;
    Ok([p0, p1, p2, p3])
}

fn truth_at(
    ts0: Timestamp,
    bearing: f64,
    rng: &mut SplitMix64,
) -> Result<GroundTruthState, ScenarioError> {
    let vx = (rng.standard_normal().map_err(ScenarioError::Core)? * 3.0).clamp(-6.0, 6.0);
    let vy = (rng.standard_normal().map_err(ScenarioError::Core)? * 3.0).clamp(-6.0, 6.0);
    let state = [
        80_000.0 * libm::cos(bearing),
        vx,
        80_000.0 * libm::sin(bearing),
        vy,
    ];
    GroundTruthState::new(state, ts0).map_err(ScenarioError::Core)
}

fn advance_truth(
    gt: GroundTruthState,
    cfg: &TruthStepConfig,
    rng: &mut SplitMix64,
) -> Result<GroundTruthState, ScenarioError> {
    use stonesoup_core::linalg::mat_vec_mul;
    let ts = gt
        .timestamp
        .checked_add(cfg.dt)
        .map_err(ScenarioError::Core)?;
    let f = cfg
        .transition
        .transition_matrix(cfg.dt)
        .map_err(ScenarioError::Core)?;
    let mean = mat_vec_mul(f, stonesoup_core::VecN(gt.state)).map_err(ScenarioError::Core)?;
    let mut state = [0.0; 4];
    for (i, slot) in state.iter_mut().enumerate() {
        *slot = mean.get(i).map_err(ScenarioError::Core)?;
    }
    let q = cfg
        .transition
        .process_noise(cfg.dt)
        .map_err(ScenarioError::Core)?;
    for block in 0..2usize {
        let i = block * 2;
        let n0 = rng.standard_normal().map_err(ScenarioError::Core)?;
        let n1 = rng.standard_normal().map_err(ScenarioError::Core)?;
        let q00 = q.get(i, i).map_err(ScenarioError::Core)?;
        let q01 = q.get(i, i + 1).map_err(ScenarioError::Core)?;
        let q11 = q.get(i + 1, i + 1).map_err(ScenarioError::Core)?;
        let l00 = libm::sqrt(if q00 > 0.0 { q00 } else { 0.0 });
        let l10 = if l00 > 1e-18 { q01 / l00 } else { 0.0 };
        let rem = q11 - l10 * l10;
        let l11 = libm::sqrt(if rem > 0.0 { rem } else { 0.0 });
        state[i] += l00 * n0;
        state[i + 1] += l10 * n0 + l11 * n1;
    }
    GroundTruthState::new(state, ts).map_err(ScenarioError::Core)
}

fn euclidean_xy(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    libm::sqrt(dx * dx + dy * dy)
}

pub struct DualRadarClutterSim {
    predictor: KalmanPredictor,
    updater: ExtendedKalmanUpdater,
    ws: KalmanWorkspace,
    prior: GaussianState,
    truths: [GroundTruthState; NUM_TRUTHS],
    truth_cfg: TruthStepConfig,
    radars: [RadarConfig; 2],
    radar_rt: [RadarRuntime; 2],
    plot_counts: [usize; 2],
    total_true: usize,
    total_clutter: usize,
    truths_detected: u8,
    step: usize,
    active: [Option<ActiveTrack>; MAX_ACTIVE],
    holding: [Option<HoldingTrack>; MAX_HOLDING],
    completed: [Option<CompletedTrack>; MAX_COMPLETED],
    n_completed: usize,
    coast_steps: usize,
    rng: SplitMix64,
    true_plots_in_overlap: usize,
    true_plot_tracks: Vec<CompletedTrack>,
    next_track_id: u32,
    filter_paths: Vec<TrackPath>,
    truth_log: [Vec<(Timestamp, [f64; 4])>; NUM_TRUTHS],
}

impl DualRadarClutterSim {
    pub fn new() -> Result<Self, ScenarioError> {
        let transition = ConstantVelocity2d::new(0.5, 0.5).map_err(ScenarioError::Core)?;
        let predictor = KalmanPredictor::new(transition);
        let updater = ExtendedKalmanUpdater::new(radar_model(0.0)?);
        let truth_cfg = TruthStepConfig {
            transition,
            dt: Duration::from_seconds(DT_S).map_err(ScenarioError::Core)?,
        };
        let ts0 = Timestamp::ZERO;
        let bearings = [
            0.0,
            core::f64::consts::FRAC_PI_2,
            core::f64::consts::PI,
            3.0 * core::f64::consts::PI / 2.0,
        ];
        let mut rng = SplitMix64::new(SEED);
        let truths = init_truths(ts0, &bearings, &mut rng)?;
        let coast_steps = libm::ceil(COAST_SECONDS / DT_S) as usize;
        Ok(Self {
            predictor,
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
                    plots_this_rotation: 0,
                    rotation_count: 0,
                    max_plots_in_rotation: 0,
                },
                RadarRuntime {
                    dwell_rad: 0.0,
                    last_ts: None,
                    plots_this_rotation: 0,
                    rotation_count: 0,
                    max_plots_in_rotation: 0,
                },
            ],
            plot_counts: [0, 0],
            total_true: 0,
            total_clutter: 0,
            truths_detected: 0,
            step: 0,
            active: [None; MAX_ACTIVE],
            holding: [None; MAX_HOLDING],
            completed: [None; MAX_COMPLETED],
            n_completed: 0,
            coast_steps,
            rng: SplitMix64::new(SEED.wrapping_add(1)),
            true_plots_in_overlap: 0,
            true_plot_tracks: Vec::new(),
            next_track_id: 0,
            filter_paths: Vec::new(),
            truth_log: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        })
    }

    fn log_truths_at(&mut self, ts: Timestamp) {
        let truths = self.truths;
        for (i, gt) in truths.iter().enumerate() {
            if self.truth_log[i].len() < 4096 {
                self.truth_log[i].push((ts, gt.state));
            }
        }
    }

    fn start_filter_path(&mut self, id: u32, ts: Timestamp, mean: [f64; 4]) {
        self.filter_paths.push(TrackPath {
            id,
            states: alloc::vec![(ts, mean)],
        });
    }

    fn append_filter_path(&mut self, id: u32, ts: Timestamp, mean: [f64; 4]) {
        if let Some(p) = self.filter_paths.iter_mut().find(|p| p.id == id) {
            if p.states.last().map(|(t, _)| *t != ts).unwrap_or(true) {
                p.states.push((ts, mean));
            }
        }
    }

    fn snapshot_active_paths(&mut self, ts: Timestamp) {
        let mut snaps = [(0u32, [0.0; 4]); MAX_ACTIVE];
        let mut n = 0usize;
        for t in self.active.iter().flatten() {
            if n < MAX_ACTIVE {
                snaps[n] = (t.id, t.tail.mean.0);
                n += 1;
            }
        }
        for i in 0..n {
            let (id, mean) = snaps[i];
            self.append_filter_path(id, ts, mean);
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
            birth_step: self.step,
            death_step: self.step,
            end_xy: (state[0], state[2]),
            truth_id: Some(truth_idx as u8),
        });
    }

    fn ts(&self) -> Timestamp {
        Timestamp::from_nanos((self.step as i64) * truth_step_nanos())
    }

    fn advance_truths(&mut self) -> Result<(), ScenarioError> {
        for gt in &mut self.truths {
            *gt = advance_truth(*gt, &self.truth_cfg, &mut self.rng)?;
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
        if self.radar_rt[radar_idx].plots_this_rotation >= PLOTS_PER_ROTATION {
            return Ok(0);
        }
        let dwell = self.radar_rt[radar_idx].dwell_rad;
        let model = radar_model(dwell)?;
        let mut n = 0usize;
        let truths = self.truths;
        for (ti, gt) in truths.iter().copied().enumerate() {
            if self.radar_rt[radar_idx].plots_this_rotation >= PLOTS_PER_ROTATION {
                break;
            }
            if is_in_fov(&model, gt.state, cfg.max_range_m)? {
                let det = measure_truth(&model, gt, cfg.sensor, &mut self.rng)?;
                if n < out.len() {
                    out[n] = det;
                    n += 1;
                    self.bump_radar_plot(radar_idx);
                    self.total_true += 1;
                    self.truths_detected |= 1 << ti;
                    if det.vector.0[1] <= RANGE_B_M {
                        self.true_plots_in_overlap += 1;
                    }
                    self.record_true_plot_track(ti, gt.state, gt.timestamp);
                }
            }
        }
        let n_clutter = poisson(CLUTTER_LAMBDA, &mut self.rng)?;
        for _ in 0..n_clutter {
            if n >= out.len() || self.radar_rt[radar_idx].plots_this_rotation >= PLOTS_PER_ROTATION {
                break;
            }
            out[n] = clutter_detection(cfg.max_range_m, ts, cfg.sensor, &mut self.rng)?;
            n += 1;
            self.bump_radar_plot(radar_idx);
            self.total_clutter += 1;
        }
        Ok(n)
    }

    fn bump_radar_plot(&mut self, radar_idx: usize) {
        self.plot_counts[radar_idx] += 1;
        let rt = &mut self.radar_rt[radar_idx];
        rt.plots_this_rotation += 1;
        if rt.plots_this_rotation > rt.max_plots_in_rotation {
            rt.max_plots_in_rotation = rt.plots_this_rotation;
        }
    }

    fn insert_active(&mut self, tail: GaussianState, step: usize) -> Result<(), ScenarioError> {
        let id = self.next_track_id;
        self.next_track_id = self.next_track_id.saturating_add(1);
        let ts = self.ts();
        self.start_filter_path(id, ts, tail.mean.0);
        for slot in &mut self.active {
            if slot.is_none() {
                *slot = Some(ActiveTrack {
                    id,
                    tail,
                    last_meas_step: step,
                    birth_step: step,
                });
                return Ok(());
            }
        }
        Err(ScenarioError::Core(Error::CapacityExceeded))
    }

    fn insert_holding(&mut self, tail: GaussianState) -> Result<(), ScenarioError> {
        for slot in &mut self.holding {
            if slot.is_none() {
                *slot = Some(HoldingTrack { tail, updates: 1 });
                return Ok(());
            }
        }
        Err(ScenarioError::Core(Error::CapacityExceeded))
    }

    fn finish_track(&mut self, t: ActiveTrack, step: usize) -> Result<(), ScenarioError> {
        if self.n_completed >= MAX_COMPLETED {
            return Err(ScenarioError::Core(Error::CapacityExceeded));
        }
        let m = t.tail.mean.0;
        self.completed[self.n_completed] = Some(CompletedTrack {
            birth_step: t.birth_step,
            death_step: step,
            end_xy: (m[0], m[2]),
            truth_id: None,
        });
        self.n_completed += 1;
        Ok(())
    }

    fn delete_stale_active(&mut self) -> Result<(), ScenarioError> {
        let step = self.step;
        let coast = self.coast_steps;
        let mut to_finish = [None; MAX_ACTIVE];
        for (i, slot) in self.active.iter().enumerate() {
            if let Some(t) = slot {
                if step.saturating_sub(t.last_meas_step) > coast {
                    to_finish[i] = Some(*t);
                }
            }
        }
        for (i, done) in to_finish.iter().enumerate() {
            if let Some(t) = done {
                self.finish_track(*t, step)?;
                self.active[i] = None;
            }
        }
        Ok(())
    }

    fn multi_initiate(
        &mut self,
        detections: &[Detection],
        det_models: &[CartesianToBearingRange],
    ) -> Result<(), ScenarioError> {
        let ts = self.ts();
        let mut det_used = [false; MAX_GNN_ASSIGN_DETS];
        let mut hold_tails = [self.prior; MAX_HOLDING];
        let mut hold_map = [0usize; MAX_HOLDING];
        let mut n_hold = 0usize;
        for (i, slot) in self.holding.iter().enumerate() {
            if let Some(h) = slot {
                hold_tails[n_hold] = h.tail;
                hold_map[n_hold] = i;
                n_hold += 1;
            }
        }
        if n_hold > 0 && !detections.is_empty() {
            let mut assigns = [GnnAssignment {
                hypothesis: MeasurementHypothesis {
                    prediction: hold_tails[0],
                    detection_index: None,
                    distance: GATE,
                },
            }; MAX_HOLDING];
            associate_gnn_2d_ekf_det_models(
                &hold_tails[..n_hold],
                ts,
                detections,
                GATE,
                &self.predictor,
                &self.updater,
                Some(det_models),
                &mut self.ws,
                &mut assigns[..n_hold],
            )
            .map_err(ScenarioError::Core)?;
            for t in 0..n_hold {
                let slot = hold_map[t];
                let h = &assigns[t].hypothesis;
                let ht = self.holding[slot]
                    .as_mut()
                    .ok_or(ScenarioError::Core(Error::InvalidDimension))?;
                if let Some(di) = h.detection_index {
                    self.updater.measurement = det_models[di];
                    let post = self
                        .updater
                        .update(&h.prediction, &detections[di], &mut self.ws)
                        .map_err(ScenarioError::Core)?;
                    ht.tail = post;
                    ht.updates = ht.updates.saturating_add(1);
                    det_used[di] = true;
                } else {
                    ht.tail = h.prediction;
                }
            }
            for slot in 0..MAX_HOLDING {
                if let Some(ht) = self.holding[slot].as_ref() {
                    if ht.updates >= MIN_POINTS {
                        let tail = self.holding[slot]
                            .take()
                            .map(|h| h.tail)
                            .ok_or(ScenarioError::Core(Error::InvalidDimension))?;
                        self.insert_active(tail, self.step)?;
                    }
                }
            }
        }
        for (i, det) in detections.iter().enumerate() {
            if det_used[i] {
                continue;
            }
            let init = initiate_bearing_range(&self.prior, det, det_models[i])?;
            let _ = self.insert_holding(init);
        }
        Ok(())
    }

    fn tracker_update(
        &mut self,
        detections: &[Detection],
        det_models: &[CartesianToBearingRange],
    ) -> Result<(), ScenarioError> {
        let ts = self.ts();
        let mut det_used = [false; MAX_GNN_ASSIGN_DETS];
        let mut act_tails = [self.prior; MAX_ACTIVE];
        let mut act_map = [0usize; MAX_ACTIVE];
        let mut n_act = 0usize;
        for (i, slot) in self.active.iter().enumerate() {
            if let Some(t) = slot {
                act_tails[n_act] = t.tail;
                act_map[n_act] = i;
                n_act += 1;
            }
        }
        if n_act > 0 && !detections.is_empty() {
            let mut assigns = [GnnAssignment {
                hypothesis: MeasurementHypothesis {
                    prediction: act_tails[0],
                    detection_index: None,
                    distance: GATE,
                },
            }; MAX_ACTIVE];
            associate_gnn_2d_ekf_det_models(
                &act_tails[..n_act],
                ts,
                detections,
                GATE,
                &self.predictor,
                &self.updater,
                Some(det_models),
                &mut self.ws,
                &mut assigns[..n_act],
            )
            .map_err(ScenarioError::Core)?;
            for t in 0..n_act {
                let slot = act_map[t];
                let h = &assigns[t].hypothesis;
                let track = self.active[slot]
                    .as_mut()
                    .ok_or(ScenarioError::Core(Error::InvalidDimension))?;
                if let Some(di) = h.detection_index {
                    self.updater.measurement = det_models[di];
                    let post = self
                        .updater
                        .update(&h.prediction, &detections[di], &mut self.ws)
                        .map_err(ScenarioError::Core)?;
                    track.tail = post;
                    track.last_meas_step = self.step;
                    det_used[di] = true;
                } else {
                    track.tail = h.prediction;
                }
            }
        }
        self.delete_stale_active()?;
        let mut unassoc = [Detection {
            vector: stonesoup_core::VecN([0.0, 0.0]),
            timestamp: ts,
            sensor: SensorId(0),
        }; MAX_GNN_ASSIGN_DETS];
        let mut n_unassoc = 0usize;
        for (i, det) in detections.iter().enumerate() {
            if !det_used[i] && n_unassoc < MAX_GNN_ASSIGN_DETS {
                unassoc[n_unassoc] = *det;
                n_unassoc += 1;
            }
        }
        if n_unassoc > 0 {
            let mut un_models = [det_models[0]; MAX_GNN_ASSIGN_DETS];
            let mut ui = 0usize;
            for (i, used) in det_used.iter().enumerate().take(detections.len()) {
                if !*used {
                    un_models[ui] = det_models[i];
                    ui += 1;
                }
            }
            self.multi_initiate(&unassoc[..n_unassoc], &un_models[..n_unassoc])?;
        }
        self.snapshot_active_paths(ts);
        Ok(())
    }

    pub fn run(&mut self) -> Result<DualRadarStats, ScenarioError> {
        let max_steps = max_simulation_steps();
        while self.step < max_steps {
            if self.step > 0 {
                self.advance_truths()?;
            }
            let ts = self.ts();
            self.log_truths_at(ts);
            self.prior.timestamp = ts;
            let mut det_buf = [Detection {
                vector: stonesoup_core::VecN([0.0, 0.0]),
                timestamp: ts,
                sensor: SensorId(0),
            }; MAX_GNN_ASSIGN_DETS];
            let mut det_models = [radar_model(0.0)?; MAX_GNN_ASSIGN_DETS];
            let mut n_dets = 0usize;
            for ri in 0..2 {
                let mut chunk = [Detection {
                    vector: stonesoup_core::VecN([0.0, 0.0]),
                    timestamp: ts,
                    sensor: SensorId(0),
                }; MAX_GNN_ASSIGN_DETS];
                let got = self.generate_sensor_dets(ri, ts, &mut chunk)?;
                let model = radar_model(self.radar_rt[ri].dwell_rad)?;
                for chunk_det in chunk.iter().take(got) {
                    if n_dets >= MAX_GNN_ASSIGN_DETS {
                        break;
                    }
                    det_buf[n_dets] = *chunk_det;
                    det_models[n_dets] = model;
                    n_dets += 1;
                }
            }
            if n_dets > 0 {
                self.tracker_update(&det_buf[..n_dets], &det_models[..n_dets])?;
            } else {
                self.delete_stale_active()?;
                self.snapshot_active_paths(ts);
            }
            self.step += 1;
        }
        let step = self.step;
        let remaining: [Option<ActiveTrack>; MAX_ACTIVE] = self.active;
        for t in remaining.into_iter().flatten() {
            self.finish_track(t, step)?;
        }
        self.active = [None; MAX_ACTIVE];
        let mut completed = Vec::new();
        for i in 0..self.n_completed {
            if let Some(mut c) = self.completed[i] {
                let mut best = TRUTH_ASSOC_METERS;
                let mut best_id = None;
                for (ti, gt) in self.truths.iter().enumerate() {
                    let d = euclidean_xy(c.end_xy, (gt.state[0], gt.state[2]));
                    if d < best {
                        best = d;
                        best_id = Some(ti as u8);
                    }
                }
                if best <= TRUTH_ASSOC_METERS {
                    c.truth_id = best_id;
                }
                completed.push(c);
            }
        }
        let truths_with_track = completed
            .iter()
            .filter(|c| c.truth_id.is_some())
            .fold(0u8, |acc, c| acc | (1 << c.truth_id.unwrap_or(0)));

        let truth_paths: Vec<TruthPath> = self
            .truth_log
            .iter()
            .enumerate()
            .map(|(i, states)| TruthPath {
                id: i as u8,
                states: states.clone(),
            })
            .collect();
        let tt_config = TrackToTruthConfig {
            association_threshold_m: TRUTH_ASSOC_METERS,
            consec_pairs_confirm: 3,
            consec_misses_end: 2,
        };
        let track_to_truth =
            associate_tracks_to_truths(&self.filter_paths, &truth_paths, &tt_config);
        let continuous_truth_tracks =
            continuous_tracks_per_truth(&track_to_truth, NUM_TRUTHS as u8);

        for ri in 0..2 {
            let rt = &mut self.radar_rt[ri];
            if rt.plots_this_rotation > rt.max_plots_in_rotation {
                rt.max_plots_in_rotation = rt.plots_this_rotation;
            }
        }

        Ok(DualRadarStats {
            radar_a_plots: self.plot_counts[0],
            radar_b_plots: self.plot_counts[1],
            radar_a_rotations: self.radar_rt[0].rotation_count,
            radar_b_rotations: self.radar_rt[1].rotation_count,
            radar_a_max_plots_in_rotation: self.radar_rt[0].max_plots_in_rotation,
            radar_b_max_plots_in_rotation: self.radar_rt[1].max_plots_in_rotation,
            total_true: self.total_true,
            total_clutter: self.total_clutter,
            simulation_steps: self.step,
            truths_with_plot: self.truths_detected,
            truths_with_track,
            fused_completed: completed,
            true_plot_tracks: self.true_plot_tracks.clone(),
            true_plots_in_overlap: self.true_plots_in_overlap,
            track_to_truth,
            continuous_truth_tracks,
        })
    }
}

fn truth_step_nanos() -> i64 {
    (DT_S * 1e9) as i64
}

pub fn max_simulation_steps() -> usize {
    libm::ceil(SIM_DURATION_S / DT_S) as usize
}

fn count_bits(x: u8) -> u8 {
    x.count_ones() as u8
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

    pub fn truths_with_plot_count(&self) -> u8 {
        count_bits(self.truths_with_plot)
    }

    pub fn truths_with_track_count(&self) -> u8 {
        count_bits(self.truths_with_track)
    }
}
