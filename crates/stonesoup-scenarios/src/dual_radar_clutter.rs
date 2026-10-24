//! Two co-located rotating bearing–range radars, clutter, fused multi-target EKF tracker.

extern crate alloc;

use alloc::vec::Vec;

use stonesoup_core::linalg::Mat;
use stonesoup_core::{
    associate_gnn_2d_ekf_det_models, mod_bearing, CartesianToBearingRange, ConstantVelocity2d,
    Detection, Duration, Error, ExtendedKalmanUpdater, GaussianState, GnnAssignment,
    KalmanPredictor, KalmanWorkspace, MeasurementHypothesis, SensorId, Timestamp,
    MAX_GNN_ASSIGN_DETS,
};
use stonesoup_core::GroundTruthState;
use stonesoup_sim::{SplitMix64, TruthStepConfig};

use crate::registry::ScenarioError;

pub const PLOTS_PER_RADAR: usize = 250;
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
pub const NUM_TRUTHS: usize = 4;
pub const GATE: f64 = 6.0;
/// One-point initiation so each plot can spawn its own short track (plot-driven STT).
pub const MIN_POINTS: u8 = 1;
/// Drop tracks quickly after a missed plot so ~one completed segment per plot is possible.
pub const COAST_SECONDS: f64 = 0.25;
pub const TRUTH_ASSOC_METERS: f64 = 2_000.0;
pub const SEED: u64 = 42;

const MAX_ACTIVE: usize = 32;
const MAX_HOLDING: usize = 256;
const MAX_COMPLETED: usize = 768;

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
}

#[derive(Debug, Clone, Copy)]
struct HoldingTrack {
    tail: GaussianState,
    updates: u8,
}

#[derive(Debug, Clone, Copy)]
struct ActiveTrack {
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
    pub total_true: usize,
    pub total_clutter: usize,
    pub simulation_steps: usize,
    pub truths_with_plot: u8,
    pub truths_with_track: u8,
    pub completed: Vec<CompletedTrack>,
    /// True plots whose range falls inside radar B max range (both sensors’ overlap disk).
    pub true_plots_in_overlap: usize,
    pub track_initiations: usize,
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
    let Some(prev) = rt.last_ts else {
        rt.last_ts = Some(ts);
        return Ok(rt.dwell_rad);
    };
    let dt = ts
        .checked_sub(prev)
        .map_err(ScenarioError::Core)?
        .as_seconds();
    let rps = cfg.rpm / 60.0;
    let angle = rt.dwell_rad + dt * rps * 2.0 * core::f64::consts::PI;
    rt.dwell_rad = mod_bearing(angle).map_err(ScenarioError::Core)?;
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
    track_initiations: usize,
    truth_scan_idx: [usize; 2],
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
                },
                RadarRuntime {
                    dwell_rad: 0.0,
                    last_ts: None,
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
            track_initiations: 0,
            truth_scan_idx: [0, 0],
        })
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

    /// Exactly one plot (detection) for this radar, if budget remains.
    fn generate_next_plot(
        &mut self,
        radar_idx: usize,
        ts: Timestamp,
    ) -> Result<Option<(Detection, CartesianToBearingRange)>, ScenarioError> {
        if self.plot_counts[radar_idx] >= PLOTS_PER_RADAR {
            return Ok(None);
        }
        let cfg = self.radars[radar_idx];
        let dwell = advance_dwell(cfg, &mut self.radar_rt[radar_idx], ts)?;
        let model = radar_model(dwell)?;

        for _ in 0..NUM_TRUTHS {
            let ti = self.truth_scan_idx[radar_idx] % NUM_TRUTHS;
            self.truth_scan_idx[radar_idx] = self.truth_scan_idx[radar_idx].wrapping_add(1);
            let gt = self.truths[ti];
            if is_in_fov(&model, gt.state, cfg.max_range_m)? {
                let det = measure_truth(&model, gt, cfg.sensor, &mut self.rng)?;
                self.plot_counts[radar_idx] += 1;
                self.total_true += 1;
                self.truths_detected |= 1 << ti;
                let range = det.vector.0[1];
                if range <= RANGE_B_M {
                    self.true_plots_in_overlap += 1;
                }
                return Ok(Some((det, model)));
            }
        }

        let n_clutter = poisson(CLUTTER_LAMBDA, &mut self.rng)?;
        if n_clutter > 0 {
            let det = clutter_detection(cfg.max_range_m, ts, cfg.sensor, &mut self.rng)?;
            self.plot_counts[radar_idx] += 1;
            self.total_clutter += 1;
            return Ok(Some((det, model)));
        }

        // No truth in FOV and no clutter this slot — advance time and retry on next step.
        Ok(None)
    }

    fn insert_active(&mut self, tail: GaussianState, step: usize) -> Result<(), ScenarioError> {
        for slot in &mut self.active {
            if slot.is_none() {
                *slot = Some(ActiveTrack {
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

    fn promote_holding(&mut self) -> Result<(), ScenarioError> {
        for slot in 0..MAX_HOLDING {
            if let Some(ht) = self.holding[slot].as_ref() {
                if ht.updates >= MIN_POINTS {
                    let tail = self
                        .holding[slot]
                        .take()
                        .ok_or(ScenarioError::Core(Error::InvalidDimension))?
                        .tail;
                    self.insert_active(tail, self.step)?;
                }
            }
        }
        Ok(())
    }

    /// Plot-driven STT: each plot closes open track segments (≈ one report per plot).
    fn flush_active_tracks(&mut self) -> Result<(), ScenarioError> {
        let step = self.step;
        let snapshot = self.active;
        for t in snapshot.into_iter().flatten() {
            if self.finish_track(t, step).is_err() {
                break;
            }
        }
        self.active = [None; MAX_ACTIVE];
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
                        self.track_initiations += 1;
                    }
                }
            }
        }
        for (i, det) in detections.iter().enumerate() {
            if det_used[i] {
                continue;
            }
            let init = initiate_bearing_range(&self.prior, det, det_models[i])?;
            if self.insert_holding(init).is_ok() {
                self.track_initiations += 1;
            }
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
        Ok(())
    }

    pub fn run(&mut self) -> Result<DualRadarStats, ScenarioError> {
        // Plot-driven loop: up to 500 tracker updates (250 per radar), one plot at a time.
        while self.plot_counts[0] < PLOTS_PER_RADAR || self.plot_counts[1] < PLOTS_PER_RADAR {
            if self.step > 0 {
                self.advance_truths()?;
            }
            let ts = self.ts();
            self.prior.timestamp = ts;
            let mut plot_this_step = false;
            for ri in 0..2 {
                if self.plot_counts[ri] >= PLOTS_PER_RADAR {
                    continue;
                }
                if let Some((det, model)) = self.generate_next_plot(ri, ts)? {
                    self.tracker_update(core::slice::from_ref(&det), core::slice::from_ref(&model))?;
                    self.promote_holding()?;
                    self.flush_active_tracks()?;
                    plot_this_step = true;
                }
            }
            self.step += 1;
            if self.step > 20_000 {
                break;
            }
            if !plot_this_step && self.plot_counts[0] >= PLOTS_PER_RADAR && self.plot_counts[1] >= PLOTS_PER_RADAR {
                break;
            }
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
        Ok(DualRadarStats {
            radar_a_plots: self.plot_counts[0],
            radar_b_plots: self.plot_counts[1],
            total_true: self.total_true,
            total_clutter: self.total_clutter,
            simulation_steps: self.step,
            truths_with_plot: self.truths_detected,
            truths_with_track,
            completed,
            true_plots_in_overlap: self.true_plots_in_overlap,
            track_initiations: self.track_initiations,
        })
    }
}

fn truth_step_nanos() -> i64 {
    (DT_S * 1e9) as i64
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

    pub fn false_tracks(&self) -> usize {
        self.completed.iter().filter(|c| c.truth_id.is_none()).count()
    }

    pub fn true_tracks(&self) -> usize {
        self.completed.iter().filter(|c| c.truth_id.is_some()).count()
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
