//! Multi-target tracker with ``MultiMeasurementInitiator`` (Tutorials 09–10).

use stonesoup_core::{
    associate_gnn_2d, initiate_from_detection, should_delete_covariance_trace,
    ConstantVelocity2d, Detection, Duration, GaussianState, GnnAssignment, KalmanPredictor,
    KalmanUpdater, KalmanWorkspace, LinearGaussianMeasurement, Mat, MeasurementHypothesis,
    Timestamp, MAX_GNN_ASSIGN_DETS, MAX_GNN_ASSIGN_TRACKS,
};

use crate::lot3_meas::push_measurements;
use crate::registry::ScenarioError;

pub const MAX_LOT4_ACTIVE_TRACKS: usize = 16;
pub const MAX_LOT4_HOLDING_TRACKS: usize = 16;

struct HoldingTrack {
    tail: GaussianState,
    updates: u8,
}

pub struct Lot4TrackerParams {
    pub gate: f64,
    pub deleter_trace: f64,
    pub min_points: u8,
    pub step_seconds: u32,
    pub process_noise: f64,
    pub meas_var: f64,
    pub prior: GaussianState,
}

pub struct Lot4MultiTargetTracker {
    params: Lot4TrackerParams,
    predictor: KalmanPredictor,
    updater: KalmanUpdater,
    ws: KalmanWorkspace,
    active: [Option<GaussianState>; MAX_LOT4_ACTIVE_TRACKS],
    holding: [Option<HoldingTrack>; MAX_LOT4_HOLDING_TRACKS],
    step: usize,
    max_steps: usize,
    meas_offsets: &'static [usize],
    meas_xy: &'static [[f64; 2]],
}

impl Lot4MultiTargetTracker {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn new(
        params: Lot4TrackerParams,
        max_steps: usize,
        meas_offsets: &'static [usize],
        meas_xy: &'static [[f64; 2]],
    ) -> Result<Self, ScenarioError> {
        let transition =
            ConstantVelocity2d::new(params.process_noise, params.process_noise)
                .map_err(ScenarioError::Core)?;
        let predictor = KalmanPredictor::new(transition);
        let mut r = Mat::<2, 2>::zeros();
        r.set(0, 0, params.meas_var).map_err(ScenarioError::Core)?;
        r.set(1, 1, params.meas_var).map_err(ScenarioError::Core)?;
        let measurement =
            LinearGaussianMeasurement::position_2d(r).map_err(ScenarioError::Core)?;
        let updater = KalmanUpdater::new(measurement);
        Ok(Self {
            params,
            predictor,
            updater,
            ws: KalmanWorkspace {
                f: Mat::zeros(),
                q: Mat::zeros(),
                h: Mat::zeros(),
                r: Mat::zeros(),
            },
            active: [None; MAX_LOT4_ACTIVE_TRACKS],
            holding: [const { None }; MAX_LOT4_HOLDING_TRACKS],
            step: 0,
            max_steps,
            meas_offsets,
            meas_xy,
        })
    }

    fn ts(&self, step: usize) -> Timestamp {
        Timestamp::from_nanos((step as i64) * (self.params.step_seconds as i64) * Duration::SEC.0)
    }

    pub fn step_index(&self) -> usize {
        self.step
    }

    pub fn reset(&mut self) -> Result<(), ScenarioError> {
        self.active = [None; MAX_LOT4_ACTIVE_TRACKS];
        self.holding = [const { None }; MAX_LOT4_HOLDING_TRACKS];
        self.step = 0;
        Ok(())
    }

    fn insert_active(&mut self, tail: GaussianState) -> Result<(), ScenarioError> {
        for slot in &mut self.active {
            if slot.is_none() {
                *slot = Some(tail);
                return Ok(());
            }
        }
        Err(ScenarioError::Core(stonesoup_core::Error::CapacityExceeded))
    }

    fn insert_holding(&mut self, tail: GaussianState) -> Result<(), ScenarioError> {
        for slot in &mut self.holding {
            if slot.is_none() {
                *slot = Some(HoldingTrack { tail, updates: 1 });
                return Ok(());
            }
        }
        Err(ScenarioError::Core(stonesoup_core::Error::CapacityExceeded))
    }

    fn delete_active(&mut self) -> Result<(), ScenarioError> {
        for slot in &mut self.active {
            if let Some(tail) = slot {
                if should_delete_covariance_trace(tail, self.params.deleter_trace)
                    .map_err(ScenarioError::Core)?
                {
                    *slot = None;
                }
            }
        }
        Ok(())
    }

    fn delete_holding(&mut self) -> Result<(), ScenarioError> {
        for slot in &mut self.holding {
            if let Some(ht) = slot {
                if should_delete_covariance_trace(&ht.tail, self.params.deleter_trace)
                    .map_err(ScenarioError::Core)?
                {
                    *slot = None;
                }
            }
        }
        Ok(())
    }

    /// Stone Soup ``MultiMeasurementInitiator.initiate`` — returns confirmed track tails.
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    fn multi_measurement_initiate(
        &mut self,
        detections: &[Detection],
        ts: Timestamp,
    ) -> Result<[Option<GaussianState>; MAX_LOT4_ACTIVE_TRACKS], ScenarioError> {
        let mut sure = [None; MAX_LOT4_ACTIVE_TRACKS];
        let mut n_sure = 0usize;
        let mut det_used = [false; MAX_GNN_ASSIGN_DETS];

        let mut hold_tails = [self.params.prior; MAX_LOT4_HOLDING_TRACKS];
        let mut hold_map = [0usize; MAX_LOT4_HOLDING_TRACKS];
        let mut n_hold = 0usize;
        for (i, slot) in self.holding.iter().enumerate() {
            if let Some(h) = slot {
                hold_tails[n_hold] = h.tail;
                hold_map[n_hold] = i;
                n_hold += 1;
            }
        }

        if n_hold > 0 {
            let mut assigns = [GnnAssignment {
                hypothesis: MeasurementHypothesis {
                    prediction: hold_tails[0],
                    detection_index: None,
                    distance: self.params.gate,
                },
            }; MAX_GNN_ASSIGN_TRACKS];
            associate_gnn_2d(
                &hold_tails[..n_hold],
                ts,
                detections,
                self.params.gate,
                &self.predictor,
                &self.updater,
                &mut self.ws,
                &mut assigns,
            )
            .map_err(ScenarioError::Core)?;

            for t in 0..n_hold {
                let slot = hold_map[t];
                let h = &assigns[t].hypothesis;
                let ht = self.holding[slot]
                    .as_mut()
                    .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
                if let Some(di) = h.detection_index {
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

            for slot in 0..MAX_LOT4_HOLDING_TRACKS {
                if let Some(ht) = self.holding[slot].as_ref() {
                    if ht.updates >= self.params.min_points {
                        let tail = self.holding[slot]
                            .take()
                            .map(|h| h.tail)
                            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
                        if n_sure < MAX_LOT4_ACTIVE_TRACKS {
                            sure[n_sure] = Some(tail);
                            n_sure += 1;
                        } else {
                            return Err(ScenarioError::Core(stonesoup_core::Error::CapacityExceeded));
                        }
                    }
                }
            }
            self.delete_holding()?;
        }

        for (i, det) in detections.iter().enumerate() {
            if det_used[i] {
                continue;
            }
            let init = initiate_from_detection(
                &self.params.prior,
                det,
                &self.updater,
                &mut self.ws,
            )
            .map_err(ScenarioError::Core)?;
            self.insert_holding(init)?;
        }

        Ok(sure)
    }

    /// One tracker iteration (``MultiTargetTracker.__next__``).
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn step(&mut self) -> Result<bool, ScenarioError> {
        if self.step >= self.max_steps {
            return Ok(false);
        }
        let ts = self.ts(self.step);
        let mut det_buf = [Detection {
            vector: stonesoup_core::VecN([0.0, 0.0]),
            timestamp: ts,
            sensor: stonesoup_core::SensorId(0),
        }; MAX_GNN_ASSIGN_DETS];
        let count = push_measurements(
            self.meas_offsets,
            self.meas_xy,
            self.step,
            ts,
            &mut det_buf,
        )?;
        let detections = &det_buf[..count];
        let mut det_used = [false; MAX_GNN_ASSIGN_DETS];

        let mut act_tails = [self.params.prior; MAX_LOT4_ACTIVE_TRACKS];
        let mut act_map = [0usize; MAX_LOT4_ACTIVE_TRACKS];
        let mut n_act = 0usize;
        for (i, slot) in self.active.iter().enumerate() {
            if let Some(tail) = slot {
                act_tails[n_act] = *tail;
                act_map[n_act] = i;
                n_act += 1;
            }
        }

        if n_act > 0 {
            let mut assigns = [GnnAssignment {
                hypothesis: MeasurementHypothesis {
                    prediction: act_tails[0],
                    detection_index: None,
                    distance: self.params.gate,
                },
            }; MAX_GNN_ASSIGN_TRACKS];
            associate_gnn_2d(
                &act_tails[..n_act],
                ts,
                detections,
                self.params.gate,
                &self.predictor,
                &self.updater,
                &mut self.ws,
                &mut assigns,
            )
            .map_err(ScenarioError::Core)?;

            for t in 0..n_act {
                let slot = act_map[t];
                let h = &assigns[t].hypothesis;
                if let Some(di) = h.detection_index {
                    let post = self
                        .updater
                        .update(&h.prediction, &detections[di], &mut self.ws)
                        .map_err(ScenarioError::Core)?;
                    self.active[slot] = Some(post);
                    det_used[di] = true;
                } else {
                    self.active[slot] = Some(h.prediction);
                }
            }
        }

        self.delete_active()?;

        let mut unassoc = [Detection {
            vector: stonesoup_core::VecN([0.0, 0.0]),
            timestamp: ts,
            sensor: stonesoup_core::SensorId(0),
        }; MAX_GNN_ASSIGN_DETS];
        let mut n_unassoc = 0usize;
        for (i, det) in detections.iter().enumerate() {
            if !det_used[i] {
                unassoc[n_unassoc] = *det;
                n_unassoc += 1;
            }
        }

        let confirmed = self.multi_measurement_initiate(&unassoc[..n_unassoc], ts)?;
        for tail in confirmed.into_iter().flatten() {
            self.insert_active(tail)?;
        }

        self.step += 1;
        Ok(self.step < self.max_steps)
    }

    pub fn run_to_end(&mut self) -> Result<(), ScenarioError> {
        while self.step()? {}
        Ok(())
    }

    /// Active track tails sorted by ``(x, y)`` like Stone Soup export ``track_means``.
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn sorted_track_means(&self, out: &mut [[f64; 4]; MAX_LOT4_ACTIVE_TRACKS]) -> usize {
        let mut n = 0usize;
        for t in self.active.iter().flatten() {
            let m = t.mean.0;
            out[n] = [m[0], m[1], m[2], m[3]];
            n += 1;
        }
        for i in 1..n {
            let mut j = i;
            while j > 0 {
                let a = out[j];
                let b = out[j - 1];
                let ord = a[0]
                    .partial_cmp(&b[0])
                    .unwrap_or(core::cmp::Ordering::Equal)
                    .then(
                        a[2]
                            .partial_cmp(&b[2])
                            .unwrap_or(core::cmp::Ordering::Equal),
                    );
                if ord == core::cmp::Ordering::Less {
                    out[j] = b;
                    out[j - 1] = a;
                    j -= 1;
                } else {
                    break;
                }
            }
        }
        n
    }
}
