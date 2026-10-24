//! Accumulated state densities (ASD) Kalman predict/update (Stone Soup v1.9.1).

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::error::Error;
use crate::filter::kalman::{KalmanPredictor, KalmanUpdater, KalmanWorkspace};
use crate::linalg::fixed::{
    add4x4, add_vec4, inv4, mat4_vec4, mul4x4, sub_vec2, transpose4,
};
use crate::linalg::{Mat, VecN};
use crate::models::measurement::LinearGaussianMeasurement;
use crate::models::transition::ConstantVelocity2d;
use crate::state::GaussianState;
use crate::time::{Duration, Timestamp};
use crate::types::Detection;

pub const ASD_NDIM: usize = 4;
pub const ASD_DEFAULT_MAX_NSTEP: usize = 30;

/// One correlation-matrix slot aligned with [`AsdGaussianState::timestamps`].
#[derive(Debug, Clone)]
pub struct CorrelationEntry {
    pub p: Mat<4, 4>,
    pub p_pred: Option<Mat<4, 4>>,
    pub f: Option<Mat<4, 4>>,
    pub pfp: Option<Mat<4, 4>>,
}

impl CorrelationEntry {
    pub fn with_p(p: Mat<4, 4>) -> Self {
        Self {
            p,
            p_pred: None,
            f: None,
            pfp: None,
        }
    }

}

/// ASD multi-state (newest timestamp first in [`Self::timestamps`]).
#[derive(Debug, Clone)]
pub struct AsdGaussianState {
    pub multi_state: Vec<f64>,
    pub multi_covar: Vec<f64>,
    pub timestamps: Vec<Timestamp>,
    pub correlation_matrices: Vec<CorrelationEntry>,
    pub max_nstep: usize,
}

impl AsdGaussianState {
    pub fn new(
        multi_state: Vec<f64>,
        multi_covar: Vec<f64>,
        timestamps: Vec<Timestamp>,
        max_nstep: usize,
    ) -> Result<Self, Error> {
        let nstep = timestamps.len();
        if nstep == 0 {
            return Err(Error::InvalidParameter);
        }
        let n = nstep * ASD_NDIM;
        if multi_state.len() != n || multi_covar.len() != n * n {
            return Err(Error::InvalidParameter);
        }
        Ok(Self {
            multi_state,
            multi_covar,
            timestamps,
            correlation_matrices: Vec::new(),
            max_nstep,
        })
    }

    pub fn from_single(
        mean: VecN<4>,
        cov: Mat<4, 4>,
        timestamp: Timestamp,
        max_nstep: usize,
    ) -> Result<Self, Error> {
        let _ = GaussianState::new(mean, cov, timestamp)?;
        let mut multi_covar = vec![0.0; ASD_NDIM * ASD_NDIM];
        for r in 0..ASD_NDIM {
            for c in 0..ASD_NDIM {
                multi_covar[r * ASD_NDIM + c] = cov.0[r][c];
            }
        }
        Ok(Self {
            multi_state: mean.0.to_vec(),
            multi_covar,
            timestamps: vec![timestamp],
            correlation_matrices: Vec::new(),
            max_nstep,
        })
    }

    pub fn nstep(&self) -> usize {
        self.timestamps.len()
    }

    pub fn timestamp(&self) -> Timestamp {
        self.timestamps[0]
    }

    pub fn index_of_timestamp(&self, ts: Timestamp) -> Result<usize, Error> {
        self.timestamps
            .iter()
            .position(|&t| t == ts)
            .ok_or(Error::InvalidParameter)
    }

    pub fn state_at(&self, t_index: usize) -> Result<GaussianState, Error> {
        if t_index >= self.nstep() {
            return Err(Error::InvalidParameter);
        }
        let off = t_index * ASD_NDIM;
        let mut mean = [0.0; 4];
        mean.copy_from_slice(&self.multi_state[off..off + ASD_NDIM]);
        let cov = block4(&self.multi_covar, self.stride(), off, off)?;
        GaussianState::new(VecN(mean), cov, self.timestamps[t_index])
    }

    fn stride(&self) -> usize {
        self.nstep() * ASD_NDIM
    }
}

#[derive(Debug, Clone)]
pub struct AsdPrediction {
    pub state: AsdGaussianState,
    pub act_timestamp: Timestamp,
    /// Index of the timestamp used as the prediction origin (Stone Soup `_predict_over_interval`).
    pub origin_index: usize,
}

#[derive(Debug, Clone)]
pub struct AsdMeasurementPrediction {
    pub mean: VecN<2>,
    pub innov_cov: Mat<2, 2>,
    pub cross_covar: Vec<f64>,
    pub cross_rows: usize,
}

#[derive(Debug, Clone)]
pub struct AsdKalmanPredictor {
    pub transition: ConstantVelocity2d,
    inner: KalmanPredictor,
}

impl AsdKalmanPredictor {
    pub fn new(transition: ConstantVelocity2d) -> Self {
        let inner = KalmanPredictor::new(transition);
        Self { transition, inner }
    }

    fn predict_interval(
        prior: &AsdGaussianState,
        timestamp: Timestamp,
    ) -> Result<(Duration, Timestamp), Error> {
        let mut from_ts = prior.timestamp();
        let mut predict_over = timestamp.checked_sub(from_ts)?;
        if predict_over.0 < 0 {
            let mut best: Option<(Duration, Timestamp)> = None;
            for &t in &prior.timestamps {
                let d = timestamp.checked_sub(t)?;
                if d.0 >= 0 {
                    let pick = match best {
                        None => true,
                        Some((bd, _)) => d.0 < bd.0,
                    };
                    if pick {
                        best = Some((d, t));
                    }
                }
            }
            let (d, t) = best.ok_or(Error::InvalidTimeStep)?;
            predict_over = d;
            from_ts = t;
        }
        Ok((predict_over, from_ts))
    }

    fn generate_c_matrices(corrs: &[CorrelationEntry]) -> Result<Vec<Mat<4, 4>>, Error> {
        let mut out = Vec::with_capacity(corrs.len());
        out.push(crate::linalg::fixed::identity4()?);
        for item in corrs.iter().skip(1) {
            let pfp = item.pfp.ok_or(Error::InvalidParameter)?;
            let prev = *out.last().ok_or(Error::InvalidParameter)?;
            out.push(mul4x4(prev, pfp)?);
        }
        Ok(out)
    }

    fn predict_single(
        &self,
        prior: &GaussianState,
        dt: Duration,
        ws: &mut KalmanWorkspace,
    ) -> Result<GaussianState, Error> {
        let ts = prior
            .timestamp
            .checked_add(dt)
            .map_err(|_| Error::InvalidTimeStep)?;
        self.inner.predict(prior, ts, ws)
    }

    pub fn predict(
        &self,
        prior: &AsdGaussianState,
        timestamp: Timestamp,
        ws: &mut KalmanWorkspace,
    ) -> Result<AsdPrediction, Error> {
        let mut correlation_matrices = prior.correlation_matrices.clone();
        let (predict_over_interval, timestamp_from) = Self::predict_interval(prior, timestamp)?;
        let t_index = prior.index_of_timestamp(timestamp_from)?;

        if correlation_matrices.is_empty() {
            let p0 = prior.state_at(0)?.cov;
            correlation_matrices.push(CorrelationEntry::with_p(p0));
        }

        let ndim = ASD_NDIM;
        let nstep = prior.nstep();
        let stride = nstep * ndim;

        let (x_pred, p_pred) = if t_index == 0 {
            let prior_g = prior.state_at(0)?;
            let x_pred_m = self.predict_single(&prior_g, predict_over_interval, ws)?;
            let p_pred_m = x_pred_m.cov;
            let transition_matrix = ws.f;

            let combined =
                Self::generate_c_matrices(&correlation_matrices)?;
            let col_rows = combined.len() * ndim;
            let mut wp_col = vec![0.0; col_rows * ndim];
            for (i, c) in combined.iter().enumerate() {
                let block = mul4x4(*c, prior_g.cov)?;
                write_block4(&mut wp_col, ndim, i * ndim, 0, &block);
            }
            let f_t = transpose4(transition_matrix)?;
            let mut correlated_column = vec![0.0; col_rows * ndim];
            mat_mul(
                col_rows,
                ndim,
                ndim,
                &wp_col,
                ndim,
                ndim,
                &transition_matrix_to_flat(f_t),
                &mut correlated_column,
            )?;
            let correlated_row = transpose_vec(&correlated_column, col_rows, ndim);

            let mut x_pred = x_pred_m.mean.0.to_vec();
            x_pred.extend_from_slice(&prior.multi_state);

            let top_cols = stride + ndim;
            let mut p_pred = vec![0.0; top_cols * top_cols];
            write_block4(&mut p_pred, top_cols, 0, 0, &p_pred_m);
            write_block_flat(
                &mut p_pred,
                top_cols,
                0,
                ndim,
                col_rows,
                ndim,
                &correlated_row,
            );
            write_block_flat(
                &mut p_pred,
                top_cols,
                ndim,
                0,
                col_rows,
                ndim,
                &correlated_column,
            );
            write_block_flat(
                &mut p_pred,
                top_cols,
                ndim,
                ndim,
                stride,
                stride,
                &prior.multi_covar,
            );

            correlation_matrices[t_index] = {
                let mut e = correlation_matrices[t_index].clone();
                e.p_pred = Some(p_pred_m);
                e.f = Some(transition_matrix);
                let pfp = mul4x4(
                    mul4x4(e.p, transpose4(transition_matrix)?)?,
                    inv4(p_pred_m)?,
                )?;
                e.pfp = Some(pfp);
                e
            };
            correlation_matrices.insert(t_index, CorrelationEntry::with_p(p_pred_m));

            (x_pred, p_pred)
        } else {
            let timestamp_m_plus_1 = prior.timestamps[t_index - 1];
            let time_interval_m_to_m_plus_1 = timestamp_m_plus_1.checked_sub(timestamp)?;

            let prior_at_t = prior.state_at(t_index)?;
            let x_pred_m = self.predict_single(&prior_at_t, predict_over_interval, ws)?;
            let p_pred_m = x_pred_m.cov;
            let state_pred_m = GaussianState::new(x_pred_m.mean, p_pred_m, timestamp)?;

            let x_pred_m_plus_1 =
                self.predict_single(&state_pred_m, time_interval_m_to_m_plus_1, ws)?;
            let p_pred_m_plus_1 = x_pred_m_plus_1.cov;

            let transition_matrix_m = ws.f;
            let transition_matrix_m_plus_1 = {
                let _ = self.predict_single(&state_pred_m, time_interval_m_to_m_plus_1, ws)?;
                ws.f
            };

            let off_m1 = (t_index - 1) * ndim;
            let x_m_plus_1_given_k = array4(&prior.multi_state[off_m1..off_m1 + ndim]);
            let x_diff = sub_vec4_n(x_m_plus_1_given_k, x_pred_m_plus_1.mean);

            let p_m_plus_1_given_k =
                block4(&prior.multi_covar, stride, off_m1, off_m1)?;
            let p_diff = mat4_sub(p_m_plus_1_given_k, p_pred_m_plus_1)?;

            let w = mul4x4(
                mul4x4(p_pred_m, transpose4(transition_matrix_m_plus_1)?)?,
                inv4(p_pred_m_plus_1)?,
            )?;
            let x_pred_m_given_k_mean = add_vec4(x_pred_m.mean, mat4_vec4(w, x_diff)?)?;
            let p_pred_m_given_k = add4x4(
                p_pred_m,
                mul4x4(mul4x4(w, p_diff)?, transpose4(w)?)?,
            )?;

            let mut x_pred = Vec::with_capacity(prior.multi_state.len() + ndim);
            x_pred.extend_from_slice(&prior.multi_state[..t_index * ndim]);
            x_pred.extend_from_slice(&x_pred_m_given_k_mean.0);
            x_pred.extend_from_slice(&prior.multi_state[t_index * ndim..]);

            let p_state_pred = GaussianState::new(
                x_pred_m.mean,
                correlation_matrices[t_index].p,
                prior.timestamp(),
            )?;
            let p_p_pred = self
                .predict_single(&p_state_pred, predict_over_interval, ws)?
                .cov;

            correlation_matrices[t_index] = {
                let mut e = correlation_matrices[t_index].clone();
                e.p_pred = Some(p_p_pred);
                e.f = Some(transition_matrix_m);
                let pfp = mul4x4(
                    mul4x4(e.p, transpose4(transition_matrix_m)?)?,
                    inv4(p_p_pred)?,
                )?;
                e.pfp = Some(pfp);
                e
            };
            correlation_matrices.insert(t_index, CorrelationEntry {
                p: p_pred_m,
                p_pred: Some(p_pred_m_plus_1),
                f: Some(transition_matrix_m_plus_1),
                pfp: Some(mul4x4(
                    mul4x4(p_pred_m, transpose4(transition_matrix_m_plus_1)?)?,
                    inv4(p_pred_m_plus_1)?,
                )?),
            });

            let new_stride = (nstep + 1) * ndim;
            let mut p_pred = vec![0.0; new_stride * new_stride];
            let src_off = t_index * ndim;
            let block_size = stride - src_off;
            let right_lower = block_slice(
                &prior.multi_covar,
                stride,
                src_off,
                src_off,
                block_size,
                block_size,
            );
            write_block_flat(
                &mut p_pred,
                new_stride,
                (t_index + 1) * ndim,
                (t_index + 1) * ndim,
                block_size,
                block_size,
                &right_lower,
            );

            let mut timestamps_to_recalculate: Vec<Timestamp> =
                prior.timestamps[..t_index].to_vec();
            timestamps_to_recalculate.push(timestamp);

            let mut covars: Vec<Mat<4, 4>> = (0..t_index)
                .map(|i| {
                    let off = i * ndim;
                    block4(&prior.multi_covar, stride, off, off)
                })
                .collect::<Result<_, _>>()?;
            covars.push(p_pred_m_given_k);

            for (i, _ts) in timestamps_to_recalculate.iter().enumerate() {
                let corrs = &correlation_matrices[i..];
                let mut combined = Self::generate_c_matrices(corrs)?;
                if !combined.is_empty() {
                    combined.remove(0);
                }
                let col_rows = combined.len() * ndim;
                let mut w_column = vec![0.0; col_rows * ndim];
                for (k, c) in combined.iter().enumerate() {
                    let block = mul4x4(*c, covars[i])?;
                    write_block4(&mut w_column, ndim, k * ndim, 0, &block);
                }
                let w_row = transpose_vec(&w_column, col_rows, ndim);
                let i_off = i * ndim;
                write_block4(&mut p_pred, new_stride, i_off, i_off, &covars[i]);
                write_block_flat(
                    &mut p_pred,
                    new_stride,
                    (i + 1) * ndim,
                    i_off,
                    col_rows,
                    ndim,
                    &w_column,
                );
                write_block_flat(
                    &mut p_pred,
                    new_stride,
                    i_off,
                    (i + 1) * ndim,
                    ndim,
                    col_rows,
                    &w_row,
                );
            }

            (x_pred, p_pred)
        };

        let mut timestamps: Vec<Timestamp> = prior
            .timestamps
            .iter()
            .copied()
            .chain([timestamp])
            .collect();
        timestamps.sort_by(|a, b| b.cmp(a));

        let mut state = AsdGaussianState {
            multi_state: x_pred,
            multi_covar: p_pred,
            timestamps,
            correlation_matrices,
            max_nstep: prior.max_nstep,
        };
        self.prune_state(&mut state);

        Ok(AsdPrediction {
            act_timestamp: timestamp,
            origin_index: t_index,
            state,
        })
    }

    pub fn prune_state(&self, state: &mut AsdGaussianState) {
        if state.max_nstep == 0 || state.nstep() <= state.max_nstep {
            return;
        }
        let index = state.max_nstep * ASD_NDIM;
        state.multi_state.truncate(index);
        let stride = state.stride();
        let mut trimmed = vec![0.0; index * index];
        for r in 0..index {
            for c in 0..index {
                trimmed[r * index + c] = state.multi_covar[r * stride + c];
            }
        }
        state.multi_covar = trimmed;
        state.timestamps.truncate(state.max_nstep);
        state.correlation_matrices.truncate(state.max_nstep);
    }
}

#[derive(Debug, Clone)]
pub struct AsdKalmanUpdater {
    pub measurement: LinearGaussianMeasurement,
    inner: KalmanUpdater,
}

impl AsdKalmanUpdater {
    pub fn new(measurement: LinearGaussianMeasurement) -> Self {
        Self {
            inner: KalmanUpdater::new(measurement),
            measurement,
        }
    }

    pub fn predict_measurement(
        &self,
        predicted: &AsdPrediction,
        ws: &mut KalmanWorkspace,
    ) -> Result<AsdMeasurementPrediction, Error> {
        let t_index = predicted
            .state
            .index_of_timestamp(predicted.act_timestamp)?;
        let state_at_t = predicted.state.state_at(t_index)?;
        let mp = self.inner.predict_measurement(&state_at_t, ws)?;
        let hh = ws.h;
        let stride = predicted.state.stride();
        let mut cross = vec![0.0; stride * 2];
        for row in 0..stride {
            for col in 0..2 {
                let mut acc = 0.0;
                for k in 0..ASD_NDIM {
                    let p = predicted.state.multi_covar[row * stride + (t_index * ASD_NDIM + k)];
                    acc += p * hh.0[col][k];
                }
                cross[row * 2 + col] = acc;
            }
        }
        Ok(AsdMeasurementPrediction {
            mean: mp.mean,
            innov_cov: mp.cov,
            cross_covar: cross,
            cross_rows: stride,
        })
    }

    pub fn update(
        &self,
        predicted: &AsdPrediction,
        detection: &Detection,
        ws: &mut KalmanWorkspace,
    ) -> Result<AsdGaussianState, Error> {
        let mp = self.predict_measurement(predicted, ws)?;
        let innov_cov = mp.innov_cov;
        let innov_cov_inv = crate::linalg::fixed::inv2(innov_cov)?;

        let stride = predicted.state.stride();
        let mut kalman_gain = vec![0.0; stride * 2];
        mat_mul(
            stride,
            2,
            2,
            &mp.cross_covar,
            2,
            2,
            &flat2(&innov_cov_inv),
            &mut kalman_gain,
        )?;

        let innovation = sub_vec2(detection.vector, mp.mean)?;
        let innov_vec = [innovation.0[0], innovation.0[1]];
        let mut mean_correction = vec![0.0; stride];
        for r in 0..stride {
            mean_correction[r] =
                kalman_gain[r * 2] * innov_vec[0] + kalman_gain[r * 2 + 1] * innov_vec[1];
        }
        let mut posterior_mean = predicted.state.multi_state.clone();
        for r in 0..stride {
            posterior_mean[r] += mean_correction[r];
        }

        let mut k_s = vec![0.0; stride * 2];
        mat_mul(
            stride,
            2,
            2,
            &kalman_gain,
            2,
            2,
            &flat2(&innov_cov),
            &mut k_s,
        )?;
        let mut posterior_cov = predicted.state.multi_covar.clone();
        for r in 0..stride {
            for c in 0..stride {
                let mut acc = 0.0;
                for k in 0..2 {
                    acc += k_s[r * 2 + k] * kalman_gain[c * 2 + k];
                }
                posterior_cov[r * stride + c] -= acc;
            }
        }

        let t_index = predicted
            .state
            .index_of_timestamp(predicted.act_timestamp)?;
        let mut correlation_matrices = predicted.state.correlation_matrices.clone();
        let off = t_index * ASD_NDIM;
        correlation_matrices[t_index].p =
            block4(&posterior_cov, stride, off, off)?;
        if let (Some(f), Some(p_pred)) = (
            correlation_matrices[t_index].f,
            correlation_matrices[t_index].p_pred,
        ) {
            let pfp = mul4x4(
                mul4x4(correlation_matrices[t_index].p, transpose4(f)?)?,
                inv4(p_pred)?,
            )?;
            correlation_matrices[t_index].pfp = Some(pfp);
        }

        Ok(AsdGaussianState {
            multi_state: posterior_mean,
            multi_covar: posterior_cov,
            timestamps: predicted.state.timestamps.clone(),
            correlation_matrices,
            max_nstep: predicted.state.max_nstep,
        })
    }
}

fn array4(s: &[f64]) -> VecN<4> {
    VecN([s[0], s[1], s[2], s[3]])
}

fn sub_vec4_n(a: VecN<4>, b: VecN<4>) -> VecN<4> {
    VecN([
        a.0[0] - b.0[0],
        a.0[1] - b.0[1],
        a.0[2] - b.0[2],
        a.0[3] - b.0[3],
    ])
}

fn mat4_sub(a: Mat<4, 4>, b: Mat<4, 4>) -> Result<Mat<4, 4>, Error> {
    let mut out = Mat::<4, 4>::zeros();
    for r in 0..4 {
        for c in 0..4 {
            out.set(r, c, a.0[r][c] - b.0[r][c])?;
        }
    }
    Ok(out)
}

fn transition_matrix_to_flat(f: Mat<4, 4>) -> [f64; 16] {
    let mut out = [0.0; 16];
    for r in 0..4 {
        for c in 0..4 {
            out[r * 4 + c] = f.0[r][c];
        }
    }
    out
}

fn flat2(m: &Mat<2, 2>) -> [f64; 4] {
    [m.0[0][0], m.0[0][1], m.0[1][0], m.0[1][1]]
}

fn block4(data: &[f64], stride: usize, r0: usize, c0: usize) -> Result<Mat<4, 4>, Error> {
    let mut m = Mat::<4, 4>::zeros();
    for r in 0..4 {
        for c in 0..4 {
            m.set(r, c, data[(r0 + r) * stride + (c0 + c)])?;
        }
    }
    Ok(m)
}

fn write_block4(dst: &mut [f64], stride: usize, r0: usize, c0: usize, block: &Mat<4, 4>) {
    for r in 0..4 {
        for c in 0..4 {
            dst[(r0 + r) * stride + (c0 + c)] = block.0[r][c];
        }
    }
}

fn block_slice(
    data: &[f64],
    stride: usize,
    r0: usize,
    c0: usize,
    rows: usize,
    cols: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = data[(r0 + r) * stride + (c0 + c)];
        }
    }
    out
}

fn write_block_flat(
    dst: &mut [f64],
    dst_stride: usize,
    r0: usize,
    c0: usize,
    rows: usize,
    cols: usize,
    src: &[f64],
) {
    for r in 0..rows {
        for c in 0..cols {
            dst[(r0 + r) * dst_stride + (c0 + c)] = src[r * cols + c];
        }
    }
}

fn transpose_vec(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

fn mat_mul(
    a_rows: usize,
    a_cols: usize,
    b_cols: usize,
    a: &[f64],
    _b_rows: usize,
    _b_cols: usize,
    b: &[f64],
    out: &mut [f64],
) -> Result<(), Error> {
    if a.len() != a_rows * a_cols || b.len() != a_cols * b_cols || out.len() != a_rows * b_cols {
        return Err(Error::InvalidParameter);
    }
    for r in 0..a_rows {
        for c in 0..b_cols {
            let mut acc = 0.0;
            for k in 0..a_cols {
                acc += a[r * a_cols + k] * b[k * b_cols + c];
            }
            out[r * b_cols + c] = acc;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::fixed::identity4;

    #[test]
    fn single_step_asd_matches_kalman() {
        let transition = ConstantVelocity2d::new(0.05, 0.05).expect("q");
        let mut r = Mat::<2, 2>::zeros();
        r.set(0, 0, 5.0).unwrap();
        r.set(1, 1, 5.0).unwrap();
        let meas = LinearGaussianMeasurement::position_2d(r).expect("m");
        let pred = AsdKalmanPredictor::new(transition);
        let upd = AsdKalmanUpdater::new(meas);
        let mut ws = KalmanWorkspace {
            f: Mat::zeros(),
            q: Mat::zeros(),
            h: Mat::zeros(),
            r: Mat::zeros(),
        };
        let prior = AsdGaussianState::from_single(
            VecN([20.0, 1.0, 20.0, 1.0]),
            crate::linalg::fixed::identity4().expect("i"),
            Timestamp::ZERO,
            30,
        )
        .expect("prior");
        let ts = Timestamp::from_nanos(Duration::SEC.0);
        let prediction = pred.predict(&prior, ts, &mut ws).expect("predict");
        let det = Detection::new([21.0, 19.0], ts, crate::types::SensorId(0)).expect("det");
        let post = upd.update(&prediction, &det, &mut ws).expect("update");
        let act = post.state_at(0).expect("act");
        assert!((act.mean.0[0] - 20.0).abs() < 5.0);
    }
}
