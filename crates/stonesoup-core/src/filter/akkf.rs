//! Adaptive kernel Kalman filter (AKKF) — quadratic kernel, Tutorial `AKKF.py`.

#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::erasing_op,
    clippy::identity_op,
    clippy::too_many_arguments
)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::error::Error;
use crate::kernel::{quadratic_gram, QuadraticKernel};
use crate::linalg::dynamic::{mat_vec, DynMat};
use crate::linalg::require_finite;
use crate::models::bearing_only::Cartesian2DToBearing;
use crate::models::transition::ConstantVelocity2d;
use crate::time::{Duration, Timestamp};
use crate::types::Detection;

pub const AKKF_MAX_PARTICLES: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub struct KernelParticleState {
    pub n_particles: usize,
    /// Column-major `4 × n` state particles.
    pub state_cols: Vec<f64>,
    pub weights: Vec<f64>,
    pub kernel_covar: DynMat,
    /// Proposal particles for the next predict (`4 × n`), if present.
    pub proposal_cols: Option<Vec<f64>>,
    pub timestamp: Timestamp,
}

impl KernelParticleState {
    pub fn new_uniform(particles: &[[f64; 4]], timestamp: Timestamp) -> Result<Self, Error> {
        let n = particles.len();
        if n == 0 || n > AKKF_MAX_PARTICLES {
            return Err(Error::InvalidDimension);
        }
        let w = 1.0 / (n as f64);
        let mut state_cols = vec![0.0; 4 * n];
        for (i, p) in particles.iter().enumerate() {
            for d in 0..4 {
                state_cols[d * n + i] = p[d];
            }
        }
        let weights = vec![w; n];
        let mut kernel_covar = DynMat::identity(n)?;
        kernel_covar = kernel_covar.scale(w)?;
        Ok(Self {
            n_particles: n,
            state_cols,
            weights,
            kernel_covar,
            proposal_cols: None,
            timestamp,
        })
    }

    pub fn from_parts(
        state_cols: Vec<f64>,
        weights: Vec<f64>,
        kernel_covar: DynMat,
        proposal_cols: Option<Vec<f64>>,
        timestamp: Timestamp,
    ) -> Result<Self, Error> {
        let n = weights.len();
        if n == 0 || n > AKKF_MAX_PARTICLES || state_cols.len() != 4 * n || kernel_covar.n != n {
            return Err(Error::InvalidDimension);
        }
        if let Some(ref prop) = proposal_cols {
            if prop.len() != 4 * n {
                return Err(Error::InvalidDimension);
            }
        }
        Ok(Self {
            n_particles: n,
            state_cols,
            weights,
            kernel_covar,
            proposal_cols,
            timestamp,
        })
    }

    pub fn weighted_mean(&self) -> Result<[f64; 4], Error> {
        let n = self.n_particles;
        let mut mean = [0.0; 4];
        for d in 0..4 {
            let mut s = 0.0;
            for i in 0..n {
                s += self.state_cols[d * n + i] * self.weights[i];
            }
            require_finite(s)?;
            mean[d] = s;
        }
        Ok(mean)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveKernelKalmanPredictor {
    pub transition: ConstantVelocity2d,
    pub kernel: QuadraticKernel,
    pub lambda_predictor: f64,
}

impl AdaptiveKernelKalmanPredictor {
    pub fn tutorial_akkf() -> Result<Self, Error> {
        Ok(Self {
            transition: ConstantVelocity2d::new(1e-4, 1e-4)?,
            kernel: QuadraticKernel::tutorial_akkf(),
            lambda_predictor: 1e-3,
        })
    }

    pub fn predict(
        &self,
        prior: &KernelParticleState,
        timestamp: Timestamp,
        proposal_cols: &[f64],
        new_state_cols: &[f64],
    ) -> Result<KernelParticleState, Error> {
        let n = prior.n_particles;
        if proposal_cols.len() != 4 * n || new_state_cols.len() != 4 * n {
            return Err(Error::InvalidDimension);
        }
        let k_tilde_tilde = quadratic_gram(&self.kernel, 4, proposal_cols, proposal_cols, n)?;
        let k_tilde_nontilde =
            quadratic_gram(&self.kernel, 4, proposal_cols, &prior.state_cols, n)?;

        let mut reg = DynMat::identity(n)?;
        reg = reg.scale(self.lambda_predictor)?;
        let inv_val = k_tilde_tilde.add(&reg)?.pinv()?;
        let kernel_t = inv_val.mul(&k_tilde_nontilde)?;

        let prediction_weights = mat_vec(&kernel_t, &prior.weights)?;

        let kt_s = kernel_t.mul(&prior.kernel_covar)?;
        let kt_t = kernel_t.transpose()?;
        let mut prediction_covariance = kt_s.mul(&kt_t)?;

        let mut new_val = inv_val.mul(&k_tilde_tilde)?;
        let i = DynMat::identity(n)?;
        new_val = new_val.sub(&i)?;
        let v = new_val.mul(&new_val.transpose()?)?;
        let scale = 1.0 / (n as f64);
        prediction_covariance = prediction_covariance.add(&v.scale(scale)?)?;

        Ok(KernelParticleState {
            n_particles: n,
            state_cols: new_state_cols.to_vec(),
            weights: prediction_weights,
            kernel_covar: prediction_covariance,
            proposal_cols: None,
            timestamp,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveKernelKalmanUpdater {
    pub measurement: Cartesian2DToBearing,
    pub kernel: QuadraticKernel,
    pub lambda_updater: f64,
}

impl AdaptiveKernelKalmanUpdater {
    pub fn tutorial_akkf() -> Result<Self, Error> {
        Ok(Self {
            measurement: Cartesian2DToBearing::tutorial_akkf()?,
            kernel: QuadraticKernel::tutorial_akkf(),
            lambda_updater: 1e-3,
        })
    }

    pub fn measurement_particles(&self, state_cols: &[f64], n: usize) -> Result<Vec<f64>, Error> {
        if state_cols.len() != 4 * n {
            return Err(Error::InvalidDimension);
        }
        let mut out = vec![0.0; n];
        for i in 0..n {
            let state = [
                state_cols[0 * n + i],
                state_cols[1 * n + i],
                state_cols[2 * n + i],
                state_cols[3 * n + i],
            ];
            out[i] = self.measurement.h_state(state)?;
        }
        Ok(out)
    }

    /// Posterior kernel weights/covariance after a measurement (before proposal resampling).
    pub fn measure_update(
        &self,
        predicted: &KernelParticleState,
        detection: &Detection,
    ) -> Result<(Vec<f64>, DynMat), Error> {
        let n = predicted.n_particles;
        let meas_particles = self.measurement_particles(&predicted.state_cols, n)?;
        let g_yy = self.kernel.gram_columns(1, &meas_particles, n)?;
        let z = detection.vector.0[0];
        let g_y = self
            .kernel
            .vector_columns(1, &meas_particles, n, core::slice::from_ref(&z))?;

        let gyy_s = g_yy.mul(&predicted.kernel_covar)?;
        let mut reg = DynMat::identity(n)?;
        reg = reg.scale(self.lambda_updater)?;
        let q_akkf = predicted.kernel_covar.mul(&gyy_s.add(&reg)?.pinv()?)?;

        let gyy_w = mat_vec(&g_yy, &predicted.weights)?;
        let mut delta = vec![0.0; n];
        for i in 0..n {
            delta[i] = g_y[i] - gyy_w[i];
        }
        let q_delta = mat_vec(&q_akkf, &delta)?;
        let mut updated_weights = vec![0.0; n];
        for i in 0..n {
            updated_weights[i] = predicted.weights[i] + q_delta[i];
        }

        let q_gyy = q_akkf.mul(&g_yy)?;
        let updated_covariance = predicted
            .kernel_covar
            .sub(&q_gyy.mul(&predicted.kernel_covar)?)?;

        Ok((updated_weights, updated_covariance))
    }

    pub fn update(
        &self,
        predicted: &KernelParticleState,
        detection: &Detection,
        proposal_cols: &[f64],
    ) -> Result<KernelParticleState, Error> {
        let n = predicted.n_particles;
        if proposal_cols.len() != 4 * n {
            return Err(Error::InvalidDimension);
        }
        let (updated_weights, updated_covariance) = self.measure_update(predicted, detection)?;
        Ok(KernelParticleState {
            n_particles: n,
            state_cols: predicted.state_cols.clone(),
            weights: updated_weights,
            kernel_covar: updated_covariance,
            proposal_cols: Some(proposal_cols.to_vec()),
            timestamp: detection.timestamp,
        })
    }

    /// Mean and diagonal covariance for proposal `multivariate_normal` (Stone Soup AKKF updater).
    pub fn proposal_distribution(
        predicted: &KernelParticleState,
        weights: &[f64],
        kernel_covar: &DynMat,
    ) -> Result<([f64; 4], [[f64; 4]; 4]), Error> {
        let n = predicted.n_particles;
        if weights.len() != n || kernel_covar.n != n {
            return Err(Error::InvalidDimension);
        }
        let mut mean = [0.0; 4];
        for d in 0..4 {
            let mut s = 0.0;
            for i in 0..n {
                s += predicted.state_cols[d * n + i] * weights[i];
            }
            require_finite(s)?;
            mean[d] = s;
        }
        let mut m = [[0.0; 4]; 4];
        for a in 0..4 {
            for b in 0..4 {
                let mut s = 0.0;
                for i in 0..n {
                    let mut t = 0.0;
                    for j in 0..n {
                        t += predicted.state_cols[a * n + i]
                            * kernel_covar.get(i, j)?
                            * predicted.state_cols[b * n + j];
                    }
                    s += t;
                }
                require_finite(s)?;
                m[a][b] = s;
            }
        }
        let mut cov = [[0.0; 4]; 4];
        for i in 0..4 {
            cov[i][i] = m[i][i];
        }
        Ok((mean, cov))
    }
}

/// Propagate proposal particles with CV model and per-particle process noise columns.
pub fn transition_particles(
    transition: &ConstantVelocity2d,
    dt: Duration,
    proposal_cols: &[f64],
    n: usize,
    noise_cols: &[[f64; 4]],
) -> Result<Vec<f64>, Error> {
    if proposal_cols.len() != 4 * n || noise_cols.len() != n {
        return Err(Error::InvalidDimension);
    }
    let f = transition.transition_matrix(dt)?;
    let mut out = vec![0.0; 4 * n];
    for i in 0..n {
        let x = [
            proposal_cols[0 * n + i],
            proposal_cols[1 * n + i],
            proposal_cols[2 * n + i],
            proposal_cols[3 * n + i],
        ];
        let mut y = [0.0; 4];
        for r in 0..4 {
            let mut s = 0.0;
            for c in 0..4 {
                s += f.get(r, c)? * x[c];
            }
            y[r] = s + noise_cols[i][r];
        }
        for d in 0..4 {
            out[d * n + i] = y[d];
        }
    }
    Ok(out)
}
