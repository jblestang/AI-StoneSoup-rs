//! IMM sub-model dynamics aligned with dual-radar truth motion (CV / CA / CT±).

use crate::error::Error;
use crate::linalg::require_finite;
use crate::time::Duration;

use super::imm_state6::{GaussianState6, IMM_STATE_DIM};

/// Peak centripetal acceleration used to cap turn rate at speed (matches truth CT).
pub const CT_MANEUVER_ACCEL_MPS2: f64 = 9.0 * 9.806_65;

/// Singer-style CV on 6D state: acceleration is driven noise with exponential decay.
#[derive(Debug, Clone, Copy)]
pub struct SingerCv6d {
    pub q_accel: f64,
    /// 1/s decay on ax, ay between steps (larger ⇒ faster return to straight flight).
    pub alpha: f64,
}

impl SingerCv6d {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn f_q(
        &self,
        dt: Duration,
    ) -> Result<([[f64; IMM_STATE_DIM]; IMM_STATE_DIM], [[f64; IMM_STATE_DIM]; IMM_STATE_DIM]), Error>
    {
        let t = dt.as_seconds();
        require_finite(t)?;
        if t < 0.0 {
            return Err(Error::InvalidTimeStep);
        }
        let t2 = t * t;
        let rho = libm::exp(-self.alpha * t);
        let mut f = [[0.0; IMM_STATE_DIM]; IMM_STATE_DIM];
        f[0][0] = 1.0;
        f[0][1] = t;
        f[0][2] = 0.5 * t2;
        f[1][1] = 1.0;
        f[1][2] = t;
        f[2][2] = rho;
        f[3][3] = 1.0;
        f[3][4] = t;
        f[3][5] = 0.5 * t2;
        f[4][4] = 1.0;
        f[4][5] = t;
        f[5][5] = rho;

        let mut q = [[0.0; IMM_STATE_DIM]; IMM_STATE_DIM];
        let qa = self.q_accel;
        q[0][0] = qa * t2 * t2 / 4.0;
        q[0][2] = qa * t2 / 2.0;
        q[2][0] = qa * t2 / 2.0;
        q[2][2] = qa;
        q[3][3] = qa * t2 * t2 / 4.0;
        q[3][5] = qa * t2 / 2.0;
        q[5][3] = qa * t2 / 2.0;
        q[5][5] = qa;
        Ok((f, q))
    }
}

/// Piecewise-constant acceleration (matches truth ``TruthMotionMode::Ca``).
#[derive(Debug, Clone, Copy)]
pub struct ConstantAcceleration6d {
    pub q_accel: f64,
}

impl ConstantAcceleration6d {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn f_q(
        &self,
        dt: Duration,
    ) -> Result<([[f64; IMM_STATE_DIM]; IMM_STATE_DIM], [[f64; IMM_STATE_DIM]; IMM_STATE_DIM]), Error>
    {
        let t = dt.as_seconds();
        require_finite(t)?;
        if t < 0.0 {
            return Err(Error::InvalidTimeStep);
        }
        let t2 = t * t;
        let mut f = [[0.0; IMM_STATE_DIM]; IMM_STATE_DIM];
        f[0][0] = 1.0;
        f[0][1] = t;
        f[0][2] = 0.5 * t2;
        f[1][1] = 1.0;
        f[1][2] = t;
        f[2][2] = 1.0;
        f[3][3] = 1.0;
        f[3][4] = t;
        f[3][5] = 0.5 * t2;
        f[4][4] = 1.0;
        f[4][5] = t;
        f[5][5] = 1.0;

        let mut q = [[0.0; IMM_STATE_DIM]; IMM_STATE_DIM];
        let qa = self.q_accel;
        q[2][2] = qa;
        q[5][5] = qa;
        Ok((f, q))
    }
}

/// Coordinated turn: fixed speed, heading += ωΔt (truth-aligned).
#[derive(Debug, Clone, Copy)]
pub struct CoordinatedTurn6d {
    pub max_turn_accel_mps2: f64,
    pub q_vel: f64,
}

impl CoordinatedTurn6d {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn omega_for_speed(&self, speed_mps: f64) -> f64 {
        let v = speed_mps.max(30.0);
        self.max_turn_accel_mps2 / v
    }

    /// CT sub-models do not use longitudinal acceleration states.
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn prepare_state(prior: &GaussianState6) -> GaussianState6 {
        let mut mean = prior.mean;
        mean[2] = 0.0;
        mean[5] = 0.0;
        GaussianState6 {
            mean,
            cov: prior.cov,
            timestamp: prior.timestamp,
        }
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn predict_with_omega(
        &self,
        prior: &GaussianState6,
        timestamp: crate::time::Timestamp,
        omega_rad_s: f64,
    ) -> Result<GaussianState6, Error> {
        if prior.timestamp == timestamp {
            return Ok(*prior);
        }
        let dt = timestamp.checked_sub(prior.timestamp)?;
        let t = dt.as_seconds();
        require_finite(t)?;
        if t < 0.0 {
            return Err(Error::InvalidTimeStep);
        }
        let prior = Self::prepare_state(prior);
        let vx = prior.mean[1];
        let vy = prior.mean[4];
        let speed = libm::sqrt(vx * vx + vy * vy).max(1.0);
        let heading = libm::atan2(vy, vx);
        let w = omega_rad_s.clamp(-self.omega_for_speed(speed), self.omega_for_speed(speed));
        let heading_new = heading + w * t;
        let vx2 = speed * libm::cos(heading_new);
        let vy2 = speed * libm::sin(heading_new);
        let x2 = prior.mean[0] + vx2 * t;
        let y2 = prior.mean[3] + vy2 * t;

        let mut mean = prior.mean;
        mean[0] = x2;
        mean[1] = vx2;
        mean[2] = 0.0;
        mean[3] = y2;
        mean[4] = vy2;
        mean[5] = 0.0;

        let f = jacobian_ct(&prior.mean, t, w, speed)?;
        let fp = super::imm_state6::mat6_mul(&f, &prior.cov);
        let ft = super::imm_state6::mat6_transpose(&f);
        let fpft = super::imm_state6::mat6_mul(&fp, &ft);
        let mut q = [[0.0; IMM_STATE_DIM]; IMM_STATE_DIM];
        let qv = self.q_vel;
        q[1][1] = qv * t;
        q[4][4] = qv * t;
        q[0][0] = qv * t * t;
        q[3][3] = qv * t * t;
        let cov = super::imm_state6::mat6_add(&fpft, &q);
        GaussianState6::new(mean, cov, timestamp)
    }
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn jacobian_ct(
    s: &[f64; IMM_STATE_DIM],
    dt: f64,
    w: f64,
    speed: f64,
) -> Result<[[f64; IMM_STATE_DIM]; IMM_STATE_DIM], Error> {
    let eps = 1e-4;
    let mut f = [[0.0; IMM_STATE_DIM]; IMM_STATE_DIM];
    for i in 0..IMM_STATE_DIM {
        f[i][i] = 1.0;
    }
    for col in 0..IMM_STATE_DIM {
        let mut sp = *s;
        let mut sm = *s;
        sp[col] += eps;
        sm[col] -= eps;
        let fp = ct_mean(&sp, dt, w, speed);
        let fm = ct_mean(&sm, dt, w, speed);
        for row in 0..IMM_STATE_DIM {
            f[row][col] = (fp[row] - fm[row]) / (2.0 * eps);
        }
    }
    Ok(f)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn ct_mean(s: &[f64; IMM_STATE_DIM], dt: f64, w: f64, speed: f64) -> [f64; IMM_STATE_DIM] {
    let vx = s[1];
    let vy = s[4];
    let heading = libm::atan2(vy, vx);
    let heading_new = heading + w * dt;
    let vx2 = speed * libm::cos(heading_new);
    let vy2 = speed * libm::sin(heading_new);
    let mut out = *s;
    out[0] = s[0] + vx2 * dt;
    out[1] = vx2;
    out[3] = s[3] + vy2 * dt;
    out[4] = vy2;
    out
}
