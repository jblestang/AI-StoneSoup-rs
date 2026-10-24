//! Particle resamplers (Stone Soup `stonesoup.resampler.particle`).

use crate::numpy_random::NumpyRng;
use stonesoup_core::Error;

/// Upper bound on particles for in-place systematic resampling (Tutorial 04 uses 1000).
pub const MAX_RESAMPLE_PARTICLES: usize = 1024;

/// Stage-2 method for [`residual_resample`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualStage2Method {
    Multinomial,
    Systematic,
    Stratified,
}

/// Effective sample size from log-weights (Kish ESS).
pub fn effective_sample_size(log_weights: &[f64]) -> f64 {
    let mut max_log = f64::NEG_INFINITY;
    for &lw in log_weights {
        if lw > max_log {
            max_log = lw;
        }
    }
    let mut sum_w = 0.0;
    let mut sum_w2 = 0.0;
    for &lw in log_weights {
        let w = libm::exp(lw - max_log);
        sum_w += w;
        sum_w2 += w * w;
    }
    if sum_w2 <= 0.0 {
        return 0.0;
    }
    (sum_w * sum_w) / sum_w2
}

fn sort_indices_by_log_weight(
    log_weights: &[f64],
    n: usize,
    order: &mut [usize; MAX_RESAMPLE_PARTICLES],
) {
    for i in 0..n {
        order[i] = i;
    }
    for i in 1..n {
        let key = order[i];
        let lw = log_weights[key];
        let mut j = i;
        while j > 0 && log_weights[order[j - 1]] > lw {
            order[j] = order[j - 1];
            j -= 1;
        }
        order[j] = key;
    }
}

fn build_log_cdf(
    log_weights: &[f64],
    n: usize,
    order: &mut [usize; MAX_RESAMPLE_PARTICLES],
    cdf: &mut [f64; MAX_RESAMPLE_PARTICLES],
) {
    sort_indices_by_log_weight(log_weights, n, order);
    let max_log = log_weights[order[n - 1]];
    let mut run = 0.0;
    for i in 0..n {
        let idx = order[i];
        run += libm::exp(log_weights[idx] - max_log);
        cdf[i] = libm::log(run) + max_log;
    }
}

fn cdf_pick(order: &[usize], cdf: &[f64], n: usize, log_u: f64) -> usize {
    let mut pick = order[n - 1];
    for i in 0..n {
        if cdf[i] >= log_u {
            pick = order[i];
            break;
        }
    }
    pick
}

fn write_resampled<const D: usize>(
    particles: &[[f64; D]],
    picks: &[usize; MAX_RESAMPLE_PARTICLES],
    nparts: usize,
    out_particles: &mut [[f64; D]],
    out_log_w: &mut [f64],
) {
    let log_uniform = libm::log(1.0 / (nparts as f64));
    for j in 0..nparts {
        out_particles[j] = particles[picks[j]];
        out_log_w[j] = log_uniform;
    }
}

fn resample_from_log_u<const D: usize>(
    particles: &[[f64; D]],
    log_weights: &[f64],
    n: usize,
    log_u: &[f64; MAX_RESAMPLE_PARTICLES],
    nparts: usize,
    out_particles: &mut [[f64; D]],
    out_log_w: &mut [f64],
) -> Result<(), Error> {
    if nparts == 0 || n == 0 || nparts > MAX_RESAMPLE_PARTICLES || n > MAX_RESAMPLE_PARTICLES {
        return Err(Error::CapacityExceeded);
    }
    let mut order = [0usize; MAX_RESAMPLE_PARTICLES];
    let mut cdf = [0.0f64; MAX_RESAMPLE_PARTICLES];
    build_log_cdf(&log_weights[..n], n, &mut order, &mut cdf);
    let mut picks = [0usize; MAX_RESAMPLE_PARTICLES];
    for j in 0..nparts {
        picks[j] = cdf_pick(&order, &cdf, n, log_u[j]);
    }
    write_resampled(particles, &picks, nparts, out_particles, out_log_w);
    Ok(())
}

/// Systematic resampler (Stone Soup `SystematicResampler`).
pub fn systematic_resample<const D: usize>(
    particles: &mut [[f64; D]],
    log_weights: &mut [f64],
    n: usize,
    nparts: usize,
    rng: &mut NumpyRng,
) -> Result<(), Error> {
    if nparts == 0
        || n == 0
        || nparts > MAX_RESAMPLE_PARTICLES
        || n > MAX_RESAMPLE_PARTICLES
        || particles.len() < n
        || log_weights.len() < n
    {
        return Err(Error::CapacityExceeded);
    }
    let u_i = rng.uniform(0.0, 1.0 / (nparts as f64));
    let mut log_u = [0.0f64; MAX_RESAMPLE_PARTICLES];
    for j in 0..nparts {
        let u_lin = u_i + (j as f64) / (nparts as f64);
        log_u[j] = libm::log(u_lin);
    }
    let mut new_particles = [[0.0; D]; MAX_RESAMPLE_PARTICLES];
    let mut new_log = [0.0f64; MAX_RESAMPLE_PARTICLES];
    resample_from_log_u(
        particles,
        log_weights,
        n,
        &log_u,
        nparts,
        &mut new_particles,
        &mut new_log,
    )?;
    particles[..nparts].copy_from_slice(&new_particles[..nparts]);
    log_weights[..nparts].copy_from_slice(&new_log[..nparts]);
    Ok(())
}

/// Multinomial resampler (Stone Soup `MultinomialResampler`).
pub fn multinomial_resample<const D: usize>(
    particles: &mut [[f64; D]],
    log_weights: &mut [f64],
    n: usize,
    nparts: usize,
    rng: &mut NumpyRng,
) -> Result<(), Error> {
    if particles.len() < n || log_weights.len() < n {
        return Err(Error::CapacityExceeded);
    }
    let mut log_u = [0.0f64; MAX_RESAMPLE_PARTICLES];
    for j in 0..nparts {
        let u = rng.rk_double();
        log_u[j] = if u > 0.0 {
            libm::log(u)
        } else {
            f64::NEG_INFINITY
        };
    }
    let mut new_particles = [[0.0; D]; MAX_RESAMPLE_PARTICLES];
    let mut new_log = [0.0f64; MAX_RESAMPLE_PARTICLES];
    resample_from_log_u(
        particles,
        log_weights,
        n,
        &log_u,
        nparts,
        &mut new_particles,
        &mut new_log,
    )?;
    particles[..nparts].copy_from_slice(&new_particles[..nparts]);
    log_weights[..nparts].copy_from_slice(&new_log[..nparts]);
    Ok(())
}

/// Stratified resampler (Stone Soup `StratifiedResampler`).
pub fn stratified_resample<const D: usize>(
    particles: &mut [[f64; D]],
    log_weights: &mut [f64],
    n: usize,
    nparts: usize,
    rng: &mut NumpyRng,
) -> Result<(), Error> {
    if particles.len() < n || log_weights.len() < n {
        return Err(Error::CapacityExceeded);
    }
    let mut log_u = [0.0f64; MAX_RESAMPLE_PARTICLES];
    let inv = 1.0 / (nparts as f64);
    for j in 0..nparts {
        let s_l = (j as f64) * inv;
        let u = rng.uniform(s_l, s_l + inv);
        log_u[j] = if u > 0.0 {
            libm::log(u)
        } else {
            f64::NEG_INFINITY
        };
    }
    let mut new_particles = [[0.0; D]; MAX_RESAMPLE_PARTICLES];
    let mut new_log = [0.0f64; MAX_RESAMPLE_PARTICLES];
    resample_from_log_u(
        particles,
        log_weights,
        n,
        &log_u,
        nparts,
        &mut new_particles,
        &mut new_log,
    )?;
    particles[..nparts].copy_from_slice(&new_particles[..nparts]);
    log_weights[..nparts].copy_from_slice(&new_log[..nparts]);
    Ok(())
}

/// Residual resampler (Stone Soup `ResidualResampler`); `nparts` must equal `n`.
pub fn residual_resample<const D: usize>(
    particles: &mut [[f64; D]],
    log_weights: &mut [f64],
    n: usize,
    method: ResidualStage2Method,
    rng: &mut NumpyRng,
) -> Result<(), Error> {
    let nparts = n;
    if nparts == 0 || nparts > MAX_RESAMPLE_PARTICLES || particles.len() < n {
        return Err(Error::CapacityExceeded);
    }
    let mut weights = [0.0f64; MAX_RESAMPLE_PARTICLES];
    for i in 0..n {
        weights[i] = libm::exp(log_weights[i]);
    }
    let mut floors = [0i32; MAX_RESAMPLE_PARTICLES];
    let mut n_stage_1 = 0usize;
    for i in 0..n {
        floors[i] = libm::floor(weights[i] * (nparts as f64)) as i32;
        if floors[i] > 0 {
            n_stage_1 += floors[i] as usize;
        }
    }
    let mut picks = [0usize; MAX_RESAMPLE_PARTICLES];
    let mut out_idx = 0usize;
    for i in 0..n {
        let f = floors[i] as usize;
        for _ in 0..f {
            if out_idx >= nparts {
                return Err(Error::CapacityExceeded);
            }
            picks[out_idx] = i;
            out_idx += 1;
        }
    }
    let n_stage_2 = nparts.saturating_sub(n_stage_1);
    if n_stage_2 > 0 {
        let mut r_weights = [0.0f64; MAX_RESAMPLE_PARTICLES];
        let mut r_sum = 0.0;
        for i in 0..n {
            r_weights[i] = weights[i] - (floors[i] as f64) / (nparts as f64);
            r_sum += r_weights[i];
        }
        if r_sum <= 0.0 {
            return Err(Error::NonFiniteInput);
        }
        let mut r_log = [0.0f64; MAX_RESAMPLE_PARTICLES];
        for i in 0..n {
            let w = r_weights[i] / r_sum;
            r_log[i] = if w > 0.0 {
                libm::log(w)
            } else {
                f64::NEG_INFINITY
            };
        }
        let mut log_u = [0.0f64; MAX_RESAMPLE_PARTICLES];
        match method {
            ResidualStage2Method::Multinomial => {
                for j in 0..n_stage_2 {
                    let u = rng.rk_double();
                    log_u[j] = if u > 0.0 {
                        libm::log(u)
                    } else {
                        f64::NEG_INFINITY
                    };
                }
            }
            ResidualStage2Method::Systematic => {
                let u_i = rng.uniform(0.0, 1.0 / (n_stage_2 as f64));
                for j in 0..n_stage_2 {
                    let u_lin = u_i + (j as f64) / (n_stage_2 as f64);
                    log_u[j] = libm::log(u_lin);
                }
            }
            ResidualStage2Method::Stratified => {
                let inv = 1.0 / (n_stage_2 as f64);
                for j in 0..n_stage_2 {
                    let s_l = (j as f64) * inv;
                    let u = rng.uniform(s_l, s_l + inv);
                    log_u[j] = if u > 0.0 {
                        libm::log(u)
                    } else {
                        f64::NEG_INFINITY
                    };
                }
            }
        }
        let mut order = [0usize; MAX_RESAMPLE_PARTICLES];
        let mut cdf = [0.0f64; MAX_RESAMPLE_PARTICLES];
        build_log_cdf(&r_log[..n], n, &mut order, &mut cdf);
        for j in 0..n_stage_2 {
            if out_idx >= nparts {
                return Err(Error::CapacityExceeded);
            }
            picks[out_idx] = cdf_pick(&order, &cdf, n, log_u[j]);
            out_idx += 1;
        }
    }
    let mut new_particles = [[0.0; D]; MAX_RESAMPLE_PARTICLES];
    write_resampled(particles, &picks, nparts, &mut new_particles, log_weights);
    particles[..nparts].copy_from_slice(&new_particles[..nparts]);
    Ok(())
}

/// ESS wrapper using systematic resampling when ESS is below `threshold`.
pub fn ess_resample_if_needed<const D: usize>(
    particles: &mut [[f64; D]],
    log_weights: &mut [f64],
    nparts: usize,
    threshold: f64,
    rng: &mut NumpyRng,
) -> Result<bool, Error> {
    ess_resample_if_needed_with(
        particles,
        log_weights,
        nparts,
        threshold,
        rng,
        EssInnerResampler::Systematic,
    )
}

/// Inner resampler choice for [`ess_resample_if_needed_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EssInnerResampler {
    Systematic,
    Multinomial,
    Stratified,
    ResidualSystematic,
}

/// ESS wrapper (Stone Soup `ESSResampler`) with selectable inner resampler.
pub fn ess_resample_if_needed_with<const D: usize>(
    particles: &mut [[f64; D]],
    log_weights: &mut [f64],
    nparts: usize,
    threshold: f64,
    rng: &mut NumpyRng,
    inner: EssInnerResampler,
) -> Result<bool, Error> {
    let ess = effective_sample_size(&log_weights[..nparts]);
    if ess < threshold {
        match inner {
            EssInnerResampler::Systematic => {
                systematic_resample(particles, log_weights, nparts, nparts, rng)?;
            }
            EssInnerResampler::Multinomial => {
                multinomial_resample(particles, log_weights, nparts, nparts, rng)?;
            }
            EssInnerResampler::Stratified => {
                stratified_resample(particles, log_weights, nparts, nparts, rng)?;
            }
            EssInnerResampler::ResidualSystematic => {
                residual_resample(
                    particles,
                    log_weights,
                    nparts,
                    ResidualStage2Method::Systematic,
                    rng,
                )?;
            }
        }
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Replay RNG draws from the Resampling tutorial preamble (seed must already be set).
pub fn replay_resampling_tutorial_preamble(rng: &mut NumpyRng, n_particles: usize) {
    for _ in 0..5 {
        let _ = rng.rk_double();
    }
    let _ = rng.uniform(0.0, 1.0 / (n_particles as f64));
    let inv = 1.0 / (n_particles as f64);
    for j in 0..n_particles {
        let s_l = (j as f64) * inv;
        let _ = rng.uniform(s_l, s_l + inv);
    }
    for _ in 0..2 {
        let _ = rng.rk_double();
    }
}
