//! Particle resamplers (Stone Soup `stonesoup.resampler.particle`).

use crate::numpy_random::NumpyRng;
use stonesoup_core::Error;

/// Upper bound on particles for in-place systematic resampling (Tutorial 04 uses 1000).
pub const MAX_RESAMPLE_PARTICLES: usize = 1024;

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

/// Systematic resampler (Stone Soup `SystematicResampler`).
///
/// Resamples `particles` and `log_weights` in place to length `nparts` with uniform log-weights.
pub fn systematic_resample<const D: usize>(
    particles: &mut [[f64; D]],
    log_weights: &mut [f64],
    nparts: usize,
    rng: &mut NumpyRng,
) -> Result<(), Error> {
    let n = log_weights.len();
    if nparts == 0
        || n == 0
        || nparts > MAX_RESAMPLE_PARTICLES
        || n > MAX_RESAMPLE_PARTICLES
        || particles.len() < n
        || log_weights.len() < n
    {
        return Err(Error::CapacityExceeded);
    }
    let mut order = [0usize; MAX_RESAMPLE_PARTICLES];
    sort_indices_by_log_weight(log_weights, n, &mut order);
    let max_log = log_weights[order[n - 1]];
    let mut cdf = [0.0f64; MAX_RESAMPLE_PARTICLES];
    let mut run = 0.0;
    for i in 0..n {
        let idx = order[i];
        run += libm::exp(log_weights[idx] - max_log);
        cdf[i] = libm::log(run) + max_log;
    }
    let u_i = rng.rk_double() / (nparts as f64);
    let log_uniform = libm::log(1.0 / (nparts as f64));
    let mut new_particles = [[0.0; D]; MAX_RESAMPLE_PARTICLES];
    for j in 0..nparts {
        let u_lin = u_i + (j as f64) / (nparts as f64);
        let u_j = libm::log(u_lin);
        let mut pick = order[n - 1];
        for i in 0..n {
            if cdf[i] >= u_j {
                pick = order[i];
                break;
            }
        }
        new_particles[j] = particles[pick];
        log_weights[j] = log_uniform;
    }
    particles[..nparts].copy_from_slice(&new_particles[..nparts]);
    Ok(())
}
