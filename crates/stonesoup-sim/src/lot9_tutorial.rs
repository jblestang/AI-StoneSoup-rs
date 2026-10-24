//! AKKF tutorial RNG preamble (Stone Soup `AKKF.py`, seed 50).

use stonesoup_core::{Cartesian2DToBearing, ConstantVelocity2d, Duration, Error};

use crate::mvn::{mat4_to_array, multivariate_normal4_batch};
use crate::numpy_random::NumpyRng;

pub const LOT9_NUMPY_SEED: u32 = 50;
pub const LOT9_NUM_PARTICLES: usize = 30;
pub const LOT9_NUM_STEPS: usize = 30;

pub const LOT9_PRIOR_MEAN: [f64; 4] = [-0.3, 0.001, 0.7, -0.055];

pub fn prior_covariance() -> Result<[[f64; 4]; 4], Error> {
    let mut c = [[0.0; 4]; 4];
    c[0][0] = 0.1 * 0.1;
    c[1][1] = 0.005 * 0.005;
    c[2][2] = 0.1 * 0.1;
    c[3][3] = 0.01 * 0.01;
    Ok(c)
}

pub fn process_covariance(dt: Duration) -> Result<[[f64; 4]; 4], Error> {
    let t = ConstantVelocity2d::new(1e-4, 1e-4)?;
    mat4_to_array(t.process_noise(dt)?)
}

/// Consumes the NumPy stream for truth transitions (29×) and bearing measurements (30×).
pub fn advance_lot9_preamble(rng: &mut NumpyRng) -> Result<(), Error> {
    let q = process_covariance(Duration::SEC)?;
    let zero = [0.0; 4];
    for _ in 0..29 {
        multivariate_normal4_batch(rng, zero, &q, 1)?;
    }
    let bearing = Cartesian2DToBearing::tutorial_akkf()?;
    let std = libm::sqrt(bearing.noise_var);
    for _ in 0..30 {
        let _ = rng.standard_normal()? * std;
    }
    Ok(())
}

/// Initialise filter RNG: seed 50 → preamble → reset seed 50 → prior sample (tutorial order).
pub fn initialise_lot9_filter_rng() -> Result<(NumpyRng, alloc::vec::Vec<[f64; 4]>), Error> {
    extern crate alloc;
    let mut rng = NumpyRng::seed(LOT9_NUMPY_SEED);
    advance_lot9_preamble(&mut rng)?;
    let mut rng = NumpyRng::seed(LOT9_NUMPY_SEED);
    let cov = prior_covariance()?;
    let particles =
        multivariate_normal4_batch(&mut rng, LOT9_PRIOR_MEAN, &cov, LOT9_NUM_PARTICLES)?;
    Ok((rng, particles))
}
