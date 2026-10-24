//! Tutorial 02–04 shared simulation — NumPy seed 1991, Stone Soup draw order.
#![allow(clippy::approx_constant)]

use stonesoup_core::{CartesianToBearingRange, ConstantVelocity2d, Duration, Error, Timestamp};

use crate::mvn::{add_vec4, mat4_vec4, mvn4_lot1_q, z_dot_a_col};
use crate::numpy_random::NumpyRng;

pub const LOT2_NUMPY_SEED: u32 = 1991;
pub const LOT2_NUM_STEPS: usize = 21;

/// NumPy SVD `A` for bearing–range measurement noise (5° bearing, 1 m range).
#[allow(clippy::excessive_precision)]
const LOT2_A_R: [[f64; 2]; 2] = [[0.08726646, 0.0], [0.0, 1.0]];

/// NumPy SVD `A` for particle-filter prior `diag(1.5, 0.5, 1.5, 0.5)`.
#[allow(clippy::excessive_precision)]
const LOT2_A_PRIOR: [[f64; 4]; 4] = [
    [1.224744871391589, 0.0, 0.0, 0.0],
    [0.0, 0.7071067811865476, 0.0, 0.0],
    [0.0, 0.0, 1.224744871391589, 0.0],
    [0.0, 0.0, 0.0, 0.7071067811865476],
];

fn mvn4_a(rng: &mut NumpyRng, a: [[f64; 4]; 4]) -> Result<[f64; 4], Error> {
    let z = [
        rng.standard_normal()?,
        rng.standard_normal()?,
        rng.standard_normal()?,
        rng.standard_normal()?,
    ];
    Ok([
        z_dot_a_col(z, a, 0)?,
        z_dot_a_col(z, a, 1)?,
        z_dot_a_col(z, a, 2)?,
        z_dot_a_col(z, a, 3)?,
    ])
}

fn mvn2_a(rng: &mut NumpyRng, a: [[f64; 2]; 2]) -> Result<[f64; 2], Error> {
    let z = [rng.standard_normal()?, rng.standard_normal()?];
    Ok([z_dot_a_col(z, a, 0)?, z_dot_a_col(z, a, 1)?])
}

/// Sample `N(0, Q)` for Tutorial 02/04 process noise (`dt = 1 s`).
pub fn mvn4_lot2_q(rng: &mut NumpyRng) -> Result<[f64; 4], Error> {
    mvn4_lot1_q(rng)
}

/// Sample `N(0, R)` for bearing–range sensor noise.
pub fn mvn2_lot2_r(rng: &mut NumpyRng) -> Result<[f64; 2], Error> {
    mvn2_a(rng, LOT2_A_R)
}

/// Sample one particle from the Tutorial 04 prior Gaussian.
pub fn sample_lot2_pf_prior_particle(
    rng: &mut NumpyRng,
    mean: [f64; 4],
) -> Result<[f64; 4], Error> {
    let delta = mvn4_a(rng, LOT2_A_PRIOR)?;
    add_vec4(mean, delta)
}

pub struct Lot2TutorialData {
    pub truth: [[f64; 4]; LOT2_NUM_STEPS],
    pub measurements: [[f64; 2]; LOT2_NUM_STEPS],
}

/// Regenerate truth and bearing–range measurements (same stream as Tutorial 04 preamble).
pub fn generate_lot2_tutorial(seed: u32) -> Result<Lot2TutorialData, Error> {
    let mut rng = NumpyRng::seed(seed);
    let transition = ConstantVelocity2d::new(0.05, 0.05)?;
    let dt = Duration::SEC;
    let f = transition.transition_matrix(dt)?;
    let measurement = CartesianToBearingRange::tutorial02()?;

    let mut truth = [[0.0; 4]; LOT2_NUM_STEPS];
    truth[0] = [0.0, 1.0, 0.0, 1.0];
    for k in 1..LOT2_NUM_STEPS {
        let fx = mat4_vec4(f, truth[k - 1])?;
        let noise = mvn4_lot2_q(&mut rng)?;
        truth[k] = add_vec4(fx, noise)?;
    }

    let mut measurements = [[0.0; 2]; LOT2_NUM_STEPS];
    for (i, state) in truth.iter().enumerate() {
        let z = measurement.h(*state)?;
        let noise = mvn2_lot2_r(&mut rng)?;
        measurements[i] = [z[0] + noise[0], z[1] + noise[1]];
    }

    Ok(Lot2TutorialData {
        truth,
        measurements,
    })
}

/// Advance the NumPy RNG stream through truth + measurement generation (Tutorial 04 preamble).
pub fn advance_lot2_pf_preamble(rng: &mut NumpyRng) -> Result<(), Error> {
    let transition = ConstantVelocity2d::new(0.05, 0.05)?;
    let dt = Duration::SEC;
    let f = transition.transition_matrix(dt)?;
    let measurement = CartesianToBearingRange::tutorial02()?;

    let mut truth = [0.0, 1.0, 0.0, 1.0];
    for step in 0..LOT2_NUM_STEPS {
        if step > 0 {
            let fx = mat4_vec4(f, truth)?;
            let noise = mvn4_lot2_q(rng)?;
            truth = add_vec4(fx, noise)?;
        }
        let _z = measurement.h(truth)?;
        let _noise = mvn2_lot2_r(rng)?;
    }

    let _ = Timestamp::ZERO;
    Ok(())
}
