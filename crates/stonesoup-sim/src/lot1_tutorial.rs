//! Tutorial 01 measurement generator — NumPy seed 1991, Stone Soup draw order.

use stonesoup_core::{
    ConstantVelocity2d, Duration, Error, GroundTruthState, LinearGaussianMeasurement, Mat,
    Timestamp,
};

use crate::mvn::{add_vec4, mat4_vec4, measure_position, mvn2_lot1_r, mvn4_lot1_q};
use crate::numpy_random::NumpyRng;

pub const LOT1_NUMPY_SEED: u32 = 1991;
pub const LOT1_NUM_STEPS: usize = 21;

pub struct Lot1TutorialData {
    pub truth: [[f64; 4]; LOT1_NUM_STEPS],
    pub measurements: [[f64; 2]; LOT1_NUM_STEPS],
}

/// Regenerate truth path and noisy position measurements (same RNG stream as `export_lot1_fixtures.py`).
pub fn generate_lot1_tutorial(seed: u32) -> Result<Lot1TutorialData, Error> {
    let mut rng = NumpyRng::seed(seed);
    let transition = ConstantVelocity2d::new(0.05, 0.05)?;
    let dt = Duration::SEC;
    let f = transition.transition_matrix(dt)?;
    let q = transition.process_noise(dt)?;

    let mut r = Mat::<2, 2>::zeros();
    r.set(0, 0, 5.0)?;
    r.set(1, 1, 5.0)?;
    let measurement = LinearGaussianMeasurement::position_2d(r)?;
    let h = measurement.matrix_h()?;

    let mut truth = [[0.0; 4]; LOT1_NUM_STEPS];
    truth[0] = [0.0, 1.0, 0.0, 1.0];

    for k in 1..LOT1_NUM_STEPS {
        let fx = mat4_vec4(f, truth[k - 1])?;
        let _ = q;
        let noise = mvn4_lot1_q(&mut rng)?;
        truth[k] = add_vec4(fx, noise)?;
    }

    let mut measurements = [[0.0; 2]; LOT1_NUM_STEPS];
    for (i, state) in truth.iter().enumerate() {
        let z = measure_position(h, *state)?;
        let _ = measurement.r();
        let noise = mvn2_lot1_r(&mut rng)?;
        measurements[i] = [z[0] + noise[0], z[1] + noise[1]];
    }

    Ok(Lot1TutorialData {
        truth,
        measurements,
    })
}

pub fn generate_lot1_measurements(seed: u32) -> Result<[[f64; 2]; LOT1_NUM_STEPS], Error> {
    Ok(generate_lot1_tutorial(seed)?.measurements)
}

#[allow(dead_code)]
pub fn ground_truth_state(data: &Lot1TutorialData, i: usize) -> Result<GroundTruthState, Error> {
    let ts = Timestamp::from_nanos((i as i64) * Duration::SEC.0);
    GroundTruthState::new(data.truth[i], ts)
}
