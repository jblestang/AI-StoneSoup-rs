//! Native linked harness exercising strict APIs under `#[no_panic]`.
//!
//! Manifest (Lot 1): `ConstantVelocity2d`, `LinearGaussianMeasurement`, `GaussianState`,
//! `RmseAccumulator`, `tutorial_cov`. Kalman predict/update and full scenario stepping are
//! tracked in `docs/NO_PANIC.md` (Mat::mul still flagged by the linker plugin).

use no_panic::no_panic;
use stonesoup_core::{
    ConstantVelocity2d, Duration, GaussianState, LinearGaussianMeasurement, Mat, RmseAccumulator,
    Timestamp,
};
use stonesoup_scenarios::tutorial_cov;

#[no_panic]
fn exercise_core_lot1() -> u32 {
    let transition = match ConstantVelocity2d::new(0.05, 0.05) {
        Ok(t) => t,
        Err(_) => return 1,
    };
    if transition.transition_matrix(Duration::SEC).is_err() {
        return 2;
    }
    if transition.process_noise(Duration::SEC).is_err() {
        return 3;
    }
    let mut r = Mat::<2, 2>::zeros();
    if r.set(0, 0, 5.0).is_err() || r.set(1, 1, 5.0).is_err() {
        return 4;
    }
    if LinearGaussianMeasurement::position_2d(r).is_err() {
        return 5;
    }
    let cov = match tutorial_cov() {
        Ok(c) => c,
        Err(_) => return 6,
    };
    if GaussianState::new_from_array([0.0, 1.0, 0.0, 1.0], cov, Timestamp::ZERO).is_err() {
        return 7;
    }
    let mut rmse = RmseAccumulator::new();
    if rmse.push(0.2, -0.1).is_err() {
        return 8;
    }
    if rmse.rmse().is_err() {
        return 9;
    }
    0
}

fn main() {
    std::process::exit(exercise_core_lot1() as i32);
}
