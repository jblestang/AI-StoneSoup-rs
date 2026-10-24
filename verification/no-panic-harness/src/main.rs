//! Native linked harness — manifest in `docs/NO_PANIC.md`.

use no_panic::no_panic;
use stonesoup_core::{
    ConstantVelocity2d, Detection, Duration, GaussianState, KalmanPredictor, KalmanUpdater,
    KalmanWorkspace, LinearGaussianMeasurement, Mat, RmseAccumulator, SensorId, Timestamp,
};
use stonesoup_scenarios::Lot1KalmanScenario;

#[no_panic]
fn exercise_kalman_once() -> u32 {
    let transition = match ConstantVelocity2d::new(0.05, 0.05) {
        Ok(t) => t,
        Err(_) => return 1,
    };
    let predictor = KalmanPredictor::new(transition);
    let mut r = Mat::<2, 2>::zeros();
    if r.set(0, 0, 5.0).is_err() || r.set(1, 1, 5.0).is_err() {
        return 2;
    }
    let measurement = match LinearGaussianMeasurement::position_2d(r) {
        Ok(m) => m,
        Err(_) => return 3,
    };
    let updater = KalmanUpdater::new(measurement);
    let mut cov = Mat::<4, 4>::zeros();
    if cov.set(0, 0, 1.5).is_err()
        || cov.set(1, 1, 0.5).is_err()
        || cov.set(2, 2, 1.5).is_err()
        || cov.set(3, 3, 0.5).is_err()
    {
        return 4;
    }
    let prior = match GaussianState::new_from_array([0.0, 1.0, 0.0, 1.0], cov, Timestamp::ZERO) {
        Ok(p) => p,
        Err(_) => return 5,
    };
    let mut ws = KalmanWorkspace {
        f: Mat::zeros(),
        q: Mat::zeros(),
        h: Mat::zeros(),
        r: Mat::zeros(),
    };
    let ts = Timestamp::from_nanos(Duration::SEC.0);
    let predicted = match predictor.predict(&prior, ts, &mut ws) {
        Ok(p) => p,
        Err(_) => return 6,
    };
    let det = match Detection::new([1.0, 2.0], ts, SensorId(0)) {
        Ok(d) => d,
        Err(_) => return 7,
    };
    if updater.update(&predicted, &det, &mut ws).is_err() {
        return 8;
    }
    0
}

#[no_panic]
fn exercise_lot1_scenario() -> u32 {
    let mut sc = match Lot1KalmanScenario::tutorial_defaults() {
        Ok(s) => s,
        Err(_) => return 10,
    };
    if sc.run_to_end().is_err() {
        return 11;
    }
    let mut rmse = RmseAccumulator::new();
    if rmse.push(0.1, -0.2).is_err() {
        return 12;
    }
    if rmse.rmse().is_err() {
        return 13;
    }
    0
}

fn main() {
    let code = exercise_kalman_once() + exercise_lot1_scenario();
    std::process::exit(code.min(255) as i32);
}
