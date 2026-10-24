#![cfg_attr(target_os = "none", no_std)]
#![cfg_attr(target_os = "none", no_main)]
// FFI symbols for embedded link smoke; core crates remain `#![forbid(unsafe_code)]`.

use stonesoup_core::{
    ConstantVelocity2d, GaussianState, KalmanPredictor, KalmanWorkspace, Mat, Timestamp,
};
use stonesoup_scenarios::Lot1KalmanScenario;

#[no_mangle]
pub extern "C" fn stonesoup_embedded_smoke() -> i32 {
    let transition = match ConstantVelocity2d::new(0.05, 0.05) {
        Ok(t) => t,
        Err(_) => return -1,
    };
    let predictor = KalmanPredictor::new(transition);
    let mut cov = Mat::<4, 4>::zeros();
    if cov.set(0, 0, 1.0).is_err() {
        return -4;
    }
    let prior = match GaussianState::new_from_array([0.0, 1.0, 0.0, 1.0], cov, Timestamp::ZERO) {
        Ok(p) => p,
        Err(_) => return -2,
    };
    let mut ws = KalmanWorkspace {
        f: Mat::zeros(),
        q: Mat::zeros(),
        h: Mat::zeros(),
        r: Mat::zeros(),
    };
    let _ = predictor.predict(&prior, Timestamp::ZERO, &mut ws);
    let mut sc = match Lot1KalmanScenario::tutorial_defaults() {
        Ok(s) => s,
        Err(_) => return -3,
    };
    let _ = sc.step();
    0
}

#[cfg(target_os = "none")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(target_os = "none")]
#[no_mangle]
pub extern "C" fn _start() -> ! {
    let _code = stonesoup_embedded_smoke();
    loop {}
}

#[cfg(not(target_os = "none"))]
fn main() {
    std::process::exit(stonesoup_embedded_smoke());
}
