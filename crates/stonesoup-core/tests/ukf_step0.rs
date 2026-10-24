use stonesoup_core::numeric::unscented::{gauss2sigma, UkfParams, SIGMA_COUNT};
use stonesoup_core::{
    CartesianToBearingRange, ConstantVelocity2d, Detection, GaussianState, KalmanWorkspace, Mat,
    SensorId, Timestamp, UnscentedKalmanPredictor, UnscentedKalmanUpdater,
};

#[test]
fn ukf_step0() {
    let tm = ConstantVelocity2d::new(0.05, 0.05).unwrap();
    let mut ws = KalmanWorkspace {
        f: Mat::zeros(),
        q: Mat::zeros(),
        h: Mat::zeros(),
        r: Mat::zeros(),
    };
    let mut cov = Mat::<4, 4>::zeros();
    cov.set(0, 0, 1.5).unwrap();
    cov.set(1, 1, 0.5).unwrap();
    cov.set(2, 2, 1.5).unwrap();
    cov.set(3, 3, 0.5).unwrap();
    let prior = GaussianState::new_from_array([0.0, 1.0, 0.0, 1.0], cov, Timestamp::ZERO).unwrap();
    let (sp, w) = gauss2sigma(&prior, UkfParams::default()).unwrap();
    eprintln!("sp1 {:?}", sp[1].0);
    let mm = CartesianToBearingRange::tutorial02().unwrap();
    let mut zm = [0.0; 2];
    for j in 0..SIGMA_COUNT {
        let z = mm.h(sp[j].0).unwrap();
        zm[0] += w.mean[j] * z[0];
        zm[1] += w.mean[j] * z[1];
    }
    eprintln!("zmean {:?}", zm);
    let pred = UnscentedKalmanPredictor::new(tm)
        .predict(&prior, Timestamp::ZERO, &mut ws)
        .unwrap();
    let mm = CartesianToBearingRange::tutorial02().unwrap();
    let upd = UnscentedKalmanUpdater::new(mm);
    let m = [-3.112_230_520_043_526, 47.787_622_539_853_3];
    let det = Detection::new(m, Timestamp::ZERO, SensorId(0)).unwrap();
    let post = upd.update(&pred, &det, &mut ws).unwrap();
    eprintln!("post {:?}", post.mean.0);
    assert!((post.mean.0[0] - 1.336_124_9).abs() < 1e-5);
    assert!((post.mean.0[2] - (-0.107_206_88)).abs() < 1e-5);
}
