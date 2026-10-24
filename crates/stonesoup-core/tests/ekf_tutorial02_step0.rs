use stonesoup_core::{
    CartesianToBearingRange, ConstantVelocity2d, Detection, Duration, ExtendedKalmanUpdater,
    GaussianState, KalmanPredictor, KalmanWorkspace, Mat, SensorId, Timestamp,
};

#[test]
fn ekf_step0_matches_stonesoup_matrices() {
    let transition = ConstantVelocity2d::new(0.05, 0.05).unwrap();
    let predictor = KalmanPredictor::new(transition);
    let measurement = CartesianToBearingRange::tutorial02().unwrap();
    let updater = ExtendedKalmanUpdater::new(measurement);
    let mut cov = Mat::<4, 4>::zeros();
    cov.set(0, 0, 1.5).unwrap();
    cov.set(1, 1, 0.5).unwrap();
    cov.set(2, 2, 1.5).unwrap();
    cov.set(3, 3, 0.5).unwrap();
    let prior = GaussianState::new_from_array([0.0, 1.0, 0.0, 1.0], cov, Timestamp::ZERO).unwrap();
    let ts = Timestamp::ZERO;
    let mut ws = KalmanWorkspace {
        f: Mat::zeros(),
        q: Mat::zeros(),
        h: Mat::zeros(),
        r: Mat::zeros(),
    };
    let pred = predictor.predict(&prior, ts, &mut ws).unwrap();
    let h = measurement.jacobian(&pred).unwrap();
    let z_pred = measurement.predict_measurement(&pred).unwrap();
    eprintln!("H bearing row: {:?} {:?}", h.get(0, 2), h.get(0, 0));
    eprintln!("H range row: {:?} {:?}", h.get(1, 0), h.get(1, 2));
    eprintln!("z_pred {:?}", z_pred.0);
    let m = [-3.112_230_520_043_526, 47.787_622_539_853_3];
    let det = Detection::new(m, ts, SensorId(0)).unwrap();
    let post = updater.update(&pred, &det, &mut ws).unwrap();
    eprintln!("post {:?}", post.mean.0);
    assert!((post.mean.0[0] - 1.327_426_454_121_663_6).abs() < 1e-7);
    assert!((post.mean.0[2] - (-0.107_220_617_725_864_29)).abs() < 1e-7);
}
