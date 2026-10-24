use stonesoup_core::{
    CartesianToBearingRange, ConstantVelocity2d, Detection, Duration, ExtendedKalmanUpdater,
    GaussianState, KalmanPredictor, KalmanWorkspace, Mat, SensorId, Timestamp,
};

#[test]
fn ekf_step1_from_step0_posterior() {
    let transition = ConstantVelocity2d::new(0.05, 0.05).unwrap();
    let predictor = KalmanPredictor::new(transition);
    let measurement = CartesianToBearingRange::tutorial02().unwrap();
    let updater = ExtendedKalmanUpdater::new(measurement);
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
    let prior0 = GaussianState::new_from_array([0.0, 1.0, 0.0, 1.0], cov, Timestamp::ZERO).unwrap();
    let m0 = [-3.112_230_520_043_526, 47.787_622_539_853_3];
    let det0 = Detection::new(m0, Timestamp::ZERO, SensorId(0)).unwrap();
    let pred0 = predictor
        .predict(&prior0, Timestamp::ZERO, &mut ws)
        .unwrap();
    let post0 = updater.update(&pred0, &det0, &mut ws).unwrap();
    let ts1 = Timestamp::from_nanos(Duration::SEC.0);
    let pred1 = predictor.predict(&post0, ts1, &mut ws).unwrap();
    eprintln!("pred1 {:?}", pred1.mean.0);
    let m1 = [-3.065_156_009_477_388_6, 48.500_342_356_725_41];
    let det1 = Detection::new(m1, ts1, SensorId(0)).unwrap();
    let z_pred = measurement.predict_measurement(&pred1).unwrap();
    let innov = stonesoup_core::CartesianToBearingRange::innovation(m1, z_pred.0).unwrap();
    eprintln!("z_pred {:?} innov {:?}", z_pred.0, innov);
    let h = measurement.jacobian(&pred1).unwrap();
    eprintln!(
        "H00 {} H02 {} H10 {} H12 {}",
        h.get(0, 0).unwrap(),
        h.get(0, 2).unwrap(),
        h.get(1, 0).unwrap(),
        h.get(1, 2).unwrap()
    );
    let post1 = updater.update(&pred1, &det1, &mut ws).unwrap();
    eprintln!("post1 {:?}", post1.mean.0);
    assert!((pred1.mean.0[0] - 2.327_426_45).abs() < 1e-5);
    assert!((pred1.mean.0[2] - 0.892_779_38).abs() < 1e-5);
    assert!((post1.mean.0[0] - 1.888_484_81).abs() < 1e-5);
    assert!((post1.mean.0[2] - 0.455_948_29).abs() < 1e-5);
}
