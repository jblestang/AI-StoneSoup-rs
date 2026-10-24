use stonesoup_core::{
    ConstantVelocity2d, Detection, Error, GaussianState, KalmanPredictor, KalmanUpdater,
    KalmanWorkspace, LinearGaussianMeasurement, Mat, SensorId, Timestamp,
};

fn tutorial_prior() -> GaussianState {
    let mut cov = Mat::<4, 4>::zeros();
    cov.set(0, 0, 1.5).expect("cov");
    cov.set(1, 1, 0.5).expect("cov");
    cov.set(2, 2, 1.5).expect("cov");
    cov.set(3, 3, 0.5).expect("cov");
    GaussianState::new_from_array([0.0, 1.0, 0.0, 1.0], cov, Timestamp::ZERO).expect("prior")
}

#[test]
fn non_finite_mean_returns_error() {
    let cov = Mat::<4, 4>::zeros();
    let err =
        GaussianState::new_from_array([f64::NAN, 1.0, 0.0, 1.0], cov, Timestamp::ZERO).unwrap_err();
    assert_eq!(err, Error::NonFiniteInput);
}

#[test]
fn zero_timestep_predict_is_identity_mean() {
    let transition = ConstantVelocity2d::new(0.05, 0.05).unwrap();
    let predictor = KalmanPredictor::new(transition);
    let prior = tutorial_prior();
    let mut ws = KalmanWorkspace {
        f: Mat::zeros(),
        q: Mat::zeros(),
        h: Mat::zeros(),
        r: Mat::zeros(),
    };
    let pred = predictor.predict(&prior, Timestamp::ZERO, &mut ws).unwrap();
    assert_eq!(pred.mean.get(0).unwrap(), prior.mean.get(0).unwrap());
}

#[test]
fn singular_innovation_is_error() {
    let r = Mat::<2, 2>::zeros();
    let measurement = LinearGaussianMeasurement::position_2d(r).unwrap();
    let updater = KalmanUpdater::new(measurement);
    let cov = Mat::<4, 4>::zeros();
    let predicted =
        GaussianState::new_from_array([0.0, 1.0, 0.0, 1.0], cov, Timestamp::ZERO).unwrap();
    let mut ws = KalmanWorkspace {
        f: Mat::zeros(),
        q: Mat::zeros(),
        h: Mat::zeros(),
        r: Mat::zeros(),
    };
    let det = Detection::new([0.0, 0.0], Timestamp::ZERO, SensorId(0)).unwrap();
    let err = updater.update(&predicted, &det, &mut ws).unwrap_err();
    assert_eq!(err, Error::SingularInnovation);
}
