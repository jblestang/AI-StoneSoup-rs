//! Systematic resampler parity vs Stone Soup (`verification/fixtures/resampler_systematic.json`).

use stonesoup_sim::{effective_sample_size, systematic_resample, NumpyRng, LOT2_NUMPY_SEED};

#[test]
fn systematic_resampler_matches_stonesoup_fixture() {
    let raw = include_str!("../../../verification/fixtures/resampler_systematic.json");
    let v: serde_json::Value = serde_json::from_str(raw).expect("fixture json");
    let n = v["n_particles"].as_u64().expect("n") as usize;
    let d = v["state_dim"].as_u64().expect("d") as usize;
    assert_eq!(d, 4);

    let log_in: Vec<f64> = v["log_weights_in"]
        .as_array()
        .expect("log in")
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let parts_in: Vec<Vec<f64>> = v["particles_in"]
        .as_array()
        .expect("parts in")
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect();

    let ess_expected = v["ess_before"].as_f64().expect("ess before");
    let ess = effective_sample_size(&log_in);
    assert!(
        (ess - ess_expected).abs() < 1e-9,
        "ess {ess} vs {ess_expected}"
    );

    let mut particles = [[0.0f64; 4]; 1024];
    let mut log_w = [0.0f64; 1024];
    for i in 0..n {
        for j in 0..4 {
            particles[i][j] = parts_in[i][j];
        }
        log_w[i] = log_in[i];
    }
    let mut rng = NumpyRng::seed(LOT2_NUMPY_SEED);
    systematic_resample(&mut particles, &mut log_w, n, n, &mut rng).unwrap();

    let log_out: Vec<f64> = v["log_weights_out"]
        .as_array()
        .expect("log out")
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let parts_out: Vec<Vec<f64>> = v["particles_out"]
        .as_array()
        .expect("parts out")
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect();

    for i in 0..n {
        assert!((log_w[i] - log_out[i]).abs() < 1e-12, "log_w[{i}]");
        for j in 0..4 {
            assert!(
                (particles[i][j] - parts_out[i][j]).abs() < 1e-12,
                "particle[{i}][{j}]"
            );
        }
    }

    let ess_after_expected = v["ess_after"].as_f64().expect("ess after");
    let ess_after = effective_sample_size(&log_w[..n]);
    assert!(
        (ess_after - ess_after_expected).abs() < 1e-6,
        "ess after {ess_after} vs {ess_after_expected}"
    );
}
