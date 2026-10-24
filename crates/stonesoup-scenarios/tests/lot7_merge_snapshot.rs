//! Run GM-PHD reduce on Stone Soup step-1 pre-reduce snapshot from the export JSON.

use std::fs;
use std::path::PathBuf;

use stonesoup_core::{
    gm_phd_merge, gm_phd_prune, squared_mahalanobis_state, GaussianState, GmPhdConfig, Mat,
    PhdComponent, PhdTag, Timestamp, MAX_GMPHD_UPDATED,
};

fn load_step1_snapshot() -> (Vec<PhdComponent>, GmPhdConfig) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("verification/fixtures/lot7_gm_phd.json");
    let text = fs::read_to_string(path).expect("read fixture json");
    let payload: serde_json::Value = serde_json::from_str(&text).expect("json");
    let snap = payload["pre_reduce_snapshot_step1"]
        .as_array()
        .expect("snapshot array");
    let mut updated = vec![
        PhdComponent {
            state: GaussianState::new_from_array([0.0; 4], Mat::zeros(), Timestamp::ZERO)
                .expect("state"),
            weight: 0.0,
            tag: PhdTag::Untagged,
        };
        MAX_GMPHD_UPDATED
    ];
    let mut n = 0usize;
    for entry in snap {
        let mean: Vec<f64> = entry["mean"]
            .as_array()
            .expect("mean")
            .iter()
            .map(|v| v.as_f64().expect("f"))
            .collect();
        let mut mean4 = [0.0; 4];
        mean4.copy_from_slice(&mean[..4]);
        let mut cov = Mat::<4, 4>::zeros();
        let rows = entry["cov"].as_array().expect("cov");
        for (i, row) in rows.iter().enumerate() {
            for (j, v) in row.as_array().expect("row").iter().enumerate() {
                cov.set(i, j, v.as_f64().expect("c")).expect("set");
            }
        }
        updated[n] = PhdComponent {
            state: GaussianState::new_from_array(mean4, cov, Timestamp::ZERO).expect("gs"),
            weight: entry["weight"].as_f64().expect("w"),
            tag: PhdTag::Untagged,
        };
        n += 1;
    }
    updated.truncate(n);
    let config = GmPhdConfig {
        prob_detection: payload["prob_detection"].as_f64().expect("pd"),
        prob_survival: 1.0 - payload["death_probability"].as_f64().expect("dp"),
        clutter_spatial_density: payload["clutter_spatial_density"].as_f64().expect("csd"),
        missed_distance: payload["missed_distance"].as_f64().expect("md"),
        merge_threshold_sq: payload["merge_threshold"].as_f64().expect("mt"),
        prune_threshold: payload["prune_threshold"].as_f64().expect("pt"),
    };
    (updated, config)
}

#[test]
fn mahalanobis_matches_stonesoup_on_step1_pair() {
    let (updated, _config) = load_step1_snapshot();
    // Python pruned pair (10, 0): weights 0.09523966230060397 / 0.99483466062189, d≈0.667476
    let a = updated[10].state;
    let b = updated[0].state;
    let d = squared_mahalanobis_state(&a, &b).expect("dist");
    assert!((d - 0.6674760438265253).abs() < 1e-3, "dist {d}");
}

#[test]
fn merge_step1_snapshot_matches_stonesoup_count() {
    let (mut updated, config) = load_step1_snapshot();
    let mut n = updated.len();
    let mut merge_pool = vec![
        PhdComponent {
            state: GaussianState::new_from_array([0.0; 4], Mat::zeros(), Timestamp::ZERO)
                .expect("state"),
            weight: 0.0,
            tag: PhdTag::Untagged,
        };
        MAX_GMPHD_UPDATED
    ];
    gm_phd_prune(&mut updated, &mut n, config.prune_threshold).expect("prune");
    assert_eq!(n, 14);
    gm_phd_merge(
        &mut updated,
        &mut n,
        config.merge_threshold_sq,
        &mut merge_pool,
    )
    .expect("merge");
    assert_eq!(n, 8, "merge count on Stone Soup step-1 snapshot");
}
