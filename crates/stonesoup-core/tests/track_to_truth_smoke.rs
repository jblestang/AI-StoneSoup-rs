use stonesoup_core::{
    associate_tracks_to_truths, continuous_tracks_per_truth, TrackPath, TrackToTruthConfig,
    TruthPath,
};
use stonesoup_core::Timestamp;

#[test]
fn track_to_truth_links_one_path() {
    let truths = vec![TruthPath {
        id: 0,
        states: vec![
            (Timestamp::ZERO, [100.0, 0.0, 0.0, 0.0]),
            (Timestamp::from_nanos(1_000_000_000), [101.0, 0.0, 0.0, 0.0]),
            (Timestamp::from_nanos(2_000_000_000), [102.0, 0.0, 0.0, 0.0]),
        ],
    }];
    let tracks = vec![TrackPath {
        id: 1,
        states: vec![
            (Timestamp::ZERO, [100.5, 0.0, 0.0, 0.0]),
            (Timestamp::from_nanos(1_000_000_000), [101.5, 0.0, 0.0, 0.0]),
            (Timestamp::from_nanos(2_000_000_000), [102.5, 0.0, 0.0, 0.0]),
        ],
    }];
    let cfg = TrackToTruthConfig {
        association_threshold_m: 10.0,
        consec_pairs_confirm: 2,
        consec_misses_end: 2,
    };
    let assoc = associate_tracks_to_truths(&tracks, &truths, &cfg);
    assert!(!assoc.is_empty());
    let cont = continuous_tracks_per_truth(&assoc, 1);
    assert_eq!(cont.len(), 1);
    assert_eq!(cont[0].truth_id, 0);
}
