//! Track quality vs ground truth: following error and identity switches (track jumps).

extern crate alloc;

use alloc::vec::Vec;

use crate::time::Timestamp;

use super::track_to_truth::{
    associate_tracks_to_truths, position_distance, truth_state_at,
    ContinuousTruthTrack, TrackPath, TrackToTruthConfig, TrackTruthAssociation, TruthPath,
};
use crate::metric::rmse::RmseAccumulator;

/// Per filter-path quality (one fused track id).
#[derive(Debug, Clone)]
pub struct FilterTrackQuality {
    pub track_id: u32,
    /// ``TrackToTruth`` segments on this path (each may reference a different truth).
    pub association_segments: usize,
    /// Times the path ended one truth association and later locked onto another (track jump).
    pub identity_switch_count: u32,
    /// Large position error spikes while still associated with the same truth.
    pub jump_event_count: u32,
    pub samples: u32,
    pub samples_in_association: u32,
    /// RMSE (m) vs the truth active for each associated sample.
    pub rmse_m: Option<f64>,
    pub max_error_m: f64,
    pub primary_truth_id: Option<u16>,
}

/// Aggregate quality over established (continuous) truths and all filter paths.
#[derive(Debug, Clone)]
pub struct TrackQualityReport {
    pub filter_paths: Vec<FilterTrackQuality>,
    pub paths_with_identity_switch: usize,
    pub total_identity_switches: u32,
    pub established_truth_count: usize,
    /// Mean RMSE (m) over samples on continuous-truth filter paths.
    pub established_mean_rmse_m: Option<f64>,
    pub established_max_rmse_m: f64,
    /// Mean filter track ids linked per continuous truth minus one (re-acquire handoffs).
    pub mean_filter_handoffs_per_truth: f64,
    pub max_filter_handoffs_per_truth: u32,
    /// Per continuous-truth summary (same order as input ``continuous`` slice).
    pub continuous_truth: Vec<ContinuousTruthQuality>,
}

#[derive(Debug, Clone)]
pub struct ContinuousTruthQuality {
    pub truth_id: u16,
    pub rmse_m: Option<f64>,
    pub max_error_m: f64,
    pub identity_switch_count: u32,
    pub filter_handoffs: u32,
    /// Heuristic 0–1 score (coverage, accuracy, stability).
    pub quality_score: f64,
    pub covered_duration_s: f64,
    pub wall_duration_s: f64,
}

pub(crate) fn identity_switches_for_track(assoc: &[TrackTruthAssociation], track_id: u32) -> u32 {
    let mut segs: Vec<&TrackTruthAssociation> =
        assoc.iter().filter(|a| a.track_id == track_id).collect();
    if segs.len() < 2 {
        return 0;
    }
    segs.sort_by_key(|a| a.range.start);
    segs.windows(2)
        .filter(|w| w[0].truth_id != w[1].truth_id)
        .count() as u32
}

fn truth_for_sample<'a>(
    assoc: &'a [TrackTruthAssociation],
    track_id: u32,
    ts: Timestamp,
) -> Option<u16> {
    assoc
        .iter()
        .filter(|a| a.track_id == track_id && a.range.start.0 <= ts.0 && ts.0 <= a.range.end.0)
        .map(|a| a.truth_id)
        .next()
}

fn assess_filter_path(
    track: &TrackPath,
    truths: &[TruthPath],
    assoc: &[TrackTruthAssociation],
    config: &TrackToTruthConfig,
) -> Result<FilterTrackQuality, crate::error::Error> {
    let track_assoc: Vec<_> = assoc.iter().filter(|a| a.track_id == track.id).collect();
    let identity_switch_count = identity_switches_for_track(assoc, track.id);
    let mut acc = RmseAccumulator::new();
    let mut max_error_m = 0.0f64;
    let mut samples = 0u32;
    let mut samples_in_association = 0u32;
    let mut jump_event_count = 0u32;
    let mut prev_err: Option<f64> = None;
    let mut truth_hits = [0u32; 256];

    for (ts, state) in &track.states {
        samples += 1;
        let Some(tid) = truth_for_sample(assoc, track.id, *ts) else {
            prev_err = None;
            continue;
        };
        samples_in_association += 1;
        if (tid as usize) < truth_hits.len() {
            truth_hits[tid as usize] = truth_hits[tid as usize].saturating_add(1);
        }
        let Some(truth_path) = truths.iter().find(|t| t.id == tid) else {
            continue;
        };
        let Some(truth_state) = truth_state_at(truth_path, *ts) else {
            continue;
        };
        let d = position_distance(*state, truth_state);
        if d > max_error_m {
            max_error_m = d;
        }
        acc.push(state[0] - truth_state[0], state[2] - truth_state[2])?;
        let jump_scale = config
            .position_gate
            .combined_sigma_m(truth_state[0], truth_state[2]);
        if d > jump_scale * 2.0 {
            if prev_err.map(|p| p <= jump_scale).unwrap_or(false) {
                jump_event_count += 1;
            }
        }
        prev_err = Some(d);
    }

    let primary_truth_id = truth_hits
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| *c)
        .and_then(|(i, c)| if *c > 0 { Some(i as u16) } else { None });

    let rmse_m = if acc.count() > 0 {
        Some(acc.rmse()?)
    } else {
        None
    };

    Ok(FilterTrackQuality {
        track_id: track.id,
        association_segments: track_assoc.len(),
        identity_switch_count,
        jump_event_count,
        samples,
        samples_in_association,
        rmse_m,
        max_error_m,
        primary_truth_id,
    })
}

fn quality_score_from(
    rmse_m: Option<f64>,
    threshold_m: f64,
    covered_s: f64,
    wall_s: f64,
    identity_switches: u32,
    filter_handoffs: u32,
) -> f64 {
    let coverage = if wall_s > 1e-9 {
        (covered_s / wall_s).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let accuracy = rmse_m.map(|r| (1.0 - (r / threshold_m).min(1.0)).clamp(0.0, 1.0)).unwrap_or(0.0);
    let stability = 1.0 - 0.25 * (identity_switches.min(4) as f64);
    let continuity = 1.0 - 0.15 * (filter_handoffs.min(5) as f64);
    (coverage * 0.35 + accuracy * 0.45 + stability * 0.1 + continuity * 0.1).clamp(0.0, 1.0)
}

/// Build a full track-quality report for filter paths vs truth paths.
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn assess_track_quality(
    tracks: &[TrackPath],
    truths: &[TruthPath],
    continuous: &[ContinuousTruthTrack],
    config: &TrackToTruthConfig,
) -> Result<TrackQualityReport, crate::error::Error> {
    let assoc = associate_tracks_to_truths(tracks, truths, config)?;
    let accuracy_scale = config.accuracy_scale_m();
    let mut filter_paths = Vec::with_capacity(tracks.len());
    for track in tracks {
        filter_paths.push(assess_filter_path(track, truths, &assoc, config)?);
    }

    let paths_with_identity_switch = filter_paths
        .iter()
        .filter(|q| q.identity_switch_count > 0)
        .count();
    let total_identity_switches = filter_paths.iter().map(|q| q.identity_switch_count).sum();

    let mut handoff_sum = 0u64;
    let mut handoff_max = 0u32;
    let mut continuous_truth = Vec::with_capacity(continuous.len());
    for ct in continuous {
        let handoffs = ct.linked_filter_track_ids.len().saturating_sub(1) as u32;
        handoff_sum += handoffs as u64;
        if handoffs > handoff_max {
            handoff_max = handoffs;
        }
        let mut acc = RmseAccumulator::new();
        let mut max_error_m = 0.0f64;
        let mut identity_switch_count = 0u32;
        for &fid in &ct.linked_filter_track_ids {
            identity_switch_count += identity_switches_for_track(&assoc, fid);
            let Some(track) = tracks.iter().find(|t| t.id == fid) else {
                continue;
            };
            for (ts, state) in &track.states {
                if truth_for_sample(&assoc, fid, *ts) != Some(ct.truth_id) {
                    continue;
                }
                let Some(truth_path) = truths.iter().find(|t| t.id == ct.truth_id) else {
                    continue;
                };
                let Some(truth_state) = truth_state_at(truth_path, *ts) else {
                    continue;
                };
                let d = position_distance(*state, truth_state);
                if d > max_error_m {
                    max_error_m = d;
                }
                acc.push(state[0] - truth_state[0], state[2] - truth_state[2])?;
            }
        }
        let rmse_m = if acc.count() > 0 { Some(acc.rmse()?) } else { None };
        let covered_s = ct.covered.as_seconds();
        let wall_s = (ct.wall_end.0 - ct.wall_start.0) as f64 / 1e9;
        let quality_score = quality_score_from(
            rmse_m,
            accuracy_scale,
            covered_s,
            wall_s,
            identity_switch_count,
            handoffs,
        );
        continuous_truth.push(ContinuousTruthQuality {
            truth_id: ct.truth_id,
            rmse_m,
            max_error_m,
            identity_switch_count,
            filter_handoffs: handoffs,
            quality_score,
            covered_duration_s: covered_s,
            wall_duration_s: wall_s,
        });
    }

    let n_ct = continuous.len().max(1) as f64;
    let mean_filter_handoffs_per_truth = handoff_sum as f64 / n_ct;

    let mut rmse_sum = 0.0f64;
    let mut rmse_n = 0u32;
    let mut established_max_rmse_m = 0.0f64;
    for ctq in &continuous_truth {
        if let Some(r) = ctq.rmse_m {
            rmse_sum += r;
            rmse_n += 1;
            if r > established_max_rmse_m {
                established_max_rmse_m = r;
            }
        }
        if ctq.max_error_m > established_max_rmse_m {
            established_max_rmse_m = ctq.max_error_m;
        }
    }
    let established_mean_rmse_m = if rmse_n > 0 {
        Some(rmse_sum / rmse_n as f64)
    } else {
        None
    };

    Ok(TrackQualityReport {
        filter_paths,
        paths_with_identity_switch,
        total_identity_switches,
        established_truth_count: continuous.len(),
        established_mean_rmse_m,
        established_max_rmse_m,
        mean_filter_handoffs_per_truth,
        max_filter_handoffs_per_truth: handoff_max,
        continuous_truth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use crate::Timestamp;
    use super::super::track_to_truth::associate_tracks_to_truths;

    #[test]
    fn detects_identity_switch_on_one_path() {
        let truths = vec![
            TruthPath {
                id: 0,
                states: vec![
                    (Timestamp::ZERO, [0.0, 0.0, 0.0, 0.0]),
                    (Timestamp::from_nanos(2_000_000_000), [0.0, 0.0, 0.0, 0.0]),
                ],
            },
            TruthPath {
                id: 1,
                states: vec![
                    (Timestamp::from_nanos(3_000_000_000), [500.0, 0.0, 0.0, 0.0]),
                    (Timestamp::from_nanos(5_000_000_000), [500.0, 0.0, 0.0, 0.0]),
                ],
            },
        ];
        let tracks = vec![TrackPath {
            id: 7,
            imm_mu: vec![],
            states: vec![
                (Timestamp::ZERO, [1.0, 0.0, 0.0, 0.0]),
                (Timestamp::from_nanos(1_000_000_000), [1.0, 0.0, 0.0, 0.0]),
                (Timestamp::from_nanos(4_000_000_000), [501.0, 0.0, 0.0, 0.0]),
                (Timestamp::from_nanos(5_000_000_000), [501.0, 0.0, 0.0, 0.0]),
            ],
        }];
        let cfg = TrackToTruthConfig {
            position_gate: crate::TrackTruthPositionGate::FixedRadius(50.0),
            consec_pairs_confirm: 1,
            consec_misses_end: 1,
            switch_score_margin: 0.0,
        };
        let assoc = associate_tracks_to_truths(&tracks, &truths, &cfg).unwrap();
        let switches = identity_switches_for_track(&assoc, 7);
        assert!(switches >= 1, "expected jump from truth 0 to 1, assoc={assoc:?}");
    }
}
