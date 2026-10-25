//! Track-to-truth association (Stone Soup ``TrackToTruth`` counting technique).

extern crate alloc;

use alloc::vec::Vec;

use crate::time::{Duration, Timestamp};

#[derive(Debug, Clone)]
pub struct TrackPath {
    pub id: u32,
    pub states: Vec<(Timestamp, [f64; 4])>,
}

#[derive(Debug, Clone)]
pub struct TruthPath {
    pub id: u16,
    pub states: Vec<(Timestamp, [f64; 4])>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeRange {
    pub start: Timestamp,
    pub end: Timestamp,
}

#[derive(Debug, Clone)]
pub struct TrackTruthAssociation {
    pub track_id: u32,
    pub truth_id: u16,
    pub range: TimeRange,
}

#[derive(Debug, Clone)]
pub struct TrackToTruthConfig {
    pub association_threshold_m: f64,
    pub consec_pairs_confirm: u8,
    pub consec_misses_end: u8,
}

impl Default for TrackToTruthConfig {
    fn default() -> Self {
        Self {
            association_threshold_m: 2_000.0,
            consec_pairs_confirm: 3,
            consec_misses_end: 2,
        }
    }
}

pub(crate) fn position_distance(track: [f64; 4], truth: [f64; 4]) -> f64 {
    let dx = track[0] - truth[0];
    let dy = track[2] - truth[2];
    libm::sqrt(dx * dx + dy * dy)
}

pub(crate) fn truth_state_at(truth: &TruthPath, ts: Timestamp) -> Option<[f64; 4]> {
    truth
        .states
        .iter()
        .find(|(t, _)| *t == ts)
        .map(|(_, s)| *s)
}

/// Associate one filter track path to truth paths (same logic as Stone Soup ``TrackToTruth``).
pub fn associate_track_to_truths(
    track: &TrackPath,
    truths: &[TruthPath],
    config: &TrackToTruthConfig,
) -> Vec<TrackTruthAssociation> {
    let mut out = Vec::new();
    if track.states.is_empty() {
        return out;
    }

    let mut current_truth: Option<u16> = None;
    let mut potential_truth: Option<u16> = None;
    let mut n_potential_successes = 0u8;
    let mut n_failures = 0u8;
    let mut potential_start: Option<Timestamp> = None;
    let mut start_ts: Option<Timestamp> = None;
    let mut end_ts: Option<Timestamp> = None;

    for (ts, track_state) in &track.states {
        let mut min_dist = config.association_threshold_m;
        let mut min_truth: Option<u16> = None;

        for truth in truths {
            if truth.states.is_empty() {
                continue;
            }
            let first = truth.states[0].0;
            let last = truth.states[truth.states.len() - 1].0;
            if first > *ts || last < *ts {
                continue;
            }
            let Some(truth_state) = truth_state_at(truth, *ts) else {
                continue;
            };
            let d = position_distance(*track_state, truth_state);
            if d < min_dist {
                min_dist = d;
                min_truth = Some(truth.id);
            }
        }

        if current_truth.is_none() {
            if min_truth.is_none() {
                n_potential_successes = 0;
                potential_truth = None;
                potential_start = None;
            } else if potential_truth != min_truth {
                potential_truth = min_truth;
                n_potential_successes = 1;
                potential_start = Some(*ts);
            } else {
                n_potential_successes = n_potential_successes.saturating_add(1);
            }
            if n_potential_successes >= config.consec_pairs_confirm {
                current_truth = min_truth;
                start_ts = potential_start;
                end_ts = Some(*ts);
                potential_start = None;
                potential_truth = None;
                n_potential_successes = 0;
            }
        } else if let Some(cur) = current_truth {
            if min_truth == Some(cur) {
                n_failures = 0;
                end_ts = Some(*ts);
            } else {
                n_failures = n_failures.saturating_add(1);
                if min_truth.is_some() && min_truth == potential_truth {
                    n_potential_successes = n_potential_successes.saturating_add(1);
                } else {
                    potential_truth = min_truth;
                    potential_start = Some(*ts);
                    n_potential_successes = 1;
                }
                if n_failures >= config.consec_misses_end {
                    if let (Some(s), Some(e)) = (start_ts, end_ts) {
                        out.push(TrackTruthAssociation {
                            track_id: track.id,
                            truth_id: cur,
                            range: TimeRange { start: s, end: e },
                        });
                    }
                    if n_potential_successes >= config.consec_pairs_confirm {
                        current_truth = potential_truth;
                        start_ts = potential_start;
                        end_ts = Some(*ts);
                    } else {
                        current_truth = None;
                        start_ts = None;
                        end_ts = None;
                    }
                    potential_start = None;
                    potential_truth = None;
                    n_potential_successes = 0;
                }
            }
        }
    }

    if let Some(cur) = current_truth {
        if let (Some(s), Some(e)) = (start_ts, end_ts) {
            out.push(TrackTruthAssociation {
                track_id: track.id,
                truth_id: cur,
                range: TimeRange { start: s, end: e },
            });
        }
    }

    out
}

pub fn associate_tracks_to_truths(
    tracks: &[TrackPath],
    truths: &[TruthPath],
    config: &TrackToTruthConfig,
) -> Vec<TrackTruthAssociation> {
    let mut all = Vec::new();
    for track in tracks {
        all.extend(associate_track_to_truths(track, truths, config));
    }
    all
}

#[derive(Debug, Clone)]
pub struct ContinuousTruthTrack {
    pub truth_id: u16,
    pub wall_start: Timestamp,
    pub wall_end: Timestamp,
    pub covered: Duration,
    pub linked_filter_track_ids: Vec<u32>,
}

fn merge_ranges(ranges: &mut Vec<TimeRange>) {
    if ranges.len() < 2 {
        return;
    }
    ranges.sort_by_key(|r| r.start);
    let mut merged: Vec<TimeRange> = Vec::new();
    for r in ranges.drain(..) {
        if let Some(last) = merged.last_mut() {
            if r.start.0 <= last.end.0 {
                if r.end.0 > last.end.0 {
                    last.end = r.end;
                }
                continue;
            }
        }
        merged.push(r);
    }
    *ranges = merged;
}

/// One logical target: merge all ``TrackToTruth`` intervals that reference the same truth id.
pub fn continuous_tracks_per_truth(
    associations: &[TrackTruthAssociation],
    num_truths: u16,
) -> Vec<ContinuousTruthTrack> {
    let mut out = Vec::new();
    for tid in 0..num_truths {
        let mut ranges = Vec::new();
        let mut ids = Vec::new();
        for a in associations {
            if a.truth_id == tid {
                ranges.push(a.range);
                if !ids.contains(&a.track_id) {
                    ids.push(a.track_id);
                }
            }
        }
        if ranges.is_empty() {
            continue;
        }
        merge_ranges(&mut ranges);
        let wall_start = ranges[0].start;
        let wall_end = ranges[ranges.len() - 1].end;
        let mut covered_ns = 0i64;
        for r in &ranges {
            covered_ns += r.end.0.saturating_sub(r.start.0);
        }
        out.push(ContinuousTruthTrack {
            truth_id: tid,
            wall_start,
            wall_end,
            covered: Duration(covered_ns.max(0)),
            linked_filter_track_ids: ids,
        });
    }
    out
}
