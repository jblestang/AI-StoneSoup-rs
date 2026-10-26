//! Track-to-truth association (Stone Soup ``TrackToTruth`` counting technique).

extern crate alloc;

use alloc::vec::Vec;

use crate::error::Error;
use crate::time::{Duration, Timestamp};

use super::track_truth_gate::TrackTruthPositionGate;

#[derive(Debug, Clone)]
pub struct TrackPath {
    pub id: u32,
    pub states: Vec<(Timestamp, [f64; 4])>,
    /// IMM model probabilities `[CV, CA, CT]` when logged (dual-radar filter paths).
    pub imm_mu: Vec<(Timestamp, [f64; 3])>,
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

/// Per-radar revisit timing for TrackToTruth segment rules (rotation period vs filter step).
#[derive(Debug, Clone, Copy)]
pub struct RadarRevisitTrackToTruth {
    pub period_s: f64,
    pub dt_s: f64,
    pub consec_pairs_confirm: u8,
}

impl RadarRevisitTrackToTruth {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn new(period_s: f64, dt_s: f64, consec_pairs_confirm: u8) -> Self {
        Self {
            period_s,
            dt_s,
            consec_pairs_confirm,
        }
    }

    /// Consecutive filter steps outside the gate before ending a label (~one beam rotation).
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn consec_misses_end(&self) -> u8 {
        if self.dt_s <= 0.0 || self.period_s <= 0.0 {
            return 2;
        }
        let steps = libm::ceil(self.period_s / self.dt_s);
        steps.clamp(1.0, 255.0) as u8
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TrackToTruthConfig {
    pub position_gate: TrackTruthPositionGate,
    pub consec_pairs_confirm: u8,
    pub consec_misses_end: u8,
    /// In χ² / score space: a challenger must beat the current truth by at least this margin.
    pub switch_score_margin: f64,
}

impl Default for TrackToTruthConfig {
    fn default() -> Self {
        Self {
            position_gate: TrackTruthPositionGate::FixedRadius(2_000.0),
            consec_pairs_confirm: 3,
            consec_misses_end: 2,
            switch_score_margin: 0.0,
        }
    }
}

impl TrackToTruthConfig {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn from_radar_revisit(
        position_gate: TrackTruthPositionGate,
        revisit: RadarRevisitTrackToTruth,
        switch_score_margin: f64,
    ) -> Self {
        Self {
            position_gate,
            consec_pairs_confirm: revisit.consec_pairs_confirm,
            consec_misses_end: revisit.consec_misses_end(),
            switch_score_margin,
        }
    }

    /// Planar scale for RMSE quality heuristics (mid-range σ for bearing–range gates).
    pub fn accuracy_scale_m(&self) -> f64 {
        match self.position_gate {
            TrackTruthPositionGate::FixedRadius(r) => r,
            TrackTruthPositionGate::BearingRangeChi2 {
                sensor_x,
                sensor_y,
                ..
            } => self
                .position_gate
                .combined_sigma_m(sensor_x + 200_000.0, sensor_y),
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

fn score_to_truth(
    truths: &[TruthPath],
    truth_id: u16,
    ts: Timestamp,
    track_state: [f64; 4],
    gate: TrackTruthPositionGate,
) -> Result<Option<f64>, Error> {
    let Some(truth) = truths.iter().find(|t| t.id == truth_id) else {
        return Ok(None);
    };
    let Some(truth_state) = truth_state_at(truth, ts) else {
        return Ok(None);
    };
    gate.score(track_state, truth_state)
}

fn best_gated_truth(
    truths: &[TruthPath],
    ts: Timestamp,
    track_state: [f64; 4],
    gate: TrackTruthPositionGate,
) -> Result<(Option<u16>, f64), Error> {
    let mut best_score = f64::INFINITY;
    let mut best_id: Option<u16> = None;

    for truth in truths {
        if truth.states.is_empty() {
            continue;
        }
        let first = truth.states[0].0;
        let last = truth.states[truth.states.len() - 1].0;
        if first > ts || last < ts {
            continue;
        }
        let Some(truth_state) = truth_state_at(truth, ts) else {
            continue;
        };
        let Some(score) = gate.score(track_state, truth_state)? else {
            continue;
        };
        if score < best_score {
            best_score = score;
            best_id = Some(truth.id);
        }
    }

    Ok((
        best_id,
        if best_id.is_some() {
            best_score
        } else {
            f64::INFINITY
        },
    ))
}

/// Associate one filter track path to truth paths (same logic as Stone Soup ``TrackToTruth``).
pub fn associate_track_to_truths(
    track: &TrackPath,
    truths: &[TruthPath],
    config: &TrackToTruthConfig,
) -> Result<Vec<TrackTruthAssociation>, Error> {
    let mut out = Vec::new();
    if track.states.is_empty() {
        return Ok(out);
    }

    let gate = config.position_gate;
    let mut current_truth: Option<u16> = None;
    let mut potential_truth: Option<u16> = None;
    let mut n_potential_successes = 0u8;
    let mut n_failures = 0u8;
    let mut potential_start: Option<Timestamp> = None;
    let mut start_ts: Option<Timestamp> = None;
    let mut end_ts: Option<Timestamp> = None;

    for (ts, track_state) in &track.states {
        let (mut min_truth, mut min_score) = if let Some(cur) = current_truth {
            if let Some(truth) = truths.iter().find(|t| t.id == cur) {
                if let Some(truth_state) = truth_state_at(truth, *ts) {
                    if let Some(sq) = gate.score_sustain(*track_state, truth_state)? {
                        (Some(cur), sq)
                    } else {
                        best_gated_truth(truths, *ts, *track_state, gate)?
                    }
                } else {
                    best_gated_truth(truths, *ts, *track_state, gate)?
                }
            } else {
                best_gated_truth(truths, *ts, *track_state, gate)?
            }
        } else {
            best_gated_truth(truths, *ts, *track_state, gate)?
        };

        if config.switch_score_margin > 0.0 {
            if let Some(cur) = current_truth {
                if let (Some(nt), Some(score_cur)) = (
                    min_truth,
                    score_to_truth(truths, cur, *ts, *track_state, gate).ok().flatten(),
                ) {
                    if nt != cur && min_score + config.switch_score_margin >= score_cur {
                        min_truth = Some(cur);
                        min_score = score_cur;
                    }
                }
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
        let _ = min_score;
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

    Ok(out)
}

pub fn associate_tracks_to_truths(
    tracks: &[TrackPath],
    truths: &[TruthPath],
    config: &TrackToTruthConfig,
) -> Result<Vec<TrackTruthAssociation>, Error> {
    let mut all = Vec::new();
    for track in tracks {
        all.extend(associate_track_to_truths(track, truths, config)?);
    }
    Ok(all)
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

#[cfg(test)]
mod revisit_tests {
    use super::*;

    #[test]
    fn consec_misses_end_matches_rotation_period() {
        let r = RadarRevisitTrackToTruth::new(6.0, 0.25, 3);
        assert_eq!(r.consec_misses_end(), 24);
        let cfg = TrackToTruthConfig::from_radar_revisit(
            TrackTruthPositionGate::FixedRadius(1000.0),
            r,
            0.0,
        );
        assert_eq!(cfg.consec_misses_end, 24);
        let b = RadarRevisitTrackToTruth::new(3.0, 0.25, 3);
        assert_eq!(b.consec_misses_end(), 12);
    }
}
