//! Diagnose TrackToTruth "identity switches" vs nearest-neighbour label flicker in dense scenes.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::time::Timestamp;

use super::track_quality::identity_switches_for_track;
use super::track_to_truth::{
    associate_track_to_truths, position_distance, truth_state_at, TrackPath, TrackToTruthConfig,
    TrackTruthAssociation, TruthPath,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchCause {
    /// Filter still within gate of both truths; nearest-neighbour id changed.
    AmbiguousNearestNeighbor,
    /// Filter left gate of prior truth before locking onto another.
    CoastThenRelock,
    /// New label is closer; prior truth was already outside gate.
    DivergedRelock,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct IdentitySwitchEvent {
    pub track_id: u32,
    pub from_truth: u16,
    pub to_truth: u16,
    pub switch_time_s: f64,
    pub dist_to_from_m: f64,
    pub dist_to_to_m: f64,
    pub second_nearest_gap_m: f64,
    pub cause: SwitchCause,
}

#[derive(Debug, Clone, Default)]
pub struct SwitchCauseCounts {
    pub ambiguous_nearest_neighbor: u32,
    pub coast_then_relock: u32,
    pub diverged_relock: u32,
    pub unknown: u32,
}

#[derive(Debug, Clone)]
pub struct SwitchDiagnosisSummary {
    pub identity_switches_t2t: u32,
    pub identity_switches_confirm3: u32,
    pub per_step_nearest_truth_flips: u32,
    pub ambiguous_nearest_steps: u32,
    pub total_path_samples: u32,
    pub switch_causes: SwitchCauseCounts,
    pub events: Vec<IdentitySwitchEvent>,
}

fn truth_state_at_index(truth: &TruthPath, ts: Timestamp) -> Option<[f64; 4]> {
    if truth.states.is_empty() {
        return None;
    }
    let t0 = truth.states[0].0;
    if ts.0 < t0.0 {
        return None;
    }
    let dt = ts.0 - t0.0;
    if dt == 0 {
        return Some(truth.states[0].1);
    }
    let idx = ((dt + 125_000_000) / 250_000_000) as usize;
    truth.states
        .get(idx)
        .map(|(_, s)| *s)
        .or_else(|| truth_state_at(truth, ts))
}

fn truths_within_gate(
    track_state: [f64; 4],
    ts: Timestamp,
    truths: &[TruthPath],
    threshold_m: f64,
) -> Vec<(u16, f64)> {
    let mut within = Vec::new();
    for truth in truths {
        let Some(truth_state) = truth_state_at_index(truth, ts) else {
            continue;
        };
        let d = position_distance(track_state, truth_state);
        if d < threshold_m {
            within.push((truth.id, d));
        }
    }
    within.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
    within
}

fn nearest_two_truths(
    track_state: [f64; 4],
    ts: Timestamp,
    truths: &[TruthPath],
    threshold_m: f64,
) -> (Option<(u16, f64)>, Option<(u16, f64)>) {
    let w = truths_within_gate(track_state, ts, truths, threshold_m);
    (w.first().copied(), w.get(1).copied())
}

fn per_step_nearest_flips(
    track: &TrackPath,
    truths: &[TruthPath],
    threshold_m: f64,
) -> (u32, u32, u32) {
    let mut prev: Option<u16> = None;
    let mut flips = 0u32;
    let mut ambiguous = 0u32;
    let mut samples = 0u32;
    for (ts, state) in &track.states {
        samples += 1;
        let (best, second) = nearest_two_truths(*state, *ts, truths, threshold_m);
        if let (Some((_, d1)), Some((_, d2))) = (best, second) {
            if d2 - d1 < 1_000.0 {
                ambiguous += 1;
            }
        }
        if let Some((tid, _)) = best {
            if let Some(p) = prev {
                if p != tid {
                    flips += 1;
                }
            }
            prev = Some(tid);
        } else {
            prev = None;
        }
    }
    (flips, ambiguous, samples)
}

fn classify_switch(
    track: &TrackPath,
    from_truth: u16,
    to_truth: u16,
    switch_ts: Timestamp,
    truths: &[TruthPath],
    threshold_m: f64,
) -> IdentitySwitchEvent {
    let state = track
        .states
        .iter()
        .find(|(t, _)| *t >= switch_ts)
        .map(|(_, s)| *s)
        .or_else(|| track.states.last().map(|(_, s)| *s))
        .unwrap_or([0.0; 4]);

    let dist = |tid: u16| -> f64 {
        truths
            .iter()
            .find(|t| t.id == tid)
            .and_then(|t| truth_state_at_index(t, switch_ts))
            .map(|ts| position_distance(state, ts))
            .unwrap_or(f64::INFINITY)
    };
    let dist_to_from_m = dist(from_truth);
    let dist_to_to_m = dist(to_truth);
    let second_nearest_gap_m = match nearest_two_truths(state, switch_ts, truths, threshold_m) {
        (Some((_, d1)), Some((_, d2))) => d2 - d1,
        _ => f64::INFINITY,
    };

    let cause = if dist_to_from_m < threshold_m && dist_to_to_m < threshold_m {
        SwitchCause::AmbiguousNearestNeighbor
    } else if dist_to_from_m >= threshold_m && dist_to_to_m < threshold_m {
        SwitchCause::CoastThenRelock
    } else if dist_to_to_m < dist_to_from_m && dist_to_to_m < threshold_m {
        SwitchCause::DivergedRelock
    } else {
        SwitchCause::Unknown
    };

    IdentitySwitchEvent {
        track_id: track.id,
        from_truth,
        to_truth,
        switch_time_s: switch_ts.0 as f64 / 1e9,
        dist_to_from_m,
        dist_to_to_m,
        second_nearest_gap_m,
        cause,
    }
}

fn for_each_switch<F>(
    tracks: &[TrackPath],
    truths: &[TruthPath],
    assoc: &[TrackTruthAssociation],
    threshold_m: f64,
    mut f: F,
) where
    F: FnMut(IdentitySwitchEvent),
{
    for track in tracks {
        let mut segs: Vec<&TrackTruthAssociation> =
            assoc.iter().filter(|a| a.track_id == track.id).collect();
        if segs.len() < 2 {
            continue;
        }
        segs.sort_by_key(|a| a.range.start);
        for w in segs.windows(2) {
            if w[0].truth_id == w[1].truth_id {
                continue;
            }
            f(classify_switch(
                track,
                w[0].truth_id,
                w[1].truth_id,
                w[1].range.start,
                truths,
                threshold_m,
            ));
        }
    }
}

/// Analyse identity switches using an existing TrackToTruth labelling (no full re-run).
pub fn diagnose_identity_switches_from_assoc(
    tracks: &[TrackPath],
    truths: &[TruthPath],
    assoc: &[TrackTruthAssociation],
    config: &TrackToTruthConfig,
    max_sample_events: usize,
) -> SwitchDiagnosisSummary {
    let config = *config;
    let mut identity_switches_t2t = 0u32;
    let mut switched_ids = BTreeSet::new();
    for track in tracks {
        let n = identity_switches_for_track(assoc, track.id);
        identity_switches_t2t += n;
        if n > 0 {
            switched_ids.insert(track.id);
        }
    }

    let strict = TrackToTruthConfig {
        association_threshold_m: config.association_threshold_m,
        consec_pairs_confirm: 3,
        consec_misses_end: config.consec_misses_end,
        switch_margin_m: config.switch_margin_m,
    };
    let mut identity_switches_confirm3 = 0u32;
    for track in tracks {
        if !switched_ids.contains(&track.id) {
            continue;
        }
        let a = associate_track_to_truths(track, truths, &strict);
        identity_switches_confirm3 += identity_switches_for_track(&a, track.id);
    }

    let mut per_step_nearest_truth_flips = 0u32;
    let mut ambiguous_nearest_steps = 0u32;
    let mut total_path_samples = 0u32;
    for track in tracks {
        if !switched_ids.contains(&track.id) {
            continue;
        }
        let (f, a, s) = per_step_nearest_flips(track, truths, config.association_threshold_m);
        per_step_nearest_truth_flips += f;
        ambiguous_nearest_steps += a;
        total_path_samples += s;
    }

    let mut switch_causes = SwitchCauseCounts::default();
    let mut events = Vec::new();
    for_each_switch(tracks, truths, assoc, config.association_threshold_m, |ev| {
        match ev.cause {
            SwitchCause::AmbiguousNearestNeighbor => {
                switch_causes.ambiguous_nearest_neighbor += 1;
            }
            SwitchCause::CoastThenRelock => switch_causes.coast_then_relock += 1,
            SwitchCause::DivergedRelock => switch_causes.diverged_relock += 1,
            SwitchCause::Unknown => switch_causes.unknown += 1,
        }
        if events.len() < max_sample_events {
            events.push(ev);
        }
    });

    SwitchDiagnosisSummary {
        identity_switches_t2t,
        identity_switches_confirm3,
        per_step_nearest_truth_flips,
        ambiguous_nearest_steps,
        total_path_samples,
        switch_causes,
        events,
    }
}

impl SwitchCause {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AmbiguousNearestNeighbor => "ambiguous_nearest_neighbor",
            Self::CoastThenRelock => "coast_then_relock",
            Self::DivergedRelock => "diverged_relock",
            Self::Unknown => "unknown",
        }
    }
}
