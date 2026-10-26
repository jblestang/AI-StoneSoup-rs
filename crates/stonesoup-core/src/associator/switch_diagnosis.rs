//! Diagnose TrackToTruth "identity switches" vs nearest-neighbour label flicker in dense scenes.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::error::Error;
use crate::time::Timestamp;

use super::track_quality::identity_switches_for_track;
use super::track_to_truth::{
    associate_track_to_truths, position_distance, truth_state_at, TrackPath, TrackToTruthConfig,
    TrackTruthAssociation, TruthPath,
};
use super::track_truth_gate::TrackTruthPositionGate;

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
    pub coast_relock: CoastRelockStudy,
}

/// Deeper breakdown of ``CoastThenRelock`` switches (filter outside prior-truth gate at re-label time).
#[derive(Debug, Clone, Default)]
pub struct CoastRelockStudy {
    pub event_count: u32,
    /// At handoff, planar distance to new truth is less than to prior truth.
    pub filter_closer_to_new_truth: u32,
    /// Prior truth was outside the gate at handoff but was inside within the lookback window.
    pub prior_in_gate_recently_then_miss: u32,
    /// Gap between TrackToTruth segment end (prior truth) and new segment start ≥ 1 s.
    pub assoc_label_gap_ge_1s: u32,
    pub mean_assoc_gap_s: f64,
    pub median_dist_to_prior_truth_m: f64,
    pub median_dist_to_new_truth_m: f64,
    /// Prior truth planar distance ≥ 2 km at handoff (large lag / wrong track).
    pub prior_truth_dist_ge_2km: u32,
    /// Tracks with ≥1 inverse pair (A→B and B→A coast-relock).
    pub ping_pong_track_count: u32,
    pub ping_pong_event_count: u32,
}

const COAST_LOOKBACK_NS: i64 = 5_000_000_000;
const ASSOC_GAP_1S_NS: i64 = 1_000_000_000;
const PRIOR_FAR_M: f64 = 2_000.0;

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
    gate: TrackTruthPositionGate,
) -> Result<Vec<(u16, f64)>, Error> {
    let mut within = Vec::new();
    for truth in truths {
        let Some(truth_state) = truth_state_at_index(truth, ts) else {
            continue;
        };
        let Some(score) = gate.score(track_state, truth_state)? else {
            continue;
        };
        within.push((truth.id, score));
    }
    within.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
    Ok(within)
}

fn nearest_two_truths(
    track_state: [f64; 4],
    ts: Timestamp,
    truths: &[TruthPath],
    gate: TrackTruthPositionGate,
) -> Result<(Option<(u16, f64)>, Option<(u16, f64)>), Error> {
    let w = truths_within_gate(track_state, ts, truths, gate)?;
    Ok((w.first().copied(), w.get(1).copied()))
}

fn per_step_nearest_flips(
    track: &TrackPath,
    truths: &[TruthPath],
    gate: TrackTruthPositionGate,
) -> Result<(u32, u32, u32), Error> {
    let mut prev: Option<u16> = None;
    let mut flips = 0u32;
    let mut ambiguous = 0u32;
    let mut samples = 0u32;
    let score_ambiguity_margin = match gate {
        TrackTruthPositionGate::FixedRadius(r) => r * 0.05,
        TrackTruthPositionGate::BearingRangeChi2 { .. } => 1.0,
    };
    for (ts, state) in &track.states {
        samples += 1;
        let (best, second) = nearest_two_truths(*state, *ts, truths, gate)?;
        if let (Some((_, s1)), Some((_, s2))) = (best, second) {
            if s2 - s1 < score_ambiguity_margin {
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
    Ok((flips, ambiguous, samples))
}

fn in_gate(
    gate: TrackTruthPositionGate,
    track_state: [f64; 4],
    truth_state: [f64; 4],
) -> bool {
    gate.score(track_state, truth_state)
        .ok()
        .flatten()
        .is_some()
}

fn classify_switch(
    track: &TrackPath,
    from_truth: u16,
    to_truth: u16,
    switch_ts: Timestamp,
    truths: &[TruthPath],
    gate: TrackTruthPositionGate,
) -> Result<IdentitySwitchEvent, Error> {
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
    let second_nearest_gap_m = match nearest_two_truths(state, switch_ts, truths, gate)? {
        (Some((_, s1)), Some((_, s2))) => s2 - s1,
        _ => f64::INFINITY,
    };

    let from_state = truths
        .iter()
        .find(|t| t.id == from_truth)
        .and_then(|t| truth_state_at_index(t, switch_ts));
    let to_state = truths
        .iter()
        .find(|t| t.id == to_truth)
        .and_then(|t| truth_state_at_index(t, switch_ts));

    let from_in = from_state
        .map(|s| in_gate(gate, state, s))
        .unwrap_or(false);
    let to_in = to_state.map(|s| in_gate(gate, state, s)).unwrap_or(false);

    let cause = if from_in && to_in {
        SwitchCause::AmbiguousNearestNeighbor
    } else if !from_in && to_in {
        SwitchCause::CoastThenRelock
    } else if to_in && dist_to_to_m < dist_to_from_m {
        SwitchCause::DivergedRelock
    } else {
        SwitchCause::Unknown
    };

    Ok(IdentitySwitchEvent {
        track_id: track.id,
        from_truth,
        to_truth,
        switch_time_s: switch_ts.0 as f64 / 1e9,
        dist_to_from_m,
        dist_to_to_m,
        second_nearest_gap_m,
        cause,
    })
}

fn prior_truth_in_gate_lookback(
    track: &TrackPath,
    truths: &[TruthPath],
    from_truth: u16,
    switch_ts: Timestamp,
    lookback_ns: i64,
    gate: TrackTruthPositionGate,
) -> Result<bool, Error> {
    let Some(from_path) = truths.iter().find(|t| t.id == from_truth) else {
        return Ok(false);
    };
    let start_ns = switch_ts.0.saturating_sub(lookback_ns);
    for (ts, state) in &track.states {
        if ts.0 < start_ns || ts.0 >= switch_ts.0 {
            continue;
        }
        let Some(truth_state) = truth_state_at_index(from_path, *ts) else {
            continue;
        };
        if in_gate(gate, *state, truth_state) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn median(mut xs: Vec<f64>) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let m = xs.len() / 2;
    if xs.len() % 2 == 1 {
        xs[m]
    } else {
        (xs[m - 1] + xs[m]) * 0.5
    }
}

/// Classify all coast-then-relock handoffs (TrackToTruth segment boundaries).
pub fn study_coast_relocks(
    tracks: &[TrackPath],
    truths: &[TruthPath],
    assoc: &[TrackTruthAssociation],
    config: &TrackToTruthConfig,
) -> Result<CoastRelockStudy, Error> {
    let gate = config.position_gate;
    let mut study = CoastRelockStudy::default();
    let mut gap_secs = Vec::new();
    let mut dist_prior = Vec::new();
    let mut dist_new = Vec::new();
    let mut pairs_by_track: BTreeMap<u32, Vec<(u16, u16)>> = BTreeMap::new();

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
            let ev = classify_switch(
                track,
                w[0].truth_id,
                w[1].truth_id,
                w[1].range.start,
                truths,
                gate,
            )?;
            if ev.cause != SwitchCause::CoastThenRelock {
                continue;
            }
            study.event_count += 1;
            dist_prior.push(ev.dist_to_from_m);
            dist_new.push(ev.dist_to_to_m);
            if ev.dist_to_to_m < ev.dist_to_from_m {
                study.filter_closer_to_new_truth += 1;
            }
            if ev.dist_to_from_m >= PRIOR_FAR_M {
                study.prior_truth_dist_ge_2km += 1;
            }
            if prior_truth_in_gate_lookback(
                track,
                truths,
                w[0].truth_id,
                w[1].range.start,
                COAST_LOOKBACK_NS,
                gate,
            )? {
                study.prior_in_gate_recently_then_miss += 1;
            }
            let gap_ns = w[1].range.start.0.saturating_sub(w[0].range.end.0);
            let gap_s = gap_ns as f64 / 1e9;
            gap_secs.push(gap_s);
            if gap_ns >= ASSOC_GAP_1S_NS {
                study.assoc_label_gap_ge_1s += 1;
            }
            pairs_by_track
                .entry(track.id)
                .or_default()
                .push((w[0].truth_id, w[1].truth_id));
        }
    }

    study.mean_assoc_gap_s = if gap_secs.is_empty() {
        0.0
    } else {
        gap_secs.iter().sum::<f64>() / gap_secs.len() as f64
    };
    study.median_dist_to_prior_truth_m = median(dist_prior);
    study.median_dist_to_new_truth_m = median(dist_new);

    for pairs in pairs_by_track.values() {
        let mut ping_events = 0u32;
        for &(a, b) in pairs {
            if pairs.iter().any(|&(x, y)| x == b && y == a) {
                ping_events += 1;
            }
        }
        if ping_events > 0 {
            study.ping_pong_track_count += 1;
            study.ping_pong_event_count += ping_events;
        }
    }

    Ok(study)
}

fn for_each_switch<F>(
    tracks: &[TrackPath],
    truths: &[TruthPath],
    assoc: &[TrackTruthAssociation],
    gate: TrackTruthPositionGate,
    f: &mut F,
) -> Result<(), Error>
where
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
                gate,
            )?);
        }
    }
    Ok(())
}

/// Analyse identity switches using an existing TrackToTruth labelling (no full re-run).
pub fn diagnose_identity_switches_from_assoc(
    tracks: &[TrackPath],
    truths: &[TruthPath],
    assoc: &[TrackTruthAssociation],
    config: &TrackToTruthConfig,
    max_sample_events: usize,
) -> Result<SwitchDiagnosisSummary, Error> {
    let config = *config;
    let gate = config.position_gate;
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
        position_gate: config.position_gate,
        consec_pairs_confirm: 3,
        consec_misses_end: config.consec_misses_end,
        switch_score_margin: config.switch_score_margin,
    };
    let mut identity_switches_confirm3 = 0u32;
    for track in tracks {
        if !switched_ids.contains(&track.id) {
            continue;
        }
        let a = associate_track_to_truths(track, truths, &strict)?;
        identity_switches_confirm3 += identity_switches_for_track(&a, track.id);
    }

    let mut per_step_nearest_truth_flips = 0u32;
    let mut ambiguous_nearest_steps = 0u32;
    let mut total_path_samples = 0u32;
    for track in tracks {
        if !switched_ids.contains(&track.id) {
            continue;
        }
        let (f, a, s) = per_step_nearest_flips(track, truths, gate)?;
        per_step_nearest_truth_flips += f;
        ambiguous_nearest_steps += a;
        total_path_samples += s;
    }

    let mut switch_causes = SwitchCauseCounts::default();
    let mut events = Vec::new();
    for_each_switch(tracks, truths, assoc, gate, &mut |ev| {
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
    })?;

    let coast_relock = study_coast_relocks(tracks, truths, assoc, &config)?;

    Ok(SwitchDiagnosisSummary {
        identity_switches_t2t,
        identity_switches_confirm3,
        per_step_nearest_truth_flips,
        ambiguous_nearest_steps,
        total_path_samples,
        switch_causes,
        events,
        coast_relock,
    })
}

impl SwitchCause {
    pub fn as_str(self) -> &'static str {
        match self {
            SwitchCause::AmbiguousNearestNeighbor => "ambiguous_nearest_neighbor",
            SwitchCause::CoastThenRelock => "coast_then_relock",
            SwitchCause::DivergedRelock => "diverged_relock",
            SwitchCause::Unknown => "unknown",
        }
    }
}
