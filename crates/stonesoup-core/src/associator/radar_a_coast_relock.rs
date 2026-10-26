//! Correlate TrackToTruth coast-relocks with radar-A plot / JPDA update gaps.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::error::Error;
use crate::time::Timestamp;

use super::switch_diagnosis::{classify_switch, SwitchCause};
use super::track_to_truth::{TrackPath, TrackToTruthConfig, TrackTruthAssociation, TruthPath};
use super::track_truth_gate::TrackTruthPositionGate;

/// Coast-relock handoffs vs radar-A scheduling and fused A JPDA updates.
#[derive(Debug, Clone, Default)]
pub struct RadarACoastRelockStudy {
    pub coast_relock_events: u32,
    /// ≥ one A rotation since last **filter** JPDA meas update from radar A.
    pub filter_no_a_jpda_ge_rotation: u32,
    pub filter_no_a_jpda_between_1s_and_rotation: u32,
    pub filter_a_jpda_within_1s: u32,
    /// ≥ one A rotation since radar A last **plotted** the prior truth target.
    pub prior_truth_a_plot_gap_ge_rotation: u32,
    pub prior_truth_a_plot_gap_between_1s_and_rotation: u32,
    pub prior_truth_a_plotted_within_1s: u32,
    pub mean_filter_a_jpda_gap_s: f64,
    pub mean_prior_truth_a_plot_gap_s: f64,
    pub median_filter_a_jpda_gap_s: f64,
    pub median_prior_truth_a_plot_gap_s: f64,
}

fn step_at_time(ts: Timestamp, dt_s: f64) -> usize {
    if dt_s <= 0.0 {
        return 0;
    }
    libm::round((ts.0 as f64 / 1e9) / dt_s) as usize
}

fn last_step_at_or_before(steps: &[usize], switch_step: usize) -> Option<usize> {
    steps.iter().copied().filter(|s| *s <= switch_step).max()
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

/// Classify coast-relock events using radar-A plot times and per-track A JPDA step log.
pub fn study_radar_a_coast_relocks(
    tracks: &[TrackPath],
    truths: &[TruthPath],
    assoc: &[TrackTruthAssociation],
    config: &TrackToTruthConfig,
    filter_a_meas_steps: &BTreeMap<u32, Vec<usize>>,
    truth_a_plot_steps: &[Vec<usize>],
    dt_s: f64,
    period_a_s: f64,
) -> Result<RadarACoastRelockStudy, Error> {
    let gate = config.position_gate;
    let rotation_steps = libm::ceil(period_a_s / dt_s) as usize;
    let one_s_steps = libm::ceil(1.0 / dt_s) as usize;
    let mut study = RadarACoastRelockStudy::default();
    let mut filter_gaps_s = Vec::new();
    let mut plot_gaps_s = Vec::new();

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
            study.coast_relock_events += 1;
            let switch_step = step_at_time(w[1].range.start, dt_s);
            let from = w[0].truth_id as usize;

            let filter_gap_steps = switch_step
                .saturating_sub(
                    filter_a_meas_steps
                        .get(&track.id)
                        .and_then(|v| last_step_at_or_before(v, switch_step))
                        .unwrap_or(0),
                );
            let filter_gap_s = filter_gap_steps as f64 * dt_s;
            filter_gaps_s.push(filter_gap_s);

            let plot_gap_steps = switch_step.saturating_sub(
                truth_a_plot_steps
                    .get(from)
                    .and_then(|v| last_step_at_or_before(v, switch_step))
                    .unwrap_or(0),
            );
            let plot_gap_s = plot_gap_steps as f64 * dt_s;
            plot_gaps_s.push(plot_gap_s);

            if filter_gap_steps >= rotation_steps {
                study.filter_no_a_jpda_ge_rotation += 1;
            } else if filter_gap_steps >= one_s_steps {
                study.filter_no_a_jpda_between_1s_and_rotation += 1;
            } else {
                study.filter_a_jpda_within_1s += 1;
            }

            if plot_gap_steps >= rotation_steps {
                study.prior_truth_a_plot_gap_ge_rotation += 1;
            } else if plot_gap_steps >= one_s_steps {
                study.prior_truth_a_plot_gap_between_1s_and_rotation += 1;
            } else {
                study.prior_truth_a_plotted_within_1s += 1;
            }
        }
    }

    study.mean_filter_a_jpda_gap_s = if filter_gaps_s.is_empty() {
        0.0
    } else {
        filter_gaps_s.iter().sum::<f64>() / filter_gaps_s.len() as f64
    };
    study.mean_prior_truth_a_plot_gap_s = if plot_gaps_s.is_empty() {
        0.0
    } else {
        plot_gaps_s.iter().sum::<f64>() / plot_gaps_s.len() as f64
    };
    study.median_filter_a_jpda_gap_s = median(filter_gaps_s);
    study.median_prior_truth_a_plot_gap_s = median(plot_gaps_s);

    Ok(study)
}
