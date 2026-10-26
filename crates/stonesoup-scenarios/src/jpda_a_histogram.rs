//! Histogram of radar-A JPDA measurement association mass (``meas_p``).

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use stonesoup_core::{
    list_coast_relock_handoffs, TrackPath, TrackToTruthConfig, TrackTruthAssociation, TruthPath,
};

pub const JPDA_MEAS_UPDATE_THRESH: f64 = 0.35;

/// One active-track × radar-A pass sample.
#[derive(Debug, Clone, Copy)]
pub struct AMeasPassSample {
    pub track_id: u32,
    pub step: u16,
    pub meas_p: f32,
}

#[derive(Debug, Clone, Default)]
pub struct MeasPHistogram {
    pub zero: u64,
    pub gt_zero_lt_0_1: u64,
    pub ge_0_1_lt_0_2: u64,
    pub ge_0_2_lt_0_35: u64,
    pub ge_0_35_lt_0_5: u64,
    pub ge_0_5_lt_0_75: u64,
    pub ge_0_75_le_1_0: u64,
    pub gt_1_0: u64,
}

impl MeasPHistogram {
    pub fn total(&self) -> u64 {
        self.zero
            + self.gt_zero_lt_0_1
            + self.ge_0_1_lt_0_2
            + self.ge_0_2_lt_0_35
            + self.ge_0_35_lt_0_5
            + self.ge_0_5_lt_0_75
            + self.ge_0_75_le_1_0
            + self.gt_1_0
    }

    pub fn push(&mut self, meas_p: f64) {
        if meas_p <= 1e-8 {
            self.zero += 1;
        } else if meas_p < 0.1 {
            self.gt_zero_lt_0_1 += 1;
        } else if meas_p < 0.2 {
            self.ge_0_1_lt_0_2 += 1;
        } else if meas_p < JPDA_MEAS_UPDATE_THRESH {
            self.ge_0_2_lt_0_35 += 1;
        } else if meas_p < 0.5 {
            self.ge_0_35_lt_0_5 += 1;
        } else if meas_p < 0.75 {
            self.ge_0_5_lt_0_75 += 1;
        } else if meas_p <= 1.0 {
            self.ge_0_75_le_1_0 += 1;
        } else {
            self.gt_1_0 += 1;
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ThresholdSensitivityRow {
    pub tau: f64,
    pub passes_with_meas_p_ge_tau: u64,
    pub fraction_of_all_passes: f64,
}

#[derive(Debug, Clone, Default)]
pub struct JpdaAHistogramReport {
    pub operational_threshold: f64,
    pub total_a_track_passes: u64,
    pub all_active_passes: MeasPHistogram,
    pub coast_relock_track_passes: MeasPHistogram,
    /// ``meas_p`` on radar-A steps when A plotted the prior truth (one A rotation before handoff).
    pub coast_relock_at_a_plot_of_prior_truth: MeasPHistogram,
    pub coast_relock_track_count: u32,
    pub coast_relock_at_plot_sample_count: u64,
    pub threshold_sensitivity: Vec<ThresholdSensitivityRow>,
    pub threshold_sensitivity_coast_relock_tracks: Vec<ThresholdSensitivityRow>,
}

const TAU_GRID: [f64; 7] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35];

fn build_sensitivity(
    total: u64,
    samples: &[AMeasPassSample],
    filter_track: Option<&BTreeSet<u32>>,
) -> Vec<ThresholdSensitivityRow> {
    let denom = total.max(1) as f64;
    TAU_GRID
        .iter()
        .map(|&tau| {
            let n = samples
                .iter()
                .filter(|s| {
                    if let Some(set) = filter_track {
                        if !set.contains(&s.track_id) {
                            return false;
                        }
                    }
                    (s.meas_p as f64) >= tau
                })
                .count() as u64;
            ThresholdSensitivityRow {
                tau,
                passes_with_meas_p_ge_tau: n,
                fraction_of_all_passes: n as f64 / denom,
            }
        })
        .collect()
}

fn lookup_meas_p(samples: &[AMeasPassSample], track_id: u32, step: usize) -> Option<f64> {
    let step = step as u16;
    samples
        .iter()
        .find(|s| s.track_id == track_id && s.step == step)
        .map(|s| s.meas_p as f64)
}

/// Build histograms and τ sensitivity from logged A passes and TrackToTruth coast-relocks.
pub fn build_jpda_a_histogram_report(
    samples: &[AMeasPassSample],
    tracks: &[TrackPath],
    truths: &[TruthPath],
    assoc: &[TrackTruthAssociation],
    config: &TrackToTruthConfig,
    truth_a_plot_steps: &[Vec<usize>],
    dt_s: f64,
    period_a_s: f64,
) -> Result<JpdaAHistogramReport, stonesoup_core::Error> {
    let mut report = JpdaAHistogramReport {
        operational_threshold: JPDA_MEAS_UPDATE_THRESH,
        total_a_track_passes: samples.len() as u64,
        ..Default::default()
    };

    for s in samples {
        report.all_active_passes.push(s.meas_p as f64);
    }

    let handoffs = list_coast_relock_handoffs(tracks, truths, assoc, config)?;
    let rotation_steps = libm::ceil(period_a_s / dt_s) as usize;
    let mut coast_track_ids = BTreeSet::new();
    let mut at_plot = MeasPHistogram::default();
    let mut at_plot_n = 0u64;

    for h in &handoffs {
        coast_track_ids.insert(h.track_id);
        let switch_step = libm::round((h.switch_ts.0 as f64 / 1e9) / dt_s) as usize;
        let lo = switch_step.saturating_sub(rotation_steps);
        let from = h.from_truth as usize;
        if let Some(plot_steps) = truth_a_plot_steps.get(from) {
            for &ps in plot_steps {
                if ps < lo || ps > switch_step {
                    continue;
                }
                if let Some(p) = lookup_meas_p(samples, h.track_id, ps) {
                    at_plot.push(p);
                    at_plot_n += 1;
                }
            }
        }
    }

    for s in samples {
        if coast_track_ids.contains(&s.track_id) {
            report.coast_relock_track_passes.push(s.meas_p as f64);
        }
    }
    report.coast_relock_at_a_plot_of_prior_truth = at_plot;
    report.coast_relock_at_plot_sample_count = at_plot_n;
    report.coast_relock_track_count = coast_track_ids.len() as u32;

    report.threshold_sensitivity = build_sensitivity(report.total_a_track_passes, samples, None);
    report.threshold_sensitivity_coast_relock_tracks =
        build_sensitivity(report.total_a_track_passes, samples, Some(&coast_track_ids));

    Ok(report)
}
