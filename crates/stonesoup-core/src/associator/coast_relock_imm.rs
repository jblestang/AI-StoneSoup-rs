//! Coast-then-relock vs IMM sub-model (CV / CA / CT) probabilities.

extern crate alloc;

use alloc::vec::Vec;

use crate::error::Error;
use crate::time::Timestamp;

use super::switch_diagnosis::list_coast_relock_handoffs;
use super::track_to_truth::{
    TrackPath, TrackToTruthConfig, TrackTruthAssociation, TruthPath,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct ImmDominantCounts {
    pub cv: u32,
    pub ca: u32,
    pub ct: u32,
}

impl ImmDominantCounts {
    pub fn record(&mut self, mu: [f64; 3]) {
        match dominant_imm_index(mu) {
            0 => self.cv += 1,
            1 => self.ca += 1,
            _ => self.ct += 1,
        }
    }

    pub fn fractions(&self) -> (f64, f64, f64) {
        let n = (self.cv + self.ca + self.ct) as f64;
        if n <= 0.0 {
            return (0.0, 0.0, 0.0);
        }
        (self.cv as f64 / n, self.ca as f64 / n, self.ct as f64 / n)
    }
}

#[derive(Debug, Clone, Default)]
pub struct CoastRelockImmStudy {
    pub coast_relock_events: u32,
    pub imm_dominant_at_switch: ImmDominantCounts,
    pub mean_mu_at_switch: [f64; 3],
    /// Last ``coast_window_s`` before each handoff (default one radar-A rotation).
    pub coast_window_s: f64,
    pub imm_dominant_coast_window: ImmDominantCounts,
    pub mean_mu_coast_window: [f64; 3],
    pub baseline_samples: u32,
    pub imm_dominant_baseline: ImmDominantCounts,
    pub mean_mu_baseline: [f64; 3],
    /// Fraction of coast-relock handoffs where filter IMM mode ≠ prior-truth motion mode.
    pub imm_truth_mode_mismatch_at_switch: f64,
    pub imm_truth_crosstab_at_switch: [[u32; 3]; 3],
}

pub fn dominant_imm_index(mu: [f64; 3]) -> u8 {
    if mu[1] >= mu[0] && mu[1] >= mu[2] {
        1
    } else if mu[2] >= mu[0] && mu[2] >= mu[1] {
        2
    } else {
        0
    }
}

fn imm_mu_at_or_before(track: &TrackPath, ts: Timestamp) -> Option<[f64; 3]> {
    track
        .imm_mu
        .iter()
        .filter(|(t, _)| t.0 <= ts.0)
        .max_by_key(|(t, _)| t.0)
        .map(|(_, m)| *m)
}

fn mean_mu(samples: &[[f64; 3]]) -> [f64; 3] {
    if samples.is_empty() {
        return [0.0; 3];
    }
    let mut s = [0.0; 3];
    for m in samples {
        for i in 0..3 {
            s[i] += m[i];
        }
    }
    let n = samples.len() as f64;
    [s[0] / n, s[1] / n, s[2] / n]
}

/// ``truth_motion_mode``: 0 = CV, 1 = CA, 2 = CT (dual-radar ground truth).
fn truth_mode_at_or_before(
    log: &[(Timestamp, u8)],
    ts: Timestamp,
) -> Option<u8> {
    log.iter()
        .filter(|(t, _)| t.0 <= ts.0)
        .max_by_key(|(t, _)| t.0)
        .map(|(_, m)| *m)
}

pub fn study_coast_relock_imm(
    tracks: &[TrackPath],
    truths: &[TruthPath],
    assoc: &[TrackTruthAssociation],
    config: &TrackToTruthConfig,
    truth_motion_log: &[Vec<(Timestamp, u8)>],
    coast_window_s: f64,
) -> Result<CoastRelockImmStudy, Error> {
    let handoffs = list_coast_relock_handoffs(tracks, truths, assoc, config)?;
    let mut study = CoastRelockImmStudy {
        coast_window_s,
        coast_relock_events: handoffs.len() as u32,
        ..Default::default()
    };

    let window_ns = (coast_window_s * 1e9) as i64;
    let track_by_id: alloc::collections::BTreeMap<u32, &TrackPath> =
        tracks.iter().map(|t| (t.id, t)).collect();

    let mut switch_mus = Vec::new();
    let mut window_mus = Vec::new();
    let mut mismatches = 0u32;

    for h in &handoffs {
        let Some(track) = track_by_id.get(&h.track_id) else {
            continue;
        };
        let switch_ts = h.switch_ts;
        if let Some(mu) = imm_mu_at_or_before(track, switch_ts) {
            switch_mus.push(mu);
            study.imm_dominant_at_switch.record(mu);
            let imm_d = dominant_imm_index(mu) as usize;
            let from = h.from_truth as usize;
            if let Some(log) = truth_motion_log.get(from) {
                if let Some(tm) = truth_mode_at_or_before(log, switch_ts) {
                    let tm = (tm as usize).min(2);
                    study.imm_truth_crosstab_at_switch[imm_d][tm] += 1;
                    if imm_d != tm {
                        mismatches += 1;
                    }
                }
            }
        }
        let start_ns = switch_ts.0.saturating_sub(window_ns);
        for (ts, mu) in &track.imm_mu {
            if ts.0 >= start_ns && ts.0 <= switch_ts.0 {
                window_mus.push(*mu);
                study.imm_dominant_coast_window.record(*mu);
            }
        }
    }

    study.mean_mu_at_switch = mean_mu(&switch_mus);
    study.mean_mu_coast_window = mean_mu(&window_mus);
    if study.coast_relock_events > 0 {
        study.imm_truth_mode_mismatch_at_switch =
            mismatches as f64 / study.coast_relock_events as f64;
    }

    let mut base = Vec::new();
    for track in tracks {
        for (_, mu) in &track.imm_mu {
            base.push(*mu);
            study.imm_dominant_baseline.record(*mu);
        }
    }
    study.baseline_samples = base.len() as u32;
    study.mean_mu_baseline = mean_mu(&base);

    Ok(study)
}
