//! ``MultiMeasurementInitiator`` (Stone Soup ``initiator.simple``).

use alloc::vec::Vec;

use crate::deleter::{holding_should_delete, HoldingDeleter};
use crate::error::Error;
use crate::state::GaussianState;

/// Tentative track held until ``min_points`` associated updates (``updates_only``).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoldingGaussianTrack {
    pub tail: GaussianState,
    /// Associated measurement updates when ``updates_only`` is true.
    pub update_count: u8,
    /// Scan steps since the last measurement update (for ``UpdateTimeStepsDeleter``).
    pub steps_since_update: u16,
}

impl HoldingGaussianTrack {
    #[must_use]
    pub fn from_init(tail: GaussianState) -> Self {
        Self {
            tail,
            update_count: 1,
            steps_since_update: 0,
        }
    }

    /// End one association scan (predict-only if ``measurement_update`` is false).
    pub fn finish_scan(&mut self, tail: GaussianState, measurement_update: bool, updates_only: bool) {
        self.tail = tail;
        if measurement_update {
            self.steps_since_update = 0;
            if updates_only {
                self.update_count = self.update_count.saturating_add(1);
            } else {
                self.update_count = self.update_count.saturating_add(1);
            }
        } else {
            self.steps_since_update = self.steps_since_update.saturating_add(1);
        }
    }

    /// Scan with detections absent (coast holding / deleter aging only).
    pub fn finish_predict_only_step(&mut self) {
        self.steps_since_update = self.steps_since_update.saturating_add(1);
    }
}

/// Holds tentative tracks and confirms them at ``min_points``.
pub struct MultiMeasurementInitiator {
    pub min_points: u8,
    pub updates_only: bool,
    holding: Vec<HoldingGaussianTrack>,
    max_holding: usize,
}

impl MultiMeasurementInitiator {
    #[must_use]
    pub fn new(min_points: u8, updates_only: bool, max_holding: usize) -> Self {
        Self {
            min_points,
            updates_only,
            holding: Vec::new(),
            max_holding,
        }
    }

    pub fn holding_len(&self) -> usize {
        self.holding.len()
    }

    pub fn holding_tracks(&self) -> &[HoldingGaussianTrack] {
        &self.holding
    }

    pub fn holding_tracks_mut(&mut self) -> &mut [HoldingGaussianTrack] {
        &mut self.holding
    }

    pub fn try_add_holding(&mut self, init: GaussianState) -> Result<(), Error> {
        if self.holding.len() >= self.max_holding {
            return Err(Error::CapacityExceeded);
        }
        self.holding.push(HoldingGaussianTrack::from_init(init));
        Ok(())
    }

    /// Promote confirmed tracks, then apply the holding deleter (Tutorial 09 order).
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn confirm_and_prune(
        &mut self,
        deleter: &HoldingDeleter,
    ) -> Result<Vec<GaussianState>, Error> {
        let min = self.min_points;
        let mut confirmed = Vec::new();
        let mut kept = Vec::with_capacity(self.holding.len());
        for ht in self.holding.drain(..) {
            if ht.update_count >= min {
                confirmed.push(ht.tail);
                continue;
            }
            if holding_should_delete(&ht.tail, ht.steps_since_update as usize, deleter)? {
                continue;
            }
            kept.push(ht);
        }
        self.holding = kept;
        Ok(confirmed)
    }

    /// Age holdings on steps with no detections; returns newly confirmed tracks.
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn predict_only_step(
        &mut self,
        deleter: &HoldingDeleter,
    ) -> Result<Vec<GaussianState>, Error> {
        for ht in &mut self.holding {
            ht.finish_predict_only_step();
        }
        self.confirm_and_prune(deleter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deleter::HoldingDeleter;
    use crate::linalg::Mat;
    use crate::time::Timestamp;

    fn dummy_state() -> GaussianState {
        GaussianState::new_from_array([0.0; 4], Mat::zeros(), Timestamp::ZERO).unwrap()
    }

    #[test]
    fn confirms_at_min_points() {
        let mut ini = MultiMeasurementInitiator::new(2, true, 8);
        ini.try_add_holding(dummy_state()).unwrap();
        ini.holding_tracks_mut()[0].update_count = 2;
        let out = ini
            .confirm_and_prune(&HoldingDeleter::CovarianceTrace(1e9))
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(ini.holding_len(), 0);
    }

    #[test]
    fn time_deleter_drops_stale_holding() {
        let mut ini = MultiMeasurementInitiator::new(5, true, 8);
        ini.try_add_holding(dummy_state()).unwrap();
        ini.holding_tracks_mut()[0].steps_since_update = 4;
        let out = ini
            .confirm_and_prune(&HoldingDeleter::update_time_steps(3))
            .unwrap();
        assert!(out.is_empty());
        assert_eq!(ini.holding_len(), 0);
    }
}
