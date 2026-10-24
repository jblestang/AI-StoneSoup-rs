use stonesoup_core::linalg::mat_vec_mul;
use stonesoup_core::{ConstantVelocity2d, Duration, Error, GroundTruthState, Timestamp, VecN};

use crate::rng::Rng;

pub struct TruthStepConfig {
    pub transition: ConstantVelocity2d,
    pub dt: Duration,
}

pub struct GroundTruthPath {
    pub states: [GroundTruthState; 32],
    pub len: usize,
}

impl GroundTruthPath {
    pub const MAX: usize = 32;

    pub fn new(initial: GroundTruthState) -> Result<Self, Error> {
        let mut states = [GroundTruthState {
            state: [0.0; 4],
            timestamp: Timestamp::ZERO,
        }; Self::MAX];
        states[0] = initial;
        Ok(Self { states, len: 1 })
    }

    pub fn push_transition<R: Rng>(
        &mut self,
        cfg: &TruthStepConfig,
        rng: &mut R,
        noise: bool,
    ) -> Result<GroundTruthState, Error> {
        if self.len >= Self::MAX {
            return Err(Error::CapacityExceeded);
        }
        let prev = self.states[self.len - 1];
        let ts = prev.timestamp.checked_add(cfg.dt)?;
        let f = cfg.transition.transition_matrix(cfg.dt)?;
        let mean = mat_vec_mul(f, VecN(prev.state))?;
        let mut state = [0.0; 4];
        for (i, slot) in state.iter_mut().enumerate() {
            *slot = mean.get(i)?;
        }
        if noise {
            let q = cfg.transition.process_noise(cfg.dt)?;
            for block in 0..2usize {
                let i = block * 2;
                let n0 = rng.standard_normal()?;
                let n1 = rng.standard_normal()?;
                let q00 = q.get(i, i)?;
                let q01 = q.get(i, i + 1)?;
                let q11 = q.get(i + 1, i + 1)?;
                let l00 = libm::sqrt(if q00 > 0.0 { q00 } else { 0.0 });
                let l10 = if l00 > 1e-18 { q01 / l00 } else { 0.0 };
                let rem = q11 - l10 * l10;
                let l11 = libm::sqrt(if rem > 0.0 { rem } else { 0.0 });
                state[i] += l00 * n0;
                state[i + 1] += l10 * n0 + l11 * n1;
            }
        }
        let gt = GroundTruthState::new(state, ts)?;
        self.states[self.len] = gt;
        self.len += 1;
        Ok(gt)
    }

    pub fn last(&self) -> Option<GroundTruthState> {
        if self.len == 0 {
            None
        } else {
            Some(self.states[self.len - 1])
        }
    }
}
