//! NumPy legacy RNG (`np.random.seed` / `RandomState`) — RandomKit algorithm, `no_std`.
//!
//! Ported from RandomKit 1.3 (BSD) for bit-compatible streams with Stone Soup tutorials.

use libm::{log, sqrt};
use stonesoup_core::Error;

const RK_STATE_LEN: usize = 624;
const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// NumPy-compatible RNG state (`rk_state`).
#[derive(Debug, Clone)]
pub struct NumpyRng {
    key: [u32; RK_STATE_LEN],
    pos: usize,
    gauss: f64,
    has_gauss: bool,
}

impl NumpyRng {
    pub fn seed(seed: u32) -> Self {
        let mut state = Self {
            key: [0; RK_STATE_LEN],
            pos: RK_STATE_LEN,
            gauss: 0.0,
            has_gauss: false,
        };
        let mut s = seed;
        for pos in 0..RK_STATE_LEN {
            state.key[pos] = s;
            s = 1812433253u32
                .wrapping_mul(s ^ (s >> 30))
                .wrapping_add(pos as u32 + 1);
        }
        state
    }

    fn rk_random(&mut self) -> u32 {
        if self.pos == RK_STATE_LEN {
            for i in 0..N - M {
                let y = (self.key[i] & UPPER_MASK) | (self.key[i + 1] & LOWER_MASK);
                self.key[i] = self.key[i + M]
                    ^ (y >> 1)
                    ^ (if y & 1 != 0 { MATRIX_A } else { 0 });
            }
            for i in N - M..N - 1 {
                let y = (self.key[i] & UPPER_MASK) | (self.key[i + 1] & LOWER_MASK);
                self.key[i] = self.key[i - (N - M)]
                    ^ (y >> 1)
                    ^ (if y & 1 != 0 { MATRIX_A } else { 0 });
            }
            let y = (self.key[N - 1] & UPPER_MASK) | (self.key[0] & LOWER_MASK);
            self.key[N - 1] = self.key[M - 1]
                ^ (y >> 1)
                ^ (if y & 1 != 0 { MATRIX_A } else { 0 });
            self.pos = 0;
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    pub fn rk_double(&mut self) -> f64 {
        let a = (self.rk_random() >> 5) as i64;
        let b = (self.rk_random() >> 6) as i64;
        (a as f64 * 67_108_864.0 + b as f64) / 9_007_199_254_740_992.0
    }

    pub fn standard_normal(&mut self) -> Result<f64, Error> {
        if self.has_gauss {
            self.has_gauss = false;
            let tmp = self.gauss;
            return Ok(tmp);
        }
        loop {
            let x1 = 2.0 * self.rk_double() - 1.0;
            let x2 = 2.0 * self.rk_double() - 1.0;
            let r2 = x1 * x1 + x2 * x2;
            if r2 >= 1.0 || r2 == 0.0 {
                continue;
            }
            let f = sqrt(-2.0 * log(r2) / r2);
            self.gauss = f * x1;
            self.has_gauss = true;
            let out = f * x2;
            if !out.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            return Ok(out);
        }
    }
}
