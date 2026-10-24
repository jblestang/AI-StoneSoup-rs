use stonesoup_core::Error;

/// Deterministic PRNG (SplitMix64). Not bit-compatible with NumPy MT19937.
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Standard normal via Box–Muller (uses two uniforms per call).
    pub fn standard_normal(&mut self) -> Result<f64, Error> {
        let u1 = (self.next_u64() as f64 + 1.0) / (u64::MAX as f64 + 1.0);
        let u2 = (self.next_u64() as f64 + 1.0) / (u64::MAX as f64 + 1.0);
        if u1 <= 0.0 {
            return Err(Error::NumericalFailure);
        }
        let r = libm::sqrt(-2.0 * libm::log(u1));
        let theta = 2.0 * core::f64::consts::PI * u2;
        let z = r * libm::cos(theta);
        if !z.is_finite() {
            return Err(Error::NumericalFailure);
        }
        Ok(z)
    }
}

pub trait Rng {
    fn standard_normal(&mut self) -> Result<f64, Error>;
}

impl Rng for SplitMix64 {
    fn standard_normal(&mut self) -> Result<f64, Error> {
        SplitMix64::standard_normal(self)
    }
}
