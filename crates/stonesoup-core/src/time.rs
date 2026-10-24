use crate::error::Error;

/// Relative time in nanoseconds (non-negative for normal simulation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Timestamp(pub i64);

impl Timestamp {
    pub const ZERO: Self = Self(0);

    pub const fn from_nanos(n: i64) -> Self {
        Self(n)
    }

    pub const fn nanos(self) -> i64 {
        self.0
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn checked_add(self, dt: Duration) -> Result<Self, Error> {
        let sum = self.0.checked_add(dt.0).ok_or(Error::InvalidTimeStep)?;
        if sum < 0 {
            return Err(Error::InvalidTimeStep);
        }
        Ok(Self(sum))
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn checked_sub(self, other: Self) -> Result<Duration, Error> {
        let d = self.0.checked_sub(other.0).ok_or(Error::InvalidTimeStep)?;
        Ok(Duration(d))
    }
}

/// Signed duration in nanoseconds (may be negative for ordering checks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Duration(pub i64);

impl Duration {
    pub const ZERO: Self = Self(0);
    pub const SEC: Self = Self(1_000_000_000);

    pub const fn from_nanos(n: i64) -> Self {
        Self(n)
    }

    pub const fn from_seconds(seconds: f64) -> Result<Self, Error> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(Error::InvalidTimeStep);
        }
        let nanos = seconds * 1e9;
        if nanos > i64::MAX as f64 {
            return Err(Error::InvalidTimeStep);
        }
        Ok(Self(nanos as i64))
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn as_seconds(self) -> f64 {
        (self.0 as f64) / 1e9
    }
}
