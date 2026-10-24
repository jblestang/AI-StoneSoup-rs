use crate::error::Error;
use crate::linalg::{require_finite, VecN};
use crate::time::Timestamp;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TrackId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SensorId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundTruthState {
    /// `[x, vx, y, vy]`
    pub state: [f64; 4],
    pub timestamp: Timestamp,
}

impl GroundTruthState {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn new(state: [f64; 4], timestamp: Timestamp) -> Result<Self, Error> {
        for &v in &state {
            require_finite(v)?;
        }
        Ok(Self { state, timestamp })
    }
}

/// Cartesian position measurement `[x, y]` (linear Gaussian tutorial).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub vector: VecN<2>,
    pub timestamp: Timestamp,
    pub sensor: SensorId,
}

impl Detection {
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn new(vector: [f64; 2], timestamp: Timestamp, sensor: SensorId) -> Result<Self, Error> {
        require_finite(vector[0])?;
        require_finite(vector[1])?;
        Ok(Self {
            vector: VecN(vector),
            timestamp,
            sensor,
        })
    }
}
