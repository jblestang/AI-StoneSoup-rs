use stonesoup_core::{Detection, SensorId, Timestamp};

use crate::registry::ScenarioError;

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn push_measurements<const MAX: usize>(
    meas_offsets: &[usize],
    meas_xy: &[[f64; 2]],
    step: usize,
    ts: Timestamp,
    out: &mut [Detection; MAX],
) -> Result<usize, ScenarioError> {
    let start = meas_offsets
        .get(step)
        .copied()
        .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
    let end = meas_offsets
        .get(step + 1)
        .copied()
        .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
    let count = end - start;
    if count > MAX {
        return Err(ScenarioError::Core(stonesoup_core::Error::CapacityExceeded));
    }
    for (j, idx) in (start..end).enumerate() {
        let xy = meas_xy
            .get(idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        out[j] = Detection::new(xy, ts, SensorId(0)).map_err(ScenarioError::Core)?;
    }
    Ok(count)
}
