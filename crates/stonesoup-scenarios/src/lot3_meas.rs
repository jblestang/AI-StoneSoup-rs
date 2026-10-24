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

/// Fill viewer detection buffers for a completed scenario step (Lot 3 tutorials 05–08).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn fill_step_detection_view(
    meas_offsets: &[usize],
    meas_xy: &[[f64; 2]],
    meas_is_true: &[u8],
    step: usize,
    det_xy: &mut [[f64; 2]],
    det_is_true: &mut [bool],
    max_dets: usize,
) -> Result<usize, ScenarioError> {
    let start = meas_offsets
        .get(step)
        .copied()
        .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
    let end = meas_offsets
        .get(step + 1)
        .copied()
        .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
    let count = (end - start).min(max_dets);
    for j in 0..count {
        let idx = start + j;
        let xy = meas_xy
            .get(idx)
            .copied()
            .ok_or(ScenarioError::Core(stonesoup_core::Error::InvalidDimension))?;
        det_xy[j] = [xy[0], xy[1]];
        det_is_true[j] = meas_is_true.get(idx).copied().unwrap_or(0) != 0;
    }
    Ok(count)
}
