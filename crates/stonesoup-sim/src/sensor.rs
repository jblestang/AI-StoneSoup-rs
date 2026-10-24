use stonesoup_core::linalg::mat_vec_mul;
use stonesoup_core::{
    Detection, Error, GroundTruthState, LinearGaussianMeasurement, SensorId, VecN,
};

use crate::rng::Rng;

pub struct LinearPositionSensor {
    pub model: LinearGaussianMeasurement,
    pub sensor_id: SensorId,
}

pub struct SensorSim;

impl SensorSim {
    pub fn measure<R: Rng>(
        sensor: &LinearPositionSensor,
        truth: &GroundTruthState,
        rng: &mut R,
        noise: bool,
    ) -> Result<Detection, Error> {
        let z = mat_vec_mul(sensor.model.matrix_h()?, VecN(truth.state))?;
        let mut out = [0.0, 0.0];
        out[0] = z.get(0)?;
        out[1] = z.get(1)?;
        if noise {
            let r = sensor.model.r();
            let n0 = rng.standard_normal()?;
            let n1 = rng.standard_normal()?;
            let r00 = r.get(0, 0)?;
            let r10 = r.get(1, 0)?;
            let r11 = r.get(1, 1)?;
            let l00 = libm::sqrt(if r00 > 0.0 { r00 } else { 0.0 });
            let l10 = if l00 > 1e-18 { r10 / l00 } else { 0.0 };
            let rem = r11 - l10 * l10;
            let l11 = libm::sqrt(if rem > 0.0 { rem } else { 0.0 });
            out[0] += l00 * n0;
            out[1] += l10 * n0 + l11 * n1;
        }
        Detection::new(out, truth.timestamp, sensor.sensor_id)
    }
}
