//! Range-dependent track–truth scoring (bearing–range Mahalanobis vs planar fallback).

use crate::error::Error;
use crate::types::mod_bearing;

use super::track_to_truth::position_distance;

/// How to score filter vs truth position for TrackToTruth and metrics.
#[derive(Debug, Clone, Copy)]
pub enum TrackTruthPositionGate {
    /// Planar distance must be below a fixed radius (m).
    FixedRadius(f64),
    /// Sensor bearing–range with diagonal R; score is squared Mahalanobis, gate is χ² limit.
    BearingRangeChi2 {
        sensor_x: f64,
        sensor_y: f64,
        bearing_sigma_rad: f64,
        range_sigma_m: f64,
        /// χ² limit to **acquire** a truth label (strict).
        chi2_limit: f64,
        /// χ² limit to **retain** an established label through filter coast (≥ ``chi2_limit``).
        chi2_sustain_limit: f64,
    },
}

impl TrackTruthPositionGate {
    /// Combined 1σ position uncertainty (m) at truth horizontal position (for diagnostics).
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn combined_sigma_m(&self, truth_x: f64, truth_y: f64) -> f64 {
        match self {
            Self::FixedRadius(r) => *r,
            Self::BearingRangeChi2 {
                sensor_x,
                sensor_y,
                bearing_sigma_rad,
                range_sigma_m,
                ..
            } => {
                let dx = truth_x - sensor_x;
                let dy = truth_y - sensor_y;
                let r = libm::sqrt(dx * dx + dy * dy);
                let s_cr = r * bearing_sigma_rad;
                libm::sqrt(s_cr * s_cr + range_sigma_m * range_sigma_m)
            }
        }
    }

    /// Squared Mahalanobis (BR) or planar distance; `None` if non-finite.
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn raw_score(&self, track: [f64; 4], truth: [f64; 4]) -> Result<Option<f64>, Error> {
        match self {
            Self::FixedRadius(_) => {
                let d = position_distance(track, truth);
                Ok(Some(d))
            }
            Self::BearingRangeChi2 {
                sensor_x,
                sensor_y,
                bearing_sigma_rad,
                range_sigma_m,
                ..
            } => {
                let sq = bearing_range_mahalanobis_sq(
                    track,
                    truth,
                    *sensor_x,
                    *sensor_y,
                    *bearing_sigma_rad,
                    *range_sigma_m,
                )?;
                Ok(Some(sq))
            }
        }
    }

    /// Lower is better. Returns `None` if outside the **acquire** gate.
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn score(
        &self,
        track: [f64; 4],
        truth: [f64; 4],
    ) -> Result<Option<f64>, Error> {
        match self {
            Self::FixedRadius(r) => {
                let d = position_distance(track, truth);
                Ok(if d < *r { Some(d) } else { None })
            }
            Self::BearingRangeChi2 {
                chi2_limit,
                ..
            } => {
                let Some(sq) = self.raw_score(track, truth)? else {
                    return Ok(None);
                };
                Ok(if sq <= *chi2_limit { Some(sq) } else { None })
            }
        }
    }

    /// Retain an existing truth label through inter-visit filter coast (wider BR gate).
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn score_sustain(
        &self,
        track: [f64; 4],
        truth: [f64; 4],
    ) -> Result<Option<f64>, Error> {
        match self {
            Self::FixedRadius(r) => {
                let d = position_distance(track, truth);
                Ok(if d < *r * 1.5 { Some(d) } else { None })
            }
            Self::BearingRangeChi2 {
                chi2_sustain_limit,
                ..
            } => {
                let Some(sq) = self.raw_score(track, truth)? else {
                    return Ok(None);
                };
                Ok(if sq <= *chi2_sustain_limit {
                    Some(sq)
                } else {
                    None
                })
            }
        }
    }
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
fn bearing_range_mahalanobis_sq(
    track: [f64; 4],
    truth: [f64; 4],
    sensor_x: f64,
    sensor_y: f64,
    bearing_sigma_rad: f64,
    range_sigma_m: f64,
) -> Result<f64, Error> {
    let dx_t = track[0] - sensor_x;
    let dy_t = track[2] - sensor_y;
    let dx_g = truth[0] - sensor_x;
    let dy_g = truth[2] - sensor_y;
    let r_t = libm::sqrt(dx_t * dx_t + dy_t * dy_t);
    let r_g = libm::sqrt(dx_g * dx_g + dy_g * dy_g);
    let b_t = libm::atan2(dy_t, dx_t);
    let b_g = libm::atan2(dy_g, dx_g);
    let db = mod_bearing(b_t - b_g)?;
    let dr = r_t - r_g;
    let nb = db / bearing_sigma_rad;
    let nr = dr / range_sigma_m;
    let sq = nb * nb + nr * nr;
    crate::linalg::require_finite(sq)?;
    Ok(sq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PDA_GATE_CHI2_2D;

    #[test]
    fn bearing_range_gate_tightens_at_short_range() {
        let gate = TrackTruthPositionGate::BearingRangeChi2 {
            sensor_x: 0.0,
            sensor_y: 0.0,
            bearing_sigma_rad: 0.2_f64.to_radians(),
            range_sigma_m: 30.0,
            chi2_limit: PDA_GATE_CHI2_2D,
            chi2_sustain_limit: PDA_GATE_CHI2_2D * 1.5,
        };
        let track = [50_000.0, 0.0, 0.0, 0.0];
        let near = [50_050.0, 0.0, 0.0, 0.0];
        assert!(gate.score(track, near).unwrap().is_some());
        assert!(gate.combined_sigma_m(50_000.0, 0.0) < gate.combined_sigma_m(350_000.0, 0.0));
        let far_offset = [350_000.0, 5_000.0, 0.0, 0.0];
        assert!(gate.score(track, far_offset).unwrap().is_none());
    }
}
