"""Repère radar local et conversions sphériques ↔ cartésiennes."""

from __future__ import annotations

import math

import numpy as np


def wrap_angle_rad(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def spherical_to_cartesian(
    range_m: float,
    azimuth_rad: float,
    elevation_rad: float,
    radar_position_m: np.ndarray | None = None,
) -> np.ndarray:
    """Azimut = bearing atan2(y,x), élévation = asin(z/r). Retourne (x,y,z) m."""
    r = range_m
    ce = math.cos(elevation_rad)
    x = r * ce * math.cos(azimuth_rad)
    y = r * ce * math.sin(azimuth_rad)
    z = r * math.sin(elevation_rad)
    p = np.array([x, y, z], dtype=float)
    if radar_position_m is not None:
        p = p + np.asarray(radar_position_m, dtype=float)
    return p


def cartesian_to_spherical(x: float, y: float, z: float) -> tuple[float, float, float]:
    r = math.sqrt(x * x + y * y + z * z)
    if r < 1e-6:
        return 0.0, 0.0, 0.0
    elev = math.asin(max(-1.0, min(1.0, z / r)))
    az = math.atan2(y, x)
    return r, az, elev


def stonesoup_measurement_vector(
    elevation_rad: float,
    azimuth_rad: float,
    range_m: float,
) -> np.ndarray:
    """Ordre Stone Soup : [élévation, bearing, range]."""
    return np.array([[elevation_rad], [azimuth_rad], [range_m]], dtype=float)


def csv_row_to_stonesoup(range_m: float, azimuth_rad: float, elevation_rad: float) -> np.ndarray:
    return stonesoup_measurement_vector(elevation_rad, azimuth_rad, range_m)
