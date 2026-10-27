from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RadarConfig:
    id: str
    position_m: list[float]
    rotation_offset_rad: list[float]


@dataclass
class MeasurementConfig:
    sigma_elevation_rad: float
    sigma_azimuth_rad: float
    sigma_range_m: float
    resolution_elevation_rad: float
    resolution_azimuth_rad: float
    resolution_range_m: float


@dataclass
class TrackingConfig:
    filter: str
    process_noise_accel_mps2: float
    probability_detection: float
    clutter_density_per_m3: float
    gate_probability: float
    init_position_sigma_m: float
    init_velocity_sigma_mps: float
    min_scans_to_confirm: int
    min_scans_to_reject: int
    confirmation_score_threshold: float
    rejection_score_threshold: float
    max_missed_scans_before_coast: int
    max_active_tracks: int


@dataclass
class HypothesisConfig:
    window_depth_scans: int
    max_hypotheses: int
    max_branches_per_event: int
    complexity_penalty: float
    common_origin_lookback_scans: int
    common_origin_sigma_multiplier: float
    separation_confirmation_scans: int
    separation_score_threshold: float


@dataclass
class PerformanceConfig:
    spatial_gate_max_range_m: float
    max_detections_per_scan: int
    log_pruned_hypotheses: bool


@dataclass
class AppConfig:
    radar: RadarConfig
    measurement: MeasurementConfig
    tracking: TrackingConfig
    hypothesis: HypothesisConfig
    performance: PerformanceConfig
    simulation: dict[str, Any] = field(default_factory=dict)
    benchmark: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> AppConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(
        radar=RadarConfig(**raw["radar"]),
        measurement=MeasurementConfig(**raw["measurement"]),
        tracking=TrackingConfig(**raw["tracking"]),
        hypothesis=HypothesisConfig(**raw["hypothesis"]),
        performance=PerformanceConfig(**raw["performance"]),
        simulation=raw.get("simulation", {}),
        benchmark=raw.get("benchmark", {}),
    )
