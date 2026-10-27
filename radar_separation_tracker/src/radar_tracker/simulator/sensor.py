"""Simulation capteur : résolution, fusion, Pd, clutter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from radar_tracker.config import AppConfig
from radar_tracker.coordinates import cartesian_to_spherical, wrap_angle_rad
from radar_tracker.simulator.truth import ScenarioTruth, TruthTarget


@dataclass
class SimDetection:
    timestamp: datetime
    radar_id: str
    scan_id: str
    detection_id: str
    range_m: float
    azimuth_rad: float
    elevation_rad: float
    fused: bool = False
    n_merged: int = 1


def _target_state_at(target: TruthTarget, t: datetime) -> np.ndarray | None:
    for i, ts in enumerate(target.timestamps):
        if ts == t:
            return target.states[i]
    return None


def _active_targets(truth: ScenarioTruth, t: datetime) -> list[tuple[str, np.ndarray]]:
    out = []
    for tgt in truth.targets:
        if t < tgt.birth_time:
            continue
        if tgt.death_time and t > tgt.death_time:
            continue
        st = _target_state_at(tgt, t)
        if st is not None:
            out.append((tgt.target_id, st))
    return out


def _resolve_groups(
    measurements: list[tuple[str, float, float, float]],
    cfg: AppConfig,
) -> list[list[tuple[str, float, float, float]]]:
    """Cluster by resolution box in (r, az, el)."""
    if not measurements:
        return []
    mcfg = cfg.measurement
    used = [False] * len(measurements)
    groups: list[list[tuple[str, float, float, float]]] = []
    for i, (tid, r, az, el) in enumerate(measurements):
        if used[i]:
            continue
        g = [(tid, r, az, el)]
        used[i] = True
        for j in range(i + 1, len(measurements)):
            if used[j]:
                continue
            tid2, r2, az2, el2 = measurements[j]
            if (
                abs(r - r2) <= mcfg.resolution_range_m
                and abs(wrap_angle_rad(az - az2)) <= mcfg.resolution_azimuth_rad
                and abs(el - el2) <= mcfg.resolution_elevation_rad
            ):
                g.append((tid2, r2, az2, el2))
                used[j] = True
        groups.append(g)
    return groups


def fuse_group(
    group: list[tuple[str, float, float, float]],
    cfg: AppConfig,
    rng: np.random.Generator,
) -> tuple[float, float, float, bool]:
    mcfg = cfg.measurement
    rs, azs, els = [], [], []
    for _, r, az, el in group:
        rs.append(r)
        azs.append(az)
        els.append(el)
    fused = len(group) > 1
    # weighted by inverse range variance (proxy)
    w = np.ones(len(group))
    w /= w.sum()
    r_f = float(np.dot(w, rs))
    az_f = float(np.arctan2(np.sum(w * np.sin(azs)), np.sum(w * np.cos(azs))))
    el_f = float(np.dot(w, els))
    # add precision noise
    r_f += rng.normal(0, mcfg.sigma_range_m)
    az_f += rng.normal(0, mcfg.sigma_azimuth_rad)
    el_f += rng.normal(0, mcfg.sigma_elevation_rad)
    if fused:
        az_f += rng.normal(0, mcfg.resolution_azimuth_rad * 0.5)
        el_f += rng.normal(0, mcfg.resolution_elevation_rad * 0.5)
        r_f += rng.normal(0, mcfg.resolution_range_m * 0.5)
    return r_f, az_f, el_f, fused


def simulate_scan(
    truth: ScenarioTruth,
    scan_index: int,
    cfg: AppConfig,
    rng: np.random.Generator,
    scenario_flags: dict | None = None,
) -> list[SimDetection]:
    scenario_flags = scenario_flags or {}
    t = truth.start_time
    from datetime import timedelta

    timestamp = truth.start_time + timedelta(seconds=scan_index * truth.scan_interval_s)
    radar_id = cfg.radar.id
    scan_id = f"S{scan_index:05d}"
    pd = cfg.tracking.probability_detection

    raw_meas: list[tuple[str, float, float, float]] = []
    for tid, st in _active_targets(truth, timestamp):
        x, _, y, _, z, _ = st
        r, az, el = cartesian_to_spherical(x, y, z)
        if rng.random() > pd:
            continue
        if scenario_flags.get("miss_burst") and 20 <= scan_index <= 25:
            continue
        r += rng.normal(0, cfg.measurement.sigma_range_m)
        az += rng.normal(0, cfg.measurement.sigma_azimuth_rad)
        el += rng.normal(0, cfg.measurement.sigma_elevation_rad)
        raw_meas.append((tid, r, az, el))

    groups = _resolve_groups(raw_meas, cfg)
    dets: list[SimDetection] = []
    for gi, group in enumerate(groups):
        r, az, el, fused = fuse_group(group, cfg, rng)
        dets.append(
            SimDetection(
                timestamp=timestamp,
                radar_id=radar_id,
                scan_id=scan_id,
                detection_id=f"D{gi:04d}",
                range_m=r,
                azimuth_rad=az,
                elevation_rad=el,
                fused=fused,
                n_merged=len(group),
            )
        )

    # Clutter
    n_clutter = scenario_flags.get("n_clutter", 0)
    if truth.name == "dense_clutter" or scenario_flags.get("clutter"):
        n_clutter = scenario_flags.get("n_clutter", 200)
    if truth.name == "stress_1000_det":
        n_clutter = max(0, cfg.simulation.get("max_detections_per_scan", 1000) - len(dets))

    r_max = cfg.performance.spatial_gate_max_range_m
    for ci in range(n_clutter):
        r = rng.uniform(5000, r_max * 0.9)
        az = rng.uniform(-math.pi, math.pi)
        el = rng.uniform(-0.3, 0.3)
        dets.append(
            SimDetection(
                timestamp=timestamp,
                radar_id=radar_id,
                scan_id=scan_id,
                detection_id=f"C{ci:04d}",
                range_m=r,
                azimuth_rad=az,
                elevation_rad=el,
                fused=False,
            )
        )

    max_d = cfg.simulation.get("max_detections_per_scan", cfg.performance.max_detections_per_scan)
    if len(dets) > max_d:
        dets = dets[:max_d]
    return dets


def write_detections_csv(path: str, all_dets: list[SimDetection]) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp",
                "radar_id",
                "scan_id",
                "detection_id",
                "range_m",
                "azimuth_rad",
                "elevation_rad",
            ]
        )
        for d in all_dets:
            w.writerow(
                [
                    d.timestamp.isoformat(),
                    d.radar_id,
                    d.scan_id,
                    d.detection_id,
                    f"{d.range_m:.6f}",
                    f"{d.azimuth_rad:.8f}",
                    f"{d.elevation_rad:.8f}",
                ]
            )
