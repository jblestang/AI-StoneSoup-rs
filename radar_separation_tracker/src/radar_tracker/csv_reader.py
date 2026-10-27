"""Lecture CSV détections radar."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from radar_tracker.coordinates import csv_row_to_stonesoup


@dataclass(frozen=True)
class DetectionRecord:
    timestamp: datetime
    radar_id: str
    scan_id: str
    detection_id: str
    range_m: float
    azimuth_rad: float
    elevation_rad: float

    @property
    def stonesoup_vector(self) -> np.ndarray:
        return csv_row_to_stonesoup(self.range_m, self.azimuth_rad, self.elevation_rad)


@dataclass
class ScanBatch:
    timestamp: datetime
    radar_id: str
    scan_id: str
    detections: list[DetectionRecord]
    missing: bool = False  # scan manquant (placeholder)


def parse_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def read_detections_csv(path: str | Path) -> list[DetectionRecord]:
    records: list[DetectionRecord] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("missing", "").lower() in ("1", "true", "yes"):
                continue
            records.append(
                DetectionRecord(
                    timestamp=parse_timestamp(row["timestamp"]),
                    radar_id=row["radar_id"],
                    scan_id=row["scan_id"],
                    detection_id=row["detection_id"],
                    range_m=float(row["range_m"]),
                    azimuth_rad=float(row["azimuth_rad"]),
                    elevation_rad=float(row["elevation_rad"]),
                )
            )
    records.sort(key=lambda r: (r.timestamp, r.scan_id, r.detection_id))
    return records


def group_by_scan(records: list[DetectionRecord]) -> list[ScanBatch]:
    if not records:
        return []
    batches: list[ScanBatch] = []
    key = (records[0].timestamp, records[0].radar_id, records[0].scan_id)
    bucket: list[DetectionRecord] = []
    for rec in records:
        k = (rec.timestamp, rec.radar_id, rec.scan_id)
        if k != key:
            batches.append(
                ScanBatch(
                    timestamp=key[0],
                    radar_id=key[1],
                    scan_id=key[2],
                    detections=bucket,
                )
            )
            key = k
            bucket = []
        bucket.append(rec)
    batches.append(
        ScanBatch(timestamp=key[0], radar_id=key[1], scan_id=key[2], detections=bucket)
    )
    return batches


def read_scans_with_gaps(
    path: str | Path,
    expected_interval_s: float | None = None,
) -> list[ScanBatch]:
    """Regroupe les détections ; insère des scans manquants si intervalle attendu."""
    records = read_detections_csv(path)
    batches = group_by_scan(records)
    if expected_interval_s is None or len(batches) < 2:
        return batches
    from datetime import timedelta

    out: list[ScanBatch] = []
    for i, batch in enumerate(batches):
        out.append(batch)
        if i + 1 >= len(batches):
            break
        dt = (batches[i + 1].timestamp - batch.timestamp).total_seconds()
        n_missing = int(round(dt / expected_interval_s)) - 1
        for j in range(max(0, n_missing)):
            ts = batch.timestamp + timedelta(seconds=expected_interval_s * (j + 1))
            out.append(
                ScanBatch(
                    timestamp=ts,
                    radar_id=batch.radar_id,
                    scan_id=f"{batch.scan_id}_missing_{j+1}",
                    detections=[],
                    missing=True,
                )
            )
    return out
