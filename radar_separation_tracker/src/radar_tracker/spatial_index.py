"""Présélection spatiale conservatrice (grille 3D)."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from radar_tracker.coordinates import spherical_to_cartesian


class SpatialGrid:
    def __init__(self, cell_size_m: float = 500.0):
        self.cell_size = cell_size_m
        self._cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    def _key(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        cs = self.cell_size
        return (int(math.floor(x / cs)), int(math.floor(y / cs)), int(math.floor(z / cs)))

    def clear(self) -> None:
        self._cells.clear()

    def insert(self, idx: int, x: float, y: float, z: float) -> None:
        self._cells[self._key(x, y, z)].append(idx)

    def query_ball(self, x: float, y: float, z: float, radius_m: float) -> list[int]:
        cs = self.cell_size
        nr = int(math.ceil(radius_m / cs))
        cx, cy, cz = self._key(x, y, z)
        out: list[int] = []
        for dx in range(-nr, nr + 1):
            for dy in range(-nr, nr + 1):
                for dz in range(-nr, nr + 1):
                    out.extend(self._cells.get((cx + dx, cy + dy, cz + dz), []))
        return out


def detection_cartesian(range_m, az, el, radar_pos) -> np.ndarray:
    return spherical_to_cartesian(range_m, az, el, radar_pos)
