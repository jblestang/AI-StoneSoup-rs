import math

import numpy as np

from radar_tracker.coordinates import (
    cartesian_to_spherical,
    csv_row_to_stonesoup,
    spherical_to_cartesian,
    wrap_angle_rad,
)


def test_stonesoup_order():
    v = csv_row_to_stonesoup(1000.0, 0.5, 0.1)
    assert v.ravel()[0] == 0.1  # elevation
    assert v.ravel()[1] == 0.5  # azimuth
    assert v.ravel()[2] == 1000.0


def test_roundtrip():
    x, y, z = 10000.0, 5000.0, 2000.0
    r, az, el = cartesian_to_spherical(x, y, z)
    p = spherical_to_cartesian(r, az, el)
    np.testing.assert_allclose([x, y, z], p, rtol=1e-6)


def test_azimuth_wrap():
    assert abs(wrap_angle_rad(math.pi + 0.1) - (-math.pi + 0.1)) < 1e-9
