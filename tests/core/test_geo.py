"""geo tests — great-circle (haversine) distance, scalar + vectorized."""

import numpy as np

from bike_router.core.constants import GeoConfig
from bike_router.core.geo import haversine_distance_m, haversine_vec, nearest_index


def test_haversine_vec():
    # Vectorized great-circle: 0 for coincident points, ~111 km per degree latitude, symmetric,
    # and elementwise over arrays (matching the scalar wrapper per element).
    assert haversine_vec(lat_a=48.0, lon_a=8.0, lat_b=48.0, lon_b=8.0) == 0.0
    assert (
        abs(float(haversine_vec(lat_a=0.0, lon_a=0.0, lat_b=1.0, lon_b=0.0)) - GeoConfig.METERS_PER_DEGREE_EQUATOR)
        < 300
    )
    ab = float(haversine_vec(lat_a=48.0, lon_a=8.0, lat_b=49.0, lon_b=9.0))
    ba = float(haversine_vec(lat_a=49.0, lon_a=9.0, lat_b=48.0, lon_b=8.0))
    assert abs(ab - ba) < 1e-6  # symmetric
    # elementwise over arrays, matching the scalar helper per element
    lat_a, lon_a = np.array([48.0, 49.0]), np.array([8.0, 9.0])
    lat_b, lon_b = np.array([48.1, 49.2]), np.array([8.1, 9.3])
    vectorized = haversine_vec(lat_a=lat_a, lon_a=lon_a, lat_b=lat_b, lon_b=lon_b)
    scalar = [
        haversine_distance_m(lat_a=sa, lon_a=so, lat_b=ea, lon_b=eo)
        for sa, so, ea, eo in zip(lat_a, lon_a, lat_b, lon_b, strict=True)
    ]
    np.testing.assert_allclose(vectorized, scalar)


def test_haversine_distance_m():
    # Scalar wrapper: float, non-negative, 0 for coincident, and equals haversine_vec elementwise.
    d = haversine_distance_m(lat_a=48.0, lon_a=8.0, lat_b=49.0, lon_b=9.0)
    assert isinstance(d, float) and d > 0
    assert d == float(haversine_vec(lat_a=48.0, lon_a=8.0, lat_b=49.0, lon_b=9.0))
    assert haversine_distance_m(lat_a=48.0, lon_a=8.0, lat_b=48.0, lon_b=8.0) == 0.0


def test_nearest_index():
    # Index of the closest (lats, lons) point by great-circle distance — the shared snap primitive.
    lats = np.array([48.0, 48.5, 49.0], dtype=np.float64)
    lons = np.array([8.0, 8.5, 9.0], dtype=np.float64)
    assert nearest_index(lat=48.52, lon=8.48, lats=lats, lons=lons) == 1  # closest to the middle point
    assert nearest_index(lat=48.0, lon=8.0, lats=lats, lons=lons) == 0  # exact match → its own index
    assert nearest_index(lat=60.0, lon=20.0, lats=lats, lons=lons) == 2  # far NE → the last (nearest) point


def test_haversine_quarter_and_half_circumference():
    # Great-circle sanity anchors: equator→pole is a quarter of Earth's circumference, and two
    # antipodal points are half — the clamp on the arcsin argument must not overshoot π·R.
    circumference = 2 * np.pi * GeoConfig.EARTH_RADIUS_M
    quarter = haversine_distance_m(lat_a=0.0, lon_a=0.0, lat_b=90.0, lon_b=0.0)
    assert abs(quarter - circumference / 4) < 1.0  # equator → north pole
    antipode = haversine_distance_m(lat_a=0.0, lon_a=0.0, lat_b=0.0, lon_b=180.0)
    assert abs(antipode - circumference / 2) < 1.0  # exactly half the way around
