"""Great-circle (Haversine) distance tests."""

import numpy as np

from bike_router.core.constants import GeoConfig
from bike_router.core.geo import haversine_distance_m, haversine_vec


def test_zero_distance():
    assert haversine_distance_m(lat_a=48.0, lon_a=8.0, lat_b=48.0, lon_b=8.0) == 0.0


def test_one_degree_latitude_is_about_111km():
    distance = haversine_distance_m(lat_a=0.0, lon_a=0.0, lat_b=1.0, lon_b=0.0)
    assert abs(distance - GeoConfig.METERS_PER_DEGREE_EQUATOR) < 300


def test_vectorized_matches_scalar():
    lat_a = np.array([48.0, 49.0])
    lon_a = np.array([8.0, 9.0])
    lat_b = np.array([48.1, 49.2])
    lon_b = np.array([8.1, 9.3])
    vectorized = haversine_vec(lat_a=lat_a, lon_a=lon_a, lat_b=lat_b, lon_b=lon_b)
    scalar = [
        haversine_distance_m(lat_a=start_lat, lon_a=start_lon, lat_b=end_lat, lon_b=end_lon)
        for start_lat, start_lon, end_lat, end_lon in zip(lat_a, lon_a, lat_b, lon_b, strict=True)
    ]
    np.testing.assert_allclose(vectorized, scalar)


def test_symmetry():
    forward = haversine_distance_m(lat_a=48.0, lon_a=8.0, lat_b=49.0, lon_b=9.0)
    backward = haversine_distance_m(lat_a=49.0, lon_a=9.0, lat_b=48.0, lon_b=8.0)
    assert abs(forward - backward) < 1e-6
