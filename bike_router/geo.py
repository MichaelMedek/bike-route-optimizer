"""Geodesic helpers — great-circle (Haversine) distance.

Refined from ski-resort-designer's `core/geo_calculator.py` (scalar + vectorized
Haversine), dropping unused bearing code. Coords are WGS84 decimal degrees,
distances metres, Earth a sphere of radius 6,371 km. Used by the A* heuristic.
"""

import numpy as np
import numpy.typing as npt

from bike_router.constants import GeoConfig


def haversine_vec(
    lat_a: "npt.NDArray[np.float64] | float",
    lon_a: "npt.NDArray[np.float64] | float",
    lat_b: "npt.NDArray[np.float64] | float",
    lon_b: "npt.NDArray[np.float64] | float",
) -> "npt.NDArray[np.float64]":
    """Vectorized great-circle distance in metres (elementwise over broadcastable arrays)."""
    assert np.all(np.abs(lat_a) <= 90.0) and np.all(np.abs(lat_b) <= 90.0), "latitude out of range"
    assert np.all(np.abs(lon_a) <= 180.0) and np.all(np.abs(lon_b) <= 180.0), "longitude out of range"
    delta_lat = np.radians(lat_b - lat_a)
    delta_lon = np.radians(lon_b - lon_a)
    chord = (
        np.sin(delta_lat / 2) ** 2 + np.cos(np.radians(lat_a)) * np.cos(np.radians(lat_b)) * np.sin(delta_lon / 2) ** 2
    )
    return GeoConfig.EARTH_RADIUS_M * 2 * np.arctan2(np.sqrt(chord), np.sqrt(1 - chord))  # type: ignore[no-any-return]


def haversine_distance_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """Scalar great-circle distance between two lat/lon points, in metres."""
    assert -90.0 <= lat_a <= 90.0, "lat_a out of range"
    assert -90.0 <= lat_b <= 90.0, "lat_b out of range"
    assert -180.0 <= lon_a <= 180.0, "lon_a out of range"
    assert -180.0 <= lon_b <= 180.0, "lon_b out of range"
    distance = float(haversine_vec(lat_a=lat_a, lon_a=lon_a, lat_b=lat_b, lon_b=lon_b))
    assert distance >= 0, "distance must be non-negative"
    return distance
