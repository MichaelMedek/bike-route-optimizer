"""Spatial corridor ("Schlauch") generation.

Buffers the start→dest line into a tube (avoids bbox Overpass 429s). Isotropic in
km: scale lon by cos(lat) into a metric frame, buffer, scale back — so N-S and E-W
get equal real width. OSMnx extracts the bike graph from this polygon.
"""

import math

from shapely.affinity import scale
from shapely.geometry import LineString, Polygon

from bike_router.constants import CorridorConfig, GeoConfig


def build_corridor(
    start_latlon: tuple[float, float],
    dest_latlon: tuple[float, float],
    half_width_km: float = CorridorConfig.HALF_WIDTH_KM,
) -> Polygon:
    """Buffer the start→dest line into a corridor polygon (EPSG:4326 degrees).

    Args:
        start_latlon: (lat, lon) of the start.
        dest_latlon: (lat, lon) of the destination.
        half_width_km: search half-width each side of the direct line, in km.

    Returns:
        A shapely Polygon in lon/lat (x=lon, y=lat) degrees.
    """
    assert half_width_km > 0, "half_width_km must be positive"
    start_lat, start_lon = start_latlon
    dest_lat, dest_lon = dest_latlon

    km_per_deg = GeoConfig.METERS_PER_DEGREE_EQUATOR / 1000.0
    mid_lat_rad = math.radians((start_lat + dest_lat) / 2.0)
    lon_scale = math.cos(mid_lat_rad)  # 1° lon is this fraction of 1° lat here

    # Work in a frame where 1 unit ≈ 1° lat in both axes: pre-scale lon by cos(lat).
    line = LineString([(start_lon * lon_scale, start_lat), (dest_lon * lon_scale, dest_lat)])
    radius_deg = half_width_km / km_per_deg
    buffered = line.buffer(radius_deg)
    # Undo the longitude scaling to return a real lon/lat polygon.
    corridor: Polygon = scale(buffered, xfact=1.0 / lon_scale, yfact=1.0, origin=(0.0, 0.0))
    assert not corridor.is_empty, "corridor polygon must not be empty"
    return corridor


def corridor_within_bbox(polygon: Polygon, bbox: tuple[float, float, float, float]) -> bool:
    """True if the corridor's bbox lies fully inside ``bbox`` (west, south, east, north).

    Args:
        polygon: The corridor polygon (lon/lat degrees).
        bbox: Coverage (west, south, east, north) in WGS84.
    """
    assert len(bbox) == 4, "bbox must be (west, south, east, north)"
    west, south, east, north = bbox
    assert west < east and south < north, "bbox must be well-ordered"
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    return bool(min_lon >= west and max_lon <= east and min_lat >= south and max_lat <= north)
