"""Spatial corridor ("Schlauch") generation.

Instead of a bounding-box square (which invites Overpass 429 rate-limiting on long
routes), we buffer the straight start→destination line into a tube of a fixed real
half-width each side. The buffer is isotropic in kilometres: we scale longitude by
cos(latitude) into a locally-metric frame, buffer with a km-derived radius, then
scale back — so N-S and E-W get the same real width (a plain lon/lat buffer would
be ~35% narrower E-W at 48°N). OSMnx extracts the bike graph from this polygon.
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


def corridor_within_dem(polygon: Polygon, dem_bounds: tuple[float, float, float, float]) -> bool:
    """True if the corridor's bbox lies fully inside the DEM's WGS84 bounds.

    Args:
        polygon: The corridor polygon (lon/lat degrees).
        dem_bounds: DEM (west, south, east, north) in WGS84.
    """
    assert len(dem_bounds) == 4, "dem_bounds must be (west, south, east, north)"
    west, south, east, north = dem_bounds
    assert west < east and south < north, "dem_bounds must be well-ordered"
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    return bool(min_lon >= west and max_lon <= east and min_lat >= south and max_lat <= north)
