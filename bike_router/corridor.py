"""Spatial corridor ("Schlauch") generation.

Instead of a bounding-box square (which invites Overpass 429 rate-limiting on
long routes), we buffer the straight start→destination line into a ~40 km-wide
tube that hugs the travel angle. OSMnx then extracts the bike graph from this
polygon.
"""

from shapely.geometry import LineString, Polygon

from bike_router.constants import CorridorConfig


def build_corridor(
    start_latlon: tuple[float, float],
    dest_latlon: tuple[float, float],
    buffer_deg: float = CorridorConfig.BUFFER_DEG,
) -> Polygon:
    """Buffer the start→dest line into a corridor polygon (EPSG:4326 degrees).

    Args:
        start_latlon: (lat, lon) of the start.
        dest_latlon: (lat, lon) of the destination.
        buffer_deg: Buffer radius in degrees (~0.2 ≈ 22 km each side).

    Returns:
        A shapely Polygon in lon/lat (x=lon, y=lat) degrees.
    """
    assert buffer_deg > 0, "buffer_deg must be positive"
    start_lat, start_lon = start_latlon
    dest_lat, dest_lon = dest_latlon
    # shapely uses (x=lon, y=lat) ordering — swap from the (lat, lon) inputs.
    line = LineString([(start_lon, start_lat), (dest_lon, dest_lat)])
    corridor: Polygon = line.buffer(buffer_deg)
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
