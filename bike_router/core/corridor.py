"""Spatial corridor ("Schlauch") generation — buffer the start→dest line into a tube.

Isotropic in km: scale lon by cos(lat) into a metric frame, buffer, scale back (equal N-S/E-W width).
One agnostic function serves both the tight bike tube and the wide rail tube; only the widths differ.
"""

import math

from shapely.affinity import scale
from shapely.geometry import LineString, Polygon

from bike_router.core.constants import GeoConfig


def build_corridor(
    *,
    start_latlon: tuple[float, float],
    dest_latlon: tuple[float, float],
    half_width_km: float,
    extend_km: float,
) -> Polygon:
    """Buffer the start→dest line into a corridor polygon (EPSG:4326 degrees).

    Args:
        start_latlon: (lat, lon) of the start.
        dest_latlon: (lat, lon) of the destination.
        half_width_km: search half-width each side of the direct line, in km.
        extend_km: extend the line this far past each endpoint before buffering.

    Returns:
        A shapely Polygon in lon/lat (x=lon, y=lat) degrees.
    """
    assert half_width_km > 0, "half_width_km must be positive"
    assert extend_km >= 0, "extend_km must be non-negative"
    start_lat, start_lon = start_latlon
    dest_lat, dest_lon = dest_latlon

    km_per_deg = GeoConfig.METERS_PER_DEGREE_EQUATOR / 1000.0
    mid_lat_rad = math.radians((start_lat + dest_lat) / 2.0)
    lon_scale = math.cos(mid_lat_rad)  # 1° lon is this fraction of 1° lat here

    # Work in a frame where 1 unit ≈ 1° lat in both axes: pre-scale lon by cos(lat).
    ax, ay = start_lon * lon_scale, start_lat
    bx, by = dest_lon * lon_scale, dest_lat
    # Extend the segment past both endpoints along its own direction. Coincident endpoints are an
    # upstream invariant violation (plan_route rejects trips < MIN_TRIP_KM), so let length==0 raise.
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    assert length > 0, f"coincident start/dest → degenerate corridor ({length=}); upstream must reject short trips"
    ux, uy = dx / length, dy / length
    ext_deg = extend_km / km_per_deg
    ax, ay = ax - ux * ext_deg, ay - uy * ext_deg
    bx, by = bx + ux * ext_deg, by + uy * ext_deg

    line = LineString([(ax, ay), (bx, by)])
    radius_deg = half_width_km / km_per_deg
    buffered = line.buffer(radius_deg)
    # Undo the longitude scaling to return a real lon/lat polygon.
    corridor: Polygon = scale(buffered, xfact=1.0 / lon_scale, yfact=1.0, origin=(0.0, 0.0))
    assert not corridor.is_empty, "corridor polygon must not be empty"
    return corridor
