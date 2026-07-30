"""Corridor buffer geometry tests — the core mathematical function build_corridor.

Confirms exact geometric properties (endpoint coverage, width scaling, isotropy in km,
and line-extension past the endpoints) with concrete inputs and asserted outputs.
"""

import math

from shapely.geometry import Point, Polygon

from bike_router.core.constants import CorridorConfig, GeoConfig
from bike_router.core.corridor import build_corridor

_KM_PER_DEG = GeoConfig.METERS_PER_DEGREE_EQUATOR / 1000.0  # one source of truth (as corridor.py uses)


def _pt(x: float, y: float) -> Point:
    return Point(x, y)


def test_build_corridor_is_polygon_covering_endpoints():
    poly = build_corridor(start_latlon=(48.0, 8.0), dest_latlon=(48.5, 8.5), half_width_km=20.0, extend_km=0.0)
    assert isinstance(poly, Polygon)
    # both endpoints (as lon/lat) lie inside the buffered tube
    assert poly.contains(_pt(x=8.0, y=48.0)) and poly.contains(_pt(x=8.5, y=48.5))


def test_build_corridor_width_scales_with_half_width():
    narrow = build_corridor(start_latlon=(48.0, 8.0), dest_latlon=(48.0, 9.0), half_width_km=5.0, extend_km=0.0)
    wide = build_corridor(start_latlon=(48.0, 8.0), dest_latlon=(48.0, 9.0), half_width_km=22.0, extend_km=0.0)
    assert wide.area > narrow.area


def test_corridor_isotropic_in_km():
    # a zero-length trip → a disk; its E-W and N-S spans should be ~equal in km
    # (a naive lon/lat buffer would be ~35% narrower E-W at 48°N).
    poly = build_corridor(start_latlon=(48.0, 8.0), dest_latlon=(48.0001, 8.0001), half_width_km=10.0, extend_km=0.0)
    lon_span_km = (poly.bounds[2] - poly.bounds[0]) * _KM_PER_DEG * math.cos(math.radians(48.0))
    lat_span_km = (poly.bounds[3] - poly.bounds[1]) * _KM_PER_DEG
    assert abs(lon_span_km - lat_span_km) < 1.0  # within 1 km → isotropic


def test_extend_reaches_past_endpoints_along_line():
    # E-W trip; with extend_km the tube must reach a point extend_km beyond each endpoint.
    half, ext = 5.0, 30.0
    poly = build_corridor(start_latlon=(48.0, 8.0), dest_latlon=(48.0, 9.0), half_width_km=half, extend_km=ext)
    ext_deg = ext / (_KM_PER_DEG * math.cos(math.radians(48.0)))  # E-W degrees for ext km
    before = _pt(x=8.0 - ext_deg * 0.8, y=48.0)  # 80% of the extension before the start
    after = _pt(x=9.0 + ext_deg * 0.8, y=48.0)  # and past the end
    assert poly.contains(before) and poly.contains(after)
    # but a point well beyond the extension is NOT covered
    assert not poly.contains(_pt(x=9.0 + ext_deg * 2.0, y=48.0))


def test_extend_zero_does_not_reach_past_endpoints():
    poly = build_corridor(start_latlon=(48.0, 8.0), dest_latlon=(48.0, 9.0), half_width_km=5.0, extend_km=0.0)
    # a point just past the end (well beyond the 5 km cap radius) is not covered without extension
    beyond_deg = 20.0 / (_KM_PER_DEG * math.cos(math.radians(48.0)))
    assert not poly.contains(_pt(x=9.0 + beyond_deg, y=48.0))


def test_rail_corridor_contains_bike_corridor():
    # Same endpoints, rail params (wider + longer) → strictly larger polygon that contains the bike one.
    start, dest = (48.0, 8.0), (48.5, 8.6)
    bike = build_corridor(
        start_latlon=start,
        dest_latlon=dest,
        half_width_km=CorridorConfig.BIKE_HALF_WIDTH_KM,
        extend_km=CorridorConfig.BIKE_EXTEND_KM,
    )
    rail = build_corridor(
        start_latlon=start,
        dest_latlon=dest,
        half_width_km=CorridorConfig.RAIL_HALF_WIDTH_KM,
        extend_km=CorridorConfig.RAIL_EXTEND_KM,
    )
    assert rail.area > bike.area
    assert rail.contains(bike)
