"""Corridor buffer geometry tests."""

from shapely.geometry import Polygon

from bike_router.corridor import build_corridor


def test_build_corridor_is_polygon_covering_endpoints():
    poly = build_corridor(start_latlon=(48.0, 8.0), dest_latlon=(48.5, 8.5))
    assert isinstance(poly, Polygon)
    # both endpoints (as lon/lat) lie inside the buffered tube
    assert poly.contains(_pt(8.0, 48.0)) and poly.contains(_pt(8.5, 48.5))


def test_build_corridor_width_scales_with_half_width():
    narrow = build_corridor(start_latlon=(48.0, 8.0), dest_latlon=(48.0, 9.0), half_width_km=5.0)
    wide = build_corridor(start_latlon=(48.0, 8.0), dest_latlon=(48.0, 9.0), half_width_km=22.0)
    assert wide.area > narrow.area


def test_corridor_isotropic_in_km():
    # a zero-length trip → a disk; its E-W and N-S spans should be ~equal in km
    # (a naive lon/lat buffer would be ~35% narrower E-W at 48°N).
    poly = build_corridor(start_latlon=(48.0, 8.0), dest_latlon=(48.0001, 8.0001), half_width_km=10.0)
    import math

    lon_span_km = (poly.bounds[2] - poly.bounds[0]) * 111.32 * math.cos(math.radians(48.0))
    lat_span_km = (poly.bounds[3] - poly.bounds[1]) * 111.32
    assert abs(lon_span_km - lat_span_km) < 1.0  # within 1 km → isotropic


def _pt(x: float, y: float) -> Polygon:
    from shapely.geometry import Point

    return Point(x, y)
