"""Corridor buffer geometry tests — the core mathematical function build_corridor.

Confirms endpoint coverage, width scaling, isotropy in km, and line-extension past the
endpoints, with concrete inputs and asserted outputs (all in one test_build_corridor).
"""

import math

from shapely.geometry import Point, Polygon

from bike_router.core.constants import CorridorConfig, GeoConfig
from bike_router.core.corridor import build_corridor

_KM_PER_DEG = GeoConfig.METERS_PER_DEGREE_EQUATOR / 1000.0  # one source of truth (as corridor.py uses)


def test_build_corridor():
    # A Polygon tube that covers both endpoints; wider half_width → larger area; isotropic in km
    # (a zero-length trip is a ~round disk, not lon-squished); extend_km reaches past the ends;
    # and rail params (wider+longer) produce a strictly larger polygon that contains the bike one.
    poly = build_corridor(start_latlon=(48.0, 8.0), dest_latlon=(48.5, 8.5), half_width_km=20.0, extend_km=0.0)
    assert isinstance(poly, Polygon)
    assert poly.contains(Point(8.0, 48.0)) and poly.contains(Point(8.5, 48.5))

    narrow = build_corridor(start_latlon=(48.0, 8.0), dest_latlon=(48.0, 9.0), half_width_km=5.0, extend_km=0.0)
    wide = build_corridor(start_latlon=(48.0, 8.0), dest_latlon=(48.0, 9.0), half_width_km=22.0, extend_km=0.0)
    assert wide.area > narrow.area

    disk = build_corridor(start_latlon=(48.0, 8.0), dest_latlon=(48.0001, 8.0001), half_width_km=10.0, extend_km=0.0)
    lon_km = (disk.bounds[2] - disk.bounds[0]) * _KM_PER_DEG * math.cos(math.radians(48.0))
    lat_km = (disk.bounds[3] - disk.bounds[1]) * _KM_PER_DEG
    assert abs(lon_km - lat_km) < 1.0  # isotropic in km

    ext = 30.0
    extended = build_corridor(start_latlon=(48.0, 8.0), dest_latlon=(48.0, 9.0), half_width_km=5.0, extend_km=ext)
    ext_deg = ext / (_KM_PER_DEG * math.cos(math.radians(48.0)))
    assert extended.contains(Point(8.0 - ext_deg * 0.8, 48.0)) and extended.contains(Point(9.0 + ext_deg * 0.8, 48.0))
    assert not extended.contains(Point(9.0 + ext_deg * 2.0, 48.0))  # not beyond the extension
    assert not narrow.contains(Point(9.0 + 20.0 / (_KM_PER_DEG * math.cos(math.radians(48.0))), 48.0))  # extend=0

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
    assert rail.area > bike.area and rail.contains(bike)
