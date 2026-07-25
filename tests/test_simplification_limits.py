"""Geometry tests: full-route LineString + Visvalingam waypoint selection."""

import numpy as np
from shapely.geometry import LineString

from bike_router.constants import GmapsConfig
from bike_router.simplify import _interpolate_to_n, _triangle_area, _visvalingam, route_to_linestring, select_waypoints
from tests.conftest import make_line_graph


def _zigzag_line(n_points: int = 200) -> LineString:
    lons = np.linspace(0.0, 1.0, n_points)
    lats = 0.001 * (np.arange(n_points) % 2)  # up/down zigzag
    return LineString(list(zip(lons.tolist(), lats.tolist(), strict=True)))


# --- select_waypoints (Google Maps) -----------------------------------------


def test_select_waypoints_exactly_n():
    points = select_waypoints(line=_zigzag_line(n_points=200), count=GmapsConfig.N_WAYPOINTS)
    assert len(points) == 10
    assert all(len(point) == 2 for point in points)  # (lat, lon)


def test_select_waypoints_various_sizes_hit_n():
    for n_points in (11, 50, 200, 500):
        assert len(select_waypoints(line=_zigzag_line(n_points=n_points), count=10)) == 10


def test_select_waypoints_endpoints_preserved():
    line = _zigzag_line(n_points=200)
    coords = list(line.coords)  # (lon, lat)
    points = select_waypoints(line=line, count=10)
    first_lat, first_lon = points[0]
    last_lat, last_lon = points[-1]
    assert (first_lon, first_lat) == coords[0]
    assert (last_lon, last_lat) == coords[-1]


def test_select_waypoints_short_line_padded():
    line = LineString([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    points = select_waypoints(line=line, count=10)
    assert len(points) == 10
    assert points[0] == (0.0, 0.0)
    assert points[-1] == (0.0, 2.0)  # (lat, lon)


def test_select_waypoints_keeps_significant_corner():
    line = LineString([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    points = select_waypoints(line=line, count=3)
    assert (0.0, 1.0) in points  # the corner (lat=0, lon=1)


# --- Visvalingam / interpolation internals -----------------------------------


def test_triangle_area_known_values():
    assert _triangle_area(point_a=(0.0, 0.0), point_b=(2.0, 0.0), point_c=(0.0, 3.0)) == 3.0
    assert _triangle_area(point_a=(0.0, 0.0), point_b=(1.0, 0.0), point_c=(2.0, 0.0)) == 0.0


def test_visvalingam_drops_least_significant_first():
    coords = [(0.0, 0.0), (1.0, 0.001), (2.0, 0.0), (3.0, 5.0), (4.0, 0.0)]
    out = _visvalingam(coords=coords, count=4)
    assert len(out) == 4
    assert (1.0, 0.001) not in out  # smallest-area interior point removed
    assert (3.0, 5.0) in out  # significant spike survives
    assert out[0] == (0.0, 0.0) and out[-1] == (4.0, 0.0)


def test_interpolate_to_n_evenly_spaced():
    out = _interpolate_to_n(coords=[(0.0, 0.0), (4.0, 0.0)], count=5)
    assert out == [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]


def test_interpolate_to_n_single_point():
    assert _interpolate_to_n(coords=[(2.0, 3.0)], count=3) == [(2.0, 3.0)] * 3


# --- route_to_linestring -----------------------------------------------------


def test_route_to_linestring_follows_nodes():
    graph = make_line_graph()
    line = route_to_linestring(graph=graph, node_path=[1, 2, 3])
    coords = list(line.coords)
    assert coords[0] == (8.0, 48.0)
    assert coords[-1] == (8.02, 48.0)
