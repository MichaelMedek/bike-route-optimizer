"""Geometry tests: full-route LineString + Visvalingam waypoint selection."""

import numpy as np
from shapely.geometry import LineString

from bike_router.constants import GmapsConfig, Mode
from bike_router.simplify import (
    BikeLeg,
    RailLeg,
    Station,
    _interpolate_to_n,
    _thin_close_points,
    _triangle_area,
    _visvalingam,
    bike_leg_endpoints,
    format_bike_legs,
    format_rail_legs,
    route_to_linestring,
    select_waypoints,
    split_bike_legs,
    split_rail_legs,
)
from tests.conftest import make_line_graph, make_mixed_mode_graph


def _zigzag_line(n_points: int = 200) -> LineString:
    lons = np.linspace(0.0, 1.0, n_points)
    lats = 0.001 * (np.arange(n_points) % 2)  # up/down zigzag
    return LineString(list(zip(lons.tolist(), lats.tolist(), strict=True)))


# --- select_waypoints (Google Maps) -----------------------------------------


def test_select_waypoints_exactly_n():
    points = select_waypoints(line=_zigzag_line(n_points=200), count=GmapsConfig.N_WAYPOINTS)
    assert len(points) == GmapsConfig.N_WAYPOINTS
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


def test_select_waypoints_thins_over_close_interior_points():
    # A tiny ~40 m leg near (48, 8): every interior point is well under the 1 km min spacing,
    # so only origin + destination survive (no clutter of near-identical waypoints).
    line = LineString([(8.0000 + i * 0.0001, 48.0) for i in range(6)])  # ~7 m steps, ~37 m total
    points = select_waypoints(line=line, count=10)
    assert len(points) == 2  # origin + destination only
    assert points[0] == (48.0, 8.0)
    assert points[-1] == (48.0, 8.0005)


def test_thin_close_points_keeps_far_apart_and_endpoints():
    # Points ~1.1 km apart (0.01° lon ≈ 743 m at 48°N; use 0.02° ≈ 1.5 km) all survive.
    pts = [(48.0, 8.0), (48.0, 8.02), (48.0, 8.04)]
    assert _thin_close_points(points=pts, min_spacing_km=1.0) == pts
    # A near-duplicate middle point is dropped; endpoints always kept.
    pts2 = [(48.0, 8.0), (48.0, 8.00001), (48.0, 8.04)]
    assert _thin_close_points(points=pts2, min_spacing_km=1.0) == [(48.0, 8.0), (48.0, 8.04)]


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
    # Deterministic: the smallest-area interior point (1.0, 0.001) is dropped, rest kept in order.
    assert out == [(0.0, 0.0), (2.0, 0.0), (3.0, 5.0), (4.0, 0.0)]


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


# --- split_bike_legs ---------------------------------------------------------


def test_split_bike_legs_pure_bike_is_one_leg():
    graph = make_mixed_mode_graph([(1, 2, Mode.BIKE), (2, 3, Mode.BIKE)])
    assert split_bike_legs(graph=graph, node_path=[1, 2, 3]) == [[1, 2, 3]]


def test_split_bike_legs_train_in_middle_yields_two_legs():
    # bike → station → rail → station → bike : two pedalled legs around the train ride.
    graph = make_mixed_mode_graph(
        [
            (1, 2, Mode.BIKE),
            (2, 3, Mode.STATION),
            (3, 4, Mode.RAIL),
            (4, 5, Mode.STATION),
            (5, 6, Mode.BIKE),
        ]
    )
    assert split_bike_legs(graph=graph, node_path=[1, 2, 3, 4, 5, 6]) == [[1, 2], [5, 6]]


def test_split_bike_legs_all_rail_yields_no_bike_leg():
    graph = make_mixed_mode_graph([(1, 2, Mode.STATION), (2, 3, Mode.RAIL), (3, 4, Mode.STATION)])
    assert split_bike_legs(graph=graph, node_path=[1, 2, 3, 4]) == []


# --- split_rail_legs / format_rail_legs --------------------------------------


def _names(legs) -> list[tuple[str | None, str | None]]:  # noqa: ANN001 — list[RailLeg]
    """(board name, alight name) per leg — the Station→name projection the assertions want."""
    return [(leg.board.name, leg.alight.name) for leg in legs]


def _rail_leg(board: str | None, alight: str | None) -> RailLeg:
    """A RailLeg from two station names (positions irrelevant for the format/endpoint tests)."""
    return RailLeg(
        board=Station(name=board, lat=48.0, lon=8.0, elevation_m=0.0),
        alight=Station(name=alight, lat=48.0, lon=8.1, elevation_m=0.0),
    )


def test_split_rail_legs_pure_bike_has_no_train():
    graph = make_mixed_mode_graph([(1, 2, Mode.BIKE), (2, 3, Mode.BIKE)])
    assert split_rail_legs(graph=graph, node_path=[1, 2, 3]) == []


def test_split_rail_legs_one_ride_names_board_and_alight():
    # bike → station → rail(3→4) → station → bike : one train ride, board 3 / alight 4.
    graph = make_mixed_mode_graph(
        [
            (1, 2, Mode.BIKE),
            (2, 3, Mode.STATION),
            (3, 4, Mode.RAIL),
            (4, 5, Mode.STATION),
            (5, 6, Mode.BIKE),
        ]
    )
    legs = split_rail_legs(graph=graph, node_path=[1, 2, 3, 4, 5, 6])
    assert _names(legs) == [("Station 3", "Station 4")]


def test_split_rail_legs_change_stays_one_ride():
    # Consecutive rail edges (an on-train change at a junction) are ONE ride: board first,
    # alight last rail node — NOT split at the intermediate station.
    graph = make_mixed_mode_graph([(1, 2, Mode.STATION), (2, 3, Mode.RAIL), (3, 4, Mode.RAIL), (4, 5, Mode.STATION)])
    legs = split_rail_legs(graph=graph, node_path=[1, 2, 3, 4, 5])
    assert _names(legs) == [("Station 2", "Station 4")]


def test_split_rail_legs_two_separate_rides():
    # bike → rail → bike → rail → bike : two distinct train rides (why 3 pedalled legs appear).
    graph = make_mixed_mode_graph(
        [
            (1, 2, Mode.STATION),
            (2, 3, Mode.RAIL),
            (3, 4, Mode.STATION),
            (4, 5, Mode.BIKE),
            (5, 6, Mode.STATION),
            (6, 7, Mode.RAIL),
            (7, 8, Mode.STATION),
        ]
    )
    legs = split_rail_legs(graph=graph, node_path=[1, 2, 3, 4, 5, 6, 7, 8])
    assert _names(legs) == [("Station 2", "Station 3"), ("Station 6", "Station 7")]


def test_split_rail_legs_ride_at_end_of_path():
    # Route whose LAST edge is rail (no closing station hop) → the ride still emits a leg.
    graph = make_mixed_mode_graph([(1, 2, Mode.STATION), (2, 3, Mode.RAIL)])
    legs = split_rail_legs(graph=graph, node_path=[1, 2, 3])
    assert _names(legs) == [("Station 2", "Station 3")]


def test_format_rail_legs_numbers_lines_and_handles_unnamed():
    legs = [_rail_leg(board="Freudenstadt", alight="Pforzheim"), _rail_leg(board=None, alight="Karlsruhe")]
    assert format_rail_legs(rail_legs=legs) == [
        "Train 1: Freudenstadt → Pforzheim",
        "Train 2: (unnamed stop) → Karlsruhe",
    ]


# --- bike_leg_endpoints / format_bike_legs -----------------------------------


def test_bike_leg_endpoints_pure_bike_is_origin_to_destination():
    # No train → one leg spanning the whole trip: origin → destination.
    graph = make_mixed_mode_graph([(1, 2, Mode.BIKE), (2, 3, Mode.BIKE)])
    ends = bike_leg_endpoints(
        graph=graph, node_path=[1, 2, 3], leg_paths=[[1, 2, 3]], origin="Horb", destination="Freudenstadt"
    )
    assert ends == [("Horb", "Freudenstadt")]


def test_bike_leg_endpoints_one_train_uses_station_names_at_inner_ends():
    # bike → station → rail → station → bike: leg 0 = origin → board station, leg 1 = alight → dest.
    graph = make_mixed_mode_graph(
        [(1, 2, Mode.BIKE), (2, 3, Mode.STATION), (3, 4, Mode.RAIL), (4, 5, Mode.STATION), (5, 6, Mode.BIKE)]
    )
    ends = bike_leg_endpoints(
        graph=graph,
        node_path=[1, 2, 3, 4, 5, 6],
        leg_paths=[[1, 2], [5, 6]],
        origin="Horb am Neckar",
        destination="Freudenstadt",
    )
    assert ends == [("Horb am Neckar", "Station 3"), ("Station 4", "Freudenstadt")]


def test_bike_leg_endpoints_route_ending_on_a_train():
    # Regression: a route that ENDS on the train has ONE bike leg but ONE train ride (not N-1),
    # which used to trip an assert. The single leg is origin → boarding station.
    graph = make_mixed_mode_graph([(1, 2, Mode.BIKE), (2, 3, Mode.STATION), (3, 4, Mode.RAIL)])
    ends = bike_leg_endpoints(
        graph=graph, node_path=[1, 2, 3, 4], leg_paths=[[1, 2]], origin="Start", destination="End"
    )
    assert ends == [("Start", "Station 3")]  # boards at station 3; never reaches destination by bike


def test_format_bike_legs_labels_each_route():
    legs = [
        BikeLeg(url="u0", from_place="Horb am Neckar", to_place="Horb-Heiligenfeld"),
        BikeLeg(url="u1", from_place="Freudenstadt Stadt", to_place="Freudenstadt"),
    ]
    assert format_bike_legs(bike_legs=legs) == [
        "Bike Route 1: Horb am Neckar → Horb-Heiligenfeld",
        "Bike Route 2: Freudenstadt Stadt → Freudenstadt",
    ]
