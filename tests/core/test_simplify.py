"""simplify tests — leg splitting, station/leg labels, full-route LineString, Visvalingam waypoints.

One test_<fn> per production symbol (exact-name mirror; leading underscores stripped) and a
TestFoo per dataclass. Each folds every scenario for its target; leg-splitting drives the mode
runs, and the Google-Maps waypoint reduction is exercised across sizes, corners and thinning.
"""

import numpy as np
import pytest
from shapely.geometry import LineString

from bike_router.core.constants import GmapsConfig, Mode, NodeType
from bike_router.core.route_path import RouteNode
from bike_router.core.simplify import (
    BikeLeg,
    RailLeg,
    Station,
    _interpolate_to_n,
    _neighbour_station_name,
    _split_mode_runs,
    _station,
    _thin_close_points,
    _triangle_area,
    _visvalingam,
    bike_leg_endpoints,
    format_bike_legs,
    format_rail_legs,
    place_label,
    rail_leg_tooltips,
    route_station_markers,
    route_to_linestring,
    select_waypoints,
    split_bike_legs,
    split_rail_legs,
)
from tests.conftest import make_line_route, make_mixed_mode_route


def _zigzag_line(n_points: int = 200) -> LineString:
    """A long up/down zigzag polyline (lon 0→1) — enough points for the Visvalingam path."""
    lons = np.linspace(0.0, 1.0, n_points)
    lats = 0.001 * (np.arange(n_points) % 2)  # up/down zigzag
    return LineString(list(zip(lons.tolist(), lats.tolist(), strict=True)))


def _names(legs) -> list[tuple[str | None, str | None]]:  # noqa: ANN001 — list[RailLeg]
    """(board name, alight name) per leg — the Station→name projection the assertions want."""
    return [(leg.board.name, leg.alight.name) for leg in legs]


def _rail_leg(board: str | None, alight: str | None) -> RailLeg:
    """A RailLeg from two station names (positions irrelevant for the format/endpoint tests)."""
    return RailLeg(
        board=Station(name=board, lat=48.0, lon=8.0, elevation_m=0.0),
        alight=Station(name=alight, lat=48.0, lon=8.1, elevation_m=0.0),
    )


# --- labels ------------------------------------------------------------------


def test_place_label():
    # The ONE "Name (elev m)" format shared by endpoints, station markers and tooltips; rounds.
    assert place_label(name="Freudenstadt", elevation_m=739.0) == "Freudenstadt (739 m)"
    assert place_label(name="Horb", elevation_m=419.6) == "Horb (420 m)"  # rounds to whole metres


# --- leg splitting -----------------------------------------------------------


def test_split_mode_runs():
    # Maximal runs of consecutive same-mode edges; other modes break a run; each run is >= 2 nodes.
    route = make_mixed_mode_route(
        [(1, 2, Mode.BIKE), (2, 3, Mode.STATION), (3, 4, Mode.RAIL), (4, 5, Mode.STATION), (5, 6, Mode.BIKE)]
    )
    bike_runs = _split_mode_runs(route=route, mode=Mode.BIKE)
    assert [[n.osmid for n in run] for run in bike_runs] == [[1, 2], [5, 6]]  # two bike runs around the ride
    rail_runs = _split_mode_runs(route=route, mode=Mode.RAIL)
    assert [[n.osmid for n in run] for run in rail_runs] == [[3, 4]]  # the single rail run
    # a mode absent from the route yields no runs at all
    assert _split_mode_runs(route=make_mixed_mode_route([(1, 2, Mode.BIKE)]), mode=Mode.RAIL) == []


def test_split_bike_legs():
    # Pure-bike → one leg; a train in the middle cuts into two; an all-rail trip has no bike leg.
    pure = make_mixed_mode_route([(1, 2, Mode.BIKE), (2, 3, Mode.BIKE)])
    assert split_bike_legs(route=pure) == [[1, 2, 3]]
    around_train = make_mixed_mode_route(
        [(1, 2, Mode.BIKE), (2, 3, Mode.STATION), (3, 4, Mode.RAIL), (4, 5, Mode.STATION), (5, 6, Mode.BIKE)]
    )
    assert split_bike_legs(route=around_train) == [[1, 2], [5, 6]]
    all_rail = make_mixed_mode_route([(1, 2, Mode.STATION), (2, 3, Mode.RAIL), (3, 4, Mode.STATION)])
    assert split_bike_legs(route=all_rail) == []


def test_split_rail_legs():
    # No train → []; one ride names board/alight; an on-train change stays ONE ride; two rides split;
    # a ride that is the route's last edge still emits a leg.
    pure = make_mixed_mode_route([(1, 2, Mode.BIKE), (2, 3, Mode.BIKE)])
    assert split_rail_legs(route=pure) == []
    one_ride = make_mixed_mode_route(
        [(1, 2, Mode.BIKE), (2, 3, Mode.STATION), (3, 4, Mode.RAIL), (4, 5, Mode.STATION), (5, 6, Mode.BIKE)]
    )
    assert _names(split_rail_legs(route=one_ride)) == [("Station 3", "Station 4")]
    change = make_mixed_mode_route([(1, 2, Mode.STATION), (2, 3, Mode.RAIL), (3, 4, Mode.RAIL), (4, 5, Mode.STATION)])
    assert _names(split_rail_legs(route=change)) == [("Station 2", "Station 4")]  # junction change = one ride
    two_rides = make_mixed_mode_route(
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
    assert _names(split_rail_legs(route=two_rides)) == [("Station 2", "Station 3"), ("Station 6", "Station 7")]
    end_ride = make_mixed_mode_route([(1, 2, Mode.STATION), (2, 3, Mode.RAIL)])
    assert _names(split_rail_legs(route=end_ride)) == [("Station 2", "Station 3")]  # last edge is rail


def test_station():
    # Builds a Station from a rail node's baked attributes (name + position carried straight through).
    node = RouteNode(osmid=7, lat=48.1, lon=8.2, elevation_m=305.0, node_type=NodeType.RAIL, station_name="Bf")
    station = _station(node=node)
    assert (station.name, station.lat, station.lon, station.elevation_m) == ("Bf", 48.1, 8.2, 305.0)


# --- dataclasses -------------------------------------------------------------


class TestStation:
    def test_name_or_placeholder_and_label(self):
        named = Station(name="Freudenstadt", lat=48.0, lon=8.0, elevation_m=739.0)
        assert named.name_or_placeholder == "Freudenstadt"
        assert named.label == "Freudenstadt (739 m)"  # shared place_label format
        unnamed = Station(name=None, lat=48.0, lon=8.0, elevation_m=500.0)
        assert unnamed.name_or_placeholder == "(unnamed stop)"  # readable placeholder, never None
        assert unnamed.label == "(unnamed stop) (500 m)"

    def test_is_frozen(self):
        with pytest.raises(AttributeError):
            Station(name="A", lat=48.0, lon=8.0, elevation_m=0.0).lat = 1.0  # type: ignore[misc]


class TestRailLeg:
    def test_holds_board_and_alight_stations(self):
        leg = _rail_leg(board="A", alight="B")
        assert leg.board.name == "A" and leg.alight.name == "B"

    def test_is_frozen(self):
        with pytest.raises(AttributeError):
            _rail_leg(board="A", alight="B").board = None  # type: ignore[misc]


class TestBikeLeg:
    def test_holds_url_and_endpoint_names(self):
        leg = BikeLeg(url="https://maps", from_place="Horb", to_place="Freudenstadt")
        assert (leg.url, leg.from_place, leg.to_place) == ("https://maps", "Horb", "Freudenstadt")

    def test_is_frozen(self):
        with pytest.raises(AttributeError):
            BikeLeg(url="u", from_place="a", to_place="b").url = "x"  # type: ignore[misc]


# --- station markers / tooltips / rail-leg lines -----------------------------


def test_route_station_markers():
    # (lat, lon, elev, label) for every boarded/alighted stop across all rides, in order.
    legs = [_rail_leg(board="Freudenstadt", alight="Pforzheim")]
    markers = route_station_markers(rail_legs=legs)
    assert markers == [
        (48.0, 8.0, 0.0, "Freudenstadt (0 m)"),
        (48.0, 8.1, 0.0, "Pforzheim (0 m)"),
    ]


def test_rail_leg_tooltips():
    # One "Train: A (e) → B (e)" hover per ride, board → alight, using the shared label format.
    legs = [_rail_leg(board="Freudenstadt", alight="Pforzheim"), _rail_leg(board=None, alight="Karlsruhe")]
    assert rail_leg_tooltips(rail_legs=legs) == [
        "Train: Freudenstadt (0 m) → Pforzheim (0 m)",
        "Train: (unnamed stop) (0 m) → Karlsruhe (0 m)",
    ]


def test_format_rail_legs():
    # "Train N: board → alight" per ride; an unnamed OSM stop renders via the placeholder, not None.
    legs = [_rail_leg(board="Freudenstadt", alight="Pforzheim"), _rail_leg(board=None, alight="Karlsruhe")]
    assert format_rail_legs(rail_legs=legs) == [
        "Train 1: Freudenstadt → Pforzheim",
        "Train 2: (unnamed stop) → Karlsruhe",
    ]


# --- bike-leg endpoints / labels ---------------------------------------------


def test_bike_leg_endpoints():
    # No train → origin → destination; a train uses the abutting station names at the inner ends;
    # a route ENDING on a train has one leg (origin → boarding station) and must NOT trip the assert.
    pure = make_mixed_mode_route([(1, 2, Mode.BIKE), (2, 3, Mode.BIKE)])
    assert bike_leg_endpoints(route=pure, leg_paths=[[1, 2, 3]], origin="Horb", destination="Freudenstadt") == [
        ("Horb", "Freudenstadt")
    ]
    one_train = make_mixed_mode_route(
        [(1, 2, Mode.BIKE), (2, 3, Mode.STATION), (3, 4, Mode.RAIL), (4, 5, Mode.STATION), (5, 6, Mode.BIKE)]
    )
    assert bike_leg_endpoints(
        route=one_train, leg_paths=[[1, 2], [5, 6]], origin="Horb am Neckar", destination="Freudenstadt"
    ) == [("Horb am Neckar", "Station 3"), ("Station 4", "Freudenstadt")]
    ends_on_train = make_mixed_mode_route([(1, 2, Mode.BIKE), (2, 3, Mode.STATION), (3, 4, Mode.RAIL)])
    assert bike_leg_endpoints(route=ends_on_train, leg_paths=[[1, 2]], origin="Start", destination="End") == [
        ("Start", "Station 3")
    ]


def test_neighbour_station_name():
    # The abutting neighbour of a pedalled leg MUST be a rail node → its name (unnamed → placeholder);
    # a stray bike neighbour is a broken path and fails loud rather than reading as an unnamed stop.
    named = RouteNode(osmid=3, lat=48.0, lon=8.0, elevation_m=0.0, node_type=NodeType.RAIL, station_name="Horb")
    assert _neighbour_station_name(node=named) == "Horb"
    unnamed = RouteNode(osmid=4, lat=48.0, lon=8.0, elevation_m=0.0, node_type=NodeType.RAIL, station_name=None)
    assert _neighbour_station_name(node=unnamed) == "(unnamed stop)"
    bike = RouteNode(osmid=5, lat=48.0, lon=8.0, elevation_m=0.0, node_type=NodeType.BIKE, station_name=None)
    with pytest.raises(AssertionError, match="must be a rail station"):
        _neighbour_station_name(node=bike)


def test_format_bike_legs():
    # One "Bike Route N: from → to" label per pedalled leg, numbered from 1.
    legs = [
        BikeLeg(url="u0", from_place="Horb am Neckar", to_place="Horb-Heiligenfeld"),
        BikeLeg(url="u1", from_place="Freudenstadt Stadt", to_place="Freudenstadt"),
    ]
    assert format_bike_legs(bike_legs=legs) == [
        "Bike Route 1: Horb am Neckar → Horb-Heiligenfeld",
        "Bike Route 2: Freudenstadt Stadt → Freudenstadt",
    ]


# --- route_to_linestring -----------------------------------------------------


def test_route_to_linestring():
    # Stitches the node path into one lon/lat LineString (x=lon, y=lat), start → end in order.
    line = route_to_linestring(route=make_line_route())
    coords = list(line.coords)
    assert coords[0] == (8.0, 48.0)
    assert coords[-1] == (8.02, 48.0)


# --- select_waypoints (Google Maps) ------------------------------------------


def test_select_waypoints():
    # Reduces to AT MOST N (lat, lon) points across sizes; endpoints always kept; a significant
    # corner survives; a short line is padded up to N; an over-close tiny leg thins to just 2.
    n = GmapsConfig.N_WAYPOINTS
    points = select_waypoints(line=_zigzag_line(n_points=200), count=n)
    assert 2 <= len(points) <= n and all(len(point) == 2 for point in points)
    for n_points in (11, 50, 200, 500):
        # count is a CEILING; the 5 km min-spacing thin may drop near-clustered corners below it.
        assert len(select_waypoints(line=_zigzag_line(n_points=n_points), count=10)) <= 10

    line = _zigzag_line(n_points=200)
    coords = list(line.coords)  # (lon, lat)
    reduced = select_waypoints(line=line, count=10)
    first_lat, first_lon = reduced[0]
    last_lat, last_lon = reduced[-1]
    assert (first_lon, first_lat) == coords[0] and (last_lon, last_lat) == coords[-1]  # endpoints preserved

    padded = select_waypoints(line=LineString([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]), count=10)
    assert len(padded) == 10 and padded[0] == (0.0, 0.0) and padded[-1] == (0.0, 2.0)  # (lat, lon)

    corner = select_waypoints(line=LineString([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]), count=3)
    assert (0.0, 1.0) in corner  # the significant corner (lat=0, lon=1)

    # a ~37 m leg near (48, 8): every interior point is under the 5 km spacing → only the two ends
    tiny = select_waypoints(line=LineString([(8.0000 + i * 0.0001, 48.0) for i in range(6)]), count=10)
    assert tiny == [(48.0, 8.0), (48.0, 8.0005)]


def test_thin_close_points():
    # Far-apart points and endpoints survive; a near-duplicate interior point is dropped.
    far = [(48.0, 8.0), (48.0, 8.02), (48.0, 8.04)]  # ~1.5 km apart at 48°N
    assert _thin_close_points(points=far, min_spacing_km=1.0) == far
    near = [(48.0, 8.0), (48.0, 8.00001), (48.0, 8.04)]
    assert _thin_close_points(points=near, min_spacing_km=1.0) == [(48.0, 8.0), (48.0, 8.04)]


def test_triangle_area():
    # Absolute planar area; a right triangle base 2 height 3 → 3.0; three collinear points → 0.
    assert _triangle_area(point_a=(0.0, 0.0), point_b=(2.0, 0.0), point_c=(0.0, 3.0)) == 3.0
    assert _triangle_area(point_a=(0.0, 0.0), point_b=(1.0, 0.0), point_c=(2.0, 0.0)) == 0.0


def test_visvalingam():
    # Drops the smallest-effective-area interior point until count remain; deterministic order kept.
    coords = [(0.0, 0.0), (1.0, 0.001), (2.0, 0.0), (3.0, 5.0), (4.0, 0.0)]
    assert _visvalingam(coords=coords, count=4) == [(0.0, 0.0), (2.0, 0.0), (3.0, 5.0), (4.0, 0.0)]


def test_interpolate_to_n():
    # Resamples a polyline to exactly N points evenly spaced by arc length; a single point repeats.
    assert _interpolate_to_n(coords=[(0.0, 0.0), (4.0, 0.0)], count=5) == [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (3.0, 0.0),
        (4.0, 0.0),
    ]
    assert _interpolate_to_n(coords=[(2.0, 3.0)], count=3) == [(2.0, 3.0)] * 3
