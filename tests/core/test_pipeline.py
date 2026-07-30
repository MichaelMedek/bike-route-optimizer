"""Pipeline orchestration test — plan_route wired with mocks (no network).

Monkeypatches the network-bound steps (geocode, corridor-table load, path re-read) with
tiny in-memory fixtures so the whole flow runs offline. No DEM is involved — elevation is
baked into the fixture. Asserts a single route is produced with real artifacts.
"""

from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import box

from bike_router.core import pipeline
from bike_router.core.constants import CorridorConfig, GeoConfig, GraphConfig, Mode, NodeType
from bike_router.core.errors import (
    GeocodeConnectionError,
    NoRouteError,
    OutOfCoverageError,
    RouteTooLargeError,
    TripTooLongError,
    TripTooShortError,
)
from bike_router.core.route_graph import RouteGraph, shortest_path
from bike_router.core.route_path import RoutePath
from tests.conftest import (
    DEFAULT_PARAMS,
    FIXTURE_GRAPH_DIR,
    make_hill_vs_rail_edges,
    make_line_edges,
    make_line_route,
    params,
    route_node_types,
)

_ROUTE_NODE_COLS = ["osmid", "lat", "lon", "elevation_m", "node_type"]
_ROUTE_EDGE_COLS = ["from_node", "to_node", "length_m", "surface", "highway", "mode"]


def _line_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Minimal routing tables (no geometry) for the line graph 1→2→3, as load_route_tables returns."""
    arr = make_line_edges()
    args = arr.route_graph_args(params=DEFAULT_PARAMS)
    nodes_df = pd.DataFrame(
        {
            "osmid": args["osmids"],
            "lat": args["lat"],
            "lon": args["lon"],
            "elevation_m": [100.0, 130.0, 100.0],
            "node_type": [NodeType.BIKE, NodeType.BIKE, NodeType.BIKE],
        },
        columns=_ROUTE_NODE_COLS,
    )
    edges_df = pd.DataFrame(
        {
            "from_node": args["from_osmid"],
            "to_node": args["to_osmid"],
            "length_m": [800.0, 800.0, 800.0, 800.0],
            "surface": ["asphalt"] * 4,
            "highway": ["residential"] * 4,
            "mode": [Mode.BIKE] * 4,
        },
        columns=_ROUTE_EDGE_COLS,
    )
    return nodes_df, edges_df


def _wire_offline(monkeypatch, tmp_path, *, nodes_df, edges_df, route: RoutePath):
    monkeypatch.setattr(pipeline, "make_geocode_fn", lambda: lambda place: None)
    monkeypatch.setattr(
        pipeline, "geocode_endpoint", lambda place, label, geocode_fn: (48.0, 8.0) if label == "Start" else (48.0, 8.2)
    )
    monkeypatch.setattr(
        pipeline,
        "build_corridor",
        lambda start_latlon, dest_latlon, half_width_km, extend_km: box(7.9, 47.9, 8.1, 48.1),
    )
    monkeypatch.setattr(pipeline, "_assert_within_coverage", lambda start_latlon, dest_latlon, graph_dir: None)
    # The CSR router loads corridor tables then re-reads the chosen path's edges into a RoutePath —
    # stub BOTH with the fixtures so the whole flow runs offline (no dataset).
    monkeypatch.setattr(
        pipeline, "load_route_tables", lambda bike_corridor, rail_corridor, graph_dir: (nodes_df, edges_df)
    )
    monkeypatch.setattr(pipeline, "load_path_edges", lambda path_nodes, params, graph_dir: route)
    monkeypatch.setattr(
        pipeline, "route_output_paths", lambda origin, destination, params: (tmp_path / "r.gpx", tmp_path / "r.png")
    )


def test_plan_route(tmp_path: Path, monkeypatch):
    nodes_df, edges_df = _line_tables()
    _wire_offline(monkeypatch, tmp_path, nodes_df=nodes_df, edges_df=edges_df, route=make_line_route())
    result = pipeline.plan_route(origin="Start", destination="End", params=DEFAULT_PARAMS)

    assert len(result.bike_legs) == 1  # pure-bike line graph → exactly one pedalled leg
    leg = result.bike_legs[0]
    assert leg.url.startswith("https://www.google.com/maps/dir/?api=1")
    assert (leg.from_place, leg.to_place) == ("Start", "End")  # outer ends = origin/destination
    assert result.rail_legs == []  # pure-bike line graph → no train ride
    assert result.gpx_path.exists() and result.gpx_path.stat().st_size > 0
    assert result.png_path.exists() and result.png_path.stat().st_size > 0
    # line graph 1→2→3: two 800 m edges = 1.6 km; 100→130→100 m = +30 / −30 m exactly.
    assert result.track.total.distance_km == pytest.approx(1.6)
    assert result.track.total.duration_min > 0
    assert result.track.total.ascent_m == pytest.approx(30.0) and result.track.total.descent_m == pytest.approx(30.0)
    assert result.track.bike == result.track.total  # pure-bike route: bike-only == total
    # composition is all bike (line graph has no rail): bike km == the full route distance.
    assert result.composition.by_mode_km["bike route"] == pytest.approx(1.6)
    assert sum(result.composition.by_surface_km.values()) == pytest.approx(1.6)


def test_plan_route_rejects_too_short_trip(monkeypatch):
    monkeypatch.setattr(pipeline, "make_geocode_fn", lambda: lambda place: None)
    monkeypatch.setattr(
        pipeline,
        "geocode_endpoint",
        lambda place, label, geocode_fn: (48.0, 8.0) if label == "Start" else (48.001, 8.0),
    )
    with pytest.raises(TripTooShortError):
        pipeline.plan_route(origin="A", destination="B", params=DEFAULT_PARAMS)


def test_plan_route_rejects_too_far_trip(monkeypatch):
    # Just beyond MAX_TRIP_KM north of the start (derive the latitude delta from the constant so
    # this stays correct if the bound changes). ~1.2× the limit in degrees of latitude.
    far_lat = 48.0 + 1.2 * CorridorConfig.MAX_TRIP_KM / (GeoConfig.METERS_PER_DEGREE_EQUATOR / 1000.0)
    monkeypatch.setattr(pipeline, "make_geocode_fn", lambda: lambda place: None)
    monkeypatch.setattr(
        pipeline,
        "geocode_endpoint",
        lambda place, label, geocode_fn: (48.0, 8.0) if label == "Start" else (far_lat, 8.0),
    )
    with pytest.raises(TripTooLongError):
        pipeline.plan_route(origin="A", destination="B", params=DEFAULT_PARAMS)


def test_plan_route_propagates_geocode_error(monkeypatch):
    def _boom(place, label, geocode_fn):
        raise GeocodeConnectionError("no connection")

    monkeypatch.setattr(pipeline, "make_geocode_fn", lambda: lambda place: None)
    monkeypatch.setattr(pipeline, "geocode_endpoint", _boom)
    with pytest.raises(GeocodeConnectionError):
        pipeline.plan_route(origin="Nowhere", destination="Elsewhere", params=DEFAULT_PARAMS)


def test_plan_route_rejects_outside_coverage(monkeypatch):
    monkeypatch.setattr(pipeline, "make_geocode_fn", lambda: lambda place: None)
    monkeypatch.setattr(
        pipeline, "geocode_endpoint", lambda place, label, geocode_fn: (48.0, 8.0) if label == "Start" else (48.0, 8.2)
    )
    monkeypatch.setattr(
        pipeline,
        "build_corridor",
        lambda start_latlon, dest_latlon, half_width_km, extend_km: box(7.9, 47.9, 8.1, 48.1),
    )

    def _outside(start_latlon, dest_latlon, graph_dir):
        raise OutOfCoverageError("Route is outside the prebuilt graph coverage")

    monkeypatch.setattr(pipeline, "_assert_within_coverage", _outside)
    with pytest.raises(OutOfCoverageError, match="coverage"):
        pipeline.plan_route(origin="Start", destination="End", params=DEFAULT_PARAMS)


def test_plan_route_no_route_raises_no_route_error(tmp_path: Path, monkeypatch):
    # Isolated bike node AT the destination (48.0, 8.2) so snap picks it, but no edge reaches it.
    nodes_df, edges_df = _line_tables()
    nodes_df = pd.concat(
        [
            nodes_df,
            pd.DataFrame([{"osmid": 99, "lat": 48.0, "lon": 8.2, "elevation_m": 100.0, "node_type": NodeType.BIKE}]),
        ],
        ignore_index=True,
    )
    _wire_offline(monkeypatch, tmp_path, nodes_df=nodes_df, edges_df=edges_df, route=make_line_route())
    with pytest.raises(NoRouteError):
        pipeline.plan_route(origin="Start", destination="End", params=DEFAULT_PARAMS)


def test_plan_route_rejects_corridor_over_edge_cap(tmp_path: Path, monkeypatch):
    # A corridor whose edge count exceeds the memory cap must raise RouteTooLargeError
    # (the hard OOM guard) instead of attempting the load.
    nodes_df, edges_df = _line_tables()
    _wire_offline(monkeypatch, tmp_path, nodes_df=nodes_df, edges_df=edges_df, route=make_line_route())
    monkeypatch.setattr(CorridorConfig, "MAX_ROUTE_EDGES", 2)  # line graph has 4 edges > 2
    with pytest.raises(RouteTooLargeError, match="too large"):
        pipeline.plan_route(origin="Start", destination="End", params=DEFAULT_PARAMS)


def test_resolve_endpoints(monkeypatch):
    # Whatever text is passed IS what's geocoded (the web app hands the box text here),
    # then each result snaps to the nearest node's (lat, lon, elevation).
    monkeypatch.setattr(pipeline, "make_geocode_fn", lambda: lambda place: None)
    monkeypatch.setattr(
        pipeline, "geocode_endpoint", lambda place, label, geocode_fn: (48.0, 8.0) if label == "Start" else (48.5, 8.5)
    )
    monkeypatch.setattr(pipeline, "snap_to_node", lambda lat, lon, graph_dir: (lat + 0.001, lon + 0.001, 200.0))
    start, end = pipeline.resolve_endpoints(origin="Freudenstadt", destination="Pforzheim")
    assert start == (48.001, 8.001, 200.0)
    assert end == (48.501, 8.501, 200.0)


class TestRouteResult:
    def test_bundles_track_paths_legs_and_composition(self, tmp_path: Path, monkeypatch):
        # The pipeline's return bundle: track + written artifact paths + bike/rail legs + composition.
        nodes_df, edges_df = _line_tables()
        _wire_offline(monkeypatch, tmp_path, nodes_df=nodes_df, edges_df=edges_df, route=make_line_route())
        result = pipeline.plan_route(origin="Start", destination="End", params=DEFAULT_PARAMS)
        assert isinstance(result, pipeline.RouteResult)
        assert result.track is not None and result.composition is not None
        assert result.gpx_path.exists() and result.png_path.exists()
        assert result.bike_legs and result.rail_legs == []
        assert isinstance(result.waypoints, list)


def test_format_cli_report(tmp_path: Path, monkeypatch):
    # The CLI's whole stdout block, assembled in core so bike_route.py stays a pure I/O shell:
    # bike-vs-total stats, composition, GPX/PNG paths, and one Maps link per pedalled leg. A pure-bike
    # route has NO "Trains to catch" section; a train route lists it.
    nodes_df, edges_df = _line_tables()
    _wire_offline(monkeypatch, tmp_path, nodes_df=nodes_df, edges_df=edges_df, route=make_line_route())
    result = pipeline.plan_route(origin="Start", destination="End", params=DEFAULT_PARAMS)
    report = pipeline.format_cli_report(result=result)
    assert "Total (bike + train):" in report and "Bike only:" in report
    assert "Mode:" in report  # the composition summary is embedded
    assert str(result.gpx_path) in report and str(result.png_path) in report
    assert "Bike legs in Google Maps" in report and result.bike_legs[0].url in report
    assert "Trains to catch:" not in report  # pure-bike line route → no train section


def test_geocode_both(monkeypatch):
    # ONE rate-limited geocode fn resolves both endpoints to (lat, lon); a bad Start fails fast.
    monkeypatch.setattr(pipeline, "make_geocode_fn", lambda: lambda place: None)
    monkeypatch.setattr(
        pipeline, "geocode_endpoint", lambda place, label, geocode_fn: (48.0, 8.0) if label == "Start" else (48.5, 8.5)
    )
    start, dest = pipeline._geocode_both(origin="Freudenstadt", destination="Pforzheim")
    assert start == (48.0, 8.0) and dest == (48.5, 8.5)

    def _boom(place, label, geocode_fn):
        raise GeocodeConnectionError("no connection")

    monkeypatch.setattr(pipeline, "geocode_endpoint", _boom)
    with pytest.raises(GeocodeConnectionError):
        pipeline._geocode_both(origin="X", destination="Y")


def test_assert_within_coverage(monkeypatch):
    # Passes silently inside the bbox; raises OutOfCoverageError for an endpoint outside it.
    monkeypatch.setattr(pipeline, "load_meta", lambda graph_dir: {"bbox": [8.0, 48.0, 8.5, 48.5]})
    pipeline._assert_within_coverage(start_latlon=(48.1, 8.1), dest_latlon=(48.4, 8.4), graph_dir=FIXTURE_GRAPH_DIR)
    with pytest.raises(OutOfCoverageError, match="coverage"):
        pipeline._assert_within_coverage(
            start_latlon=(48.1, 8.1), dest_latlon=(52.5, 13.4), graph_dir=FIXTURE_GRAPH_DIR
        )


def test_route_node_path(monkeypatch):
    # Loads the corridor tables, routes on the CSR graph, returns the path as (osmid, lat, lon);
    # a corridor over the edge cap raises RouteTooLargeError before building anything.
    nodes_df, edges_df = _line_tables()
    monkeypatch.setattr(
        pipeline, "load_route_tables", lambda bike_corridor, rail_corridor, graph_dir: (nodes_df, edges_df)
    )
    corridor = box(7.9, 47.9, 8.3, 48.1)
    path = pipeline._route_node_path(
        bike_corridor=corridor,
        rail_corridor=corridor,
        graph_dir=FIXTURE_GRAPH_DIR,
        start_latlon=(48.0, 8.0),
        dest_latlon=(48.0, 8.2),
        params=DEFAULT_PARAMS,
    )
    assert [osmid for osmid, _lat, _lon in path] == [1, 2, 3]
    assert all(isinstance(lat, float) and isinstance(lon, float) for _o, lat, lon in path)

    monkeypatch.setattr(CorridorConfig, "MAX_ROUTE_EDGES", 2)  # line graph has 4 edges > 2
    with pytest.raises(RouteTooLargeError, match="too large"):
        pipeline._route_node_path(
            bike_corridor=corridor,
            rail_corridor=corridor,
            graph_dir=FIXTURE_GRAPH_DIR,
            start_latlon=(48.0, 8.0),
            dest_latlon=(48.0, 8.2),
            params=DEFAULT_PARAMS,
        )


# --- DEFAULT-params mode choice: synthetic contract + real-route e2e (folded from test_default_params) ---


def _uses_train_path(arr, path: list[int]) -> bool:
    """True iff the route visits a rail-station node (boards a train)."""
    return NodeType.RAIL in route_node_types(arr=arr, path=path)


_VERY_STEEP, _STEEP, _MILD = 400.0, 300.0, 40.0
# (label, climb_m, rail_alternative, downhill, expect_train) — 10 synthetic scenarios.
_MODE_CASES = [
    ("steep uphill, train available → train", _STEEP, True, False, True),
    ("very steep uphill, train available → train", _VERY_STEEP, True, False, True),
    ("steep uphill, NO train → bike", _STEEP, False, False, False),
    ("steep DOWNHILL, train available → bike", _STEEP, True, True, False),
    ("very steep downhill, train available → bike", _VERY_STEEP, True, True, False),
    ("mild uphill, train available → bike", _MILD, True, False, False),
    ("mild uphill, NO train → bike", _MILD, False, False, False),
    ("mild downhill, train available → bike", _MILD, True, True, False),
    ("flat, train available → bike", 0.0, True, False, False),
    ("flat, NO train → bike", 0.0, False, False, False),
]


@pytest.mark.parametrize(
    ("label", "climb_m", "rail", "downhill", "expect_train"), _MODE_CASES, ids=[c[0] for c in _MODE_CASES]
)
def test_default_params_pick_expected_mode(
    label: str,
    climb_m: float,
    rail: bool,  # noqa: FBT001
    downhill: bool,  # noqa: FBT001
    expect_train: bool,  # noqa: FBT001
) -> None:
    """With the DEFAULT params, the synthetic router bikes or trains as a sensible rider would."""
    arr = make_hill_vs_rail_edges(climb_m=climb_m, rail_alternative=rail)
    rg = RouteGraph.from_arrays(**arr.route_graph_args(params=DEFAULT_PARAMS))
    source, target = (2, 1) if downhill else (1, 2)  # downhill = start at the high end
    path = shortest_path(route_graph=rg, source_osmid=source, target_osmid=target)
    assert _uses_train_path(arr=arr, path=path) is expect_train, f"{label}: got path {path}"


# GIVEN ground truth (origin, destination, expect_train) — real German towns, both directions.
_REAL_CASES = [
    ("Baiersbronn, Germany", "Freudenstadt, Germany", True),
    ("Freudenstadt, Germany", "Baiersbronn, Germany", False),
    ("Freudenstadt, Germany", "Pforzheim, Germany", False),
    ("Pforzheim, Germany", "Freudenstadt, Germany", True),
    ("Horb am Neckar, Germany", "Freudenstadt, Germany", True),
    ("Freudenstadt, Germany", "Horb am Neckar, Germany", False),
    ("Freudenstadt, Germany", "Nagold, Germany", False),
    ("Nagold, Germany", "Freudenstadt, Germany", True),
    ("Nagold, Germany", "Calw, Germany", False),
    ("Calw, Germany", "Nagold, Germany", False),
    ("Bad Wildbad, Germany", "Pforzheim, Germany", False),
    ("Pforzheim, Germany", "Bad Wildbad, Germany", True),
    ("Calw, Germany", "Pforzheim, Germany", False),
    ("Pforzheim, Germany", "Calw, Germany", False),
    ("Bad Wildbad, Germany", "Simmersfeld, Germany", False),
    ("Simmersfeld, Germany", "Bad Wildbad, Germany", False),
]


@pytest.mark.skipif(
    not (GraphConfig.GRAPH_DIR / GraphConfig.META_FILENAME).exists(),
    reason="real dataset not present in data/ (only the committed fixture is available)",
)
@pytest.mark.parametrize(
    ("origin", "destination", "expect_train"),
    _REAL_CASES,
    ids=[f"{o.split(',')[0]}->{d.split(',')[0]}" for o, d, _ in _REAL_CASES],
)
def test_default_params_real_route_mode(origin: str, destination: str, expect_train: bool) -> None:  # noqa: FBT001
    """FULL e2e: DEFAULT params, real dataset, real OSM geocoding — each route bikes or trains as given."""
    result = pipeline.plan_route(origin=origin, destination=destination, params=DEFAULT_PARAMS)
    assert ("train path" in result.composition.by_mode_km) is expect_train


# --- FULL e2e against the committed Schwarzwald fixture (folded from test_e2e_fixture) --------

# Two real points inside the fixture coverage (Schwarzwald, ~18 km apart, net downhill).
_SOUTH = (48.4503, 8.4608)
_NORTH = (48.5601, 8.3981)
# Baiersbronn → Freudenstadt: both inside the fixture, on the same rail line ~7 km apart.
_BAIERSBRONN = (48.5057, 8.3703)
_FREUDENSTADT = (48.4634, 8.4111)


def _stub_fixture_geocode(monkeypatch, start: tuple[float, float], end: tuple[float, float]) -> None:
    """Stub ONLY the network geocoder; everything else runs for real against the fixture."""
    monkeypatch.setattr(pipeline, "make_geocode_fn", lambda: lambda place: None)
    monkeypatch.setattr(
        pipeline, "geocode_endpoint", lambda place, label, geocode_fn: start if label == "Start" else end
    )


def _plan_fixture(monkeypatch, tmp_path, start, end, **overrides):
    _stub_fixture_geocode(monkeypatch=monkeypatch, start=start, end=end)
    monkeypatch.setattr(
        pipeline, "route_output_paths", lambda origin, destination, params: (tmp_path / "r.gpx", tmp_path / "r.png")
    )
    return pipeline.plan_route(
        origin="Start", destination="End", params=params(**overrides), graph_dir=FIXTURE_GRAPH_DIR
    )


def _rail_ride_count(track) -> int:  # noqa: ANN001 — Track from pipeline
    """Number of maximal contiguous rail-mode runs (= distinct train rides) in a track."""
    rides = 0
    prev_rail = False
    for point in track.points:
        is_rail = point.mode == Mode.RAIL
        if is_rail and not prev_rail:
            rides += 1
        prev_rail = is_rail
    return rides


def test_plan_route_e2e_real_route_from_fixture(tmp_path: Path, monkeypatch):
    """Full pipeline on the real fixture yields a plausible bike route + real artifacts."""
    result = _plan_fixture(monkeypatch=monkeypatch, tmp_path=tmp_path, start=_SOUTH, end=_NORTH)
    track = result.track
    assert result.gpx_path.exists() and result.gpx_path.stat().st_size > 0
    assert result.png_path.exists() and result.png_path.stat().st_size > 0
    assert len(result.bike_legs) >= 1
    assert all(leg.url.startswith("https://www.google.com/maps/dir/?api=1") for leg in result.bike_legs)
    assert 8.0 < track.total.distance_km < 60.0
    assert track.total.duration_min > 0
    assert track.total.ascent_m > 0 and track.total.descent_m > 0
    assert track.bike.distance_km <= track.total.distance_km
    assert len(track.points) > 50  # densified along the real 3D polyline
    assert all(p.elevation_m > 0 for p in track.points)  # baked elevations, no DEM at inference
    assert sum(result.composition.by_surface_km.values()) == pytest.approx(sum(result.composition.by_road_km.values()))
    assert result.composition.by_mode_km


def test_plan_route_e2e_flat_hater_still_routes(tmp_path: Path, monkeypatch):
    """A high uphill penalty must still produce a valid (longer/flatter) route."""
    result = _plan_fixture(
        monkeypatch=monkeypatch, tmp_path=tmp_path, start=_SOUTH, end=_NORTH, extra_km_per_uphill_100m=50.0
    )
    assert result.track.total.distance_km > 0
    assert result.gpx_path.exists()


def test_plan_route_e2e_one_train_two_bike_legs(tmp_path: Path, monkeypatch):
    """With rail made cheap, the same-line pair rides exactly ONE train, split into TWO bike legs."""
    result = _plan_fixture(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        start=_BAIERSBRONN,
        end=_FREUDENSTADT,
        extra_km_per_boarding=0.2,
        extra_km_per_rail_km=0.05,
    )
    assert _rail_ride_count(track=result.track) == 1
    assert result.composition.by_mode_km["train path"] > 0
    assert len(result.bike_legs) == 2
    assert all(leg.url.startswith("https://www.google.com/maps/dir/?api=1") for leg in result.bike_legs)
    assert result.bike_legs[0].from_place == "Start"  # _plan_fixture stubs origin="Start"
    assert result.bike_legs[1].to_place == "End"
    assert result.bike_legs[0].to_place == result.rail_legs[0].board.name_or_placeholder
    assert result.bike_legs[1].from_place == result.rail_legs[0].alight.name_or_placeholder
    assert result.composition.by_mode_km["bike route"] > 0
    assert "station" not in result.composition.by_mode_km


def test_plan_route_e2e_default_sliders_train_uphill_bike_downhill(tmp_path: Path, monkeypatch):
    """At DEFAULT sliders the tuned rider trains UP the 192 m climb but bikes back DOWN it."""
    uphill = _plan_fixture(monkeypatch=monkeypatch, tmp_path=tmp_path, start=_BAIERSBRONN, end=_FREUDENSTADT)
    assert _rail_ride_count(track=uphill.track) == 1
    assert uphill.composition.by_mode_km["train path"] > 0

    downhill = _plan_fixture(monkeypatch=monkeypatch, tmp_path=tmp_path, start=_FREUDENSTADT, end=_BAIERSBRONN)
    assert _rail_ride_count(track=downhill.track) == 0
    assert "train path" not in downhill.composition.by_mode_km
    assert len(downhill.bike_legs) == 1


def test_plan_route_e2e_out_of_coverage_raises(tmp_path: Path, monkeypatch):
    """Endpoints outside the fixture bbox fail loud (OutOfCoverageError), not silently."""
    _stub_fixture_geocode(monkeypatch=monkeypatch, start=(52.5, 13.4), end=(52.6, 13.5))  # Berlin — outside fixture
    with pytest.raises(OutOfCoverageError, match="coverage"):
        pipeline.plan_route(origin="Start", destination="End", params=params(), graph_dir=FIXTURE_GRAPH_DIR)
