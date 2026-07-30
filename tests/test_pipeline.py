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
from bike_router.core.constants import CorridorConfig, GeoConfig, Mode, NodeType
from bike_router.core.errors import (
    GeocodeConnectionError,
    NoRouteError,
    OutOfCoverageError,
    RouteTooLargeError,
    TripTooLongError,
    TripTooShortError,
)
from bike_router.core.route_path import RoutePath
from tests.conftest import DEFAULT_PARAMS, make_line_edges, make_line_route

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


def test_plan_route_end_to_end_offline(tmp_path: Path, monkeypatch):
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


def test_resolve_endpoints_geocodes_box_text_and_snaps(monkeypatch):
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
