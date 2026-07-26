"""Pipeline orchestration test — plan_route wired with mocks (no network).

Monkeypatches the network-bound steps (geocode, corridor-graph load) with the tiny
in-memory line graph so the whole flow runs offline. No DEM is involved — elevation
is baked into the graph. Asserts a single route is produced with real artifacts.
"""

from pathlib import Path

import pytest
from shapely.geometry import box

from bike_router import pipeline
from bike_router.errors import (
    GeocodeConnectionError,
    NoRouteError,
    OutOfCoverageError,
    TripTooLongError,
    TripTooShortError,
)
from tests.conftest import DEFAULT_PARAMS, make_line_graph


def _wire_offline(monkeypatch, tmp_path, graph):
    monkeypatch.setattr(pipeline, "make_geocode_fn", lambda: lambda place: None)
    monkeypatch.setattr(
        pipeline, "geocode_endpoint", lambda place, label, geocode_fn: (48.0, 8.0) if label == "Start" else (48.0, 8.2)
    )
    monkeypatch.setattr(pipeline, "build_corridor", lambda start_latlon, dest_latlon: box(7.9, 47.9, 8.1, 48.1))
    monkeypatch.setattr(pipeline, "_assert_within_coverage", lambda start_latlon, dest_latlon, graph_dir: None)
    monkeypatch.setattr(pipeline, "load_corridor_graph", lambda corridor: graph)
    monkeypatch.setattr(pipeline, "snap_endpoints", lambda graph, start_latlon, dest_latlon: (1, 3))
    monkeypatch.setattr(
        pipeline, "route_output_paths", lambda origin, destination: (tmp_path / "r.gpx", tmp_path / "r.png")
    )


def test_plan_route_end_to_end_offline(tmp_path: Path, monkeypatch):
    _wire_offline(monkeypatch, tmp_path, make_line_graph())
    result = pipeline.plan_route(origin="Start", destination="End", params=DEFAULT_PARAMS)

    assert result.gmaps_urls == [result.gmaps_urls[0]]  # pure-bike line graph → exactly one leg
    assert result.gmaps_urls[0].startswith("https://www.google.com/maps/dir/?api=1")
    assert result.rail_legs == []  # pure-bike line graph → no train ride
    assert result.gpx_path.exists() and result.gpx_path.stat().st_size > 0
    assert result.png_path.exists() and result.png_path.stat().st_size > 0
    # line graph 1→2→3: two 800 m edges = 1.6 km; 100→130→100 m = +30 / −30 m exactly.
    assert result.track.total.distance_km == pytest.approx(1.6)
    assert result.track.total.duration_min > 0
    assert result.track.total.ascent_m == pytest.approx(30.0) and result.track.total.descent_m == pytest.approx(30.0)
    assert result.track.bike == result.track.total  # pure-bike route: bike-only == total
    # composition is all bike (line graph has no rail): bike km == the full route distance.
    assert result.composition.by_mode_km["bike"] == pytest.approx(1.6)
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
    # ~330 km apart (48.0 → 51.0 lat) → beyond MAX_TRIP_KM
    monkeypatch.setattr(pipeline, "make_geocode_fn", lambda: lambda place: None)
    monkeypatch.setattr(
        pipeline, "geocode_endpoint", lambda place, label, geocode_fn: (48.0, 8.0) if label == "Start" else (51.0, 8.0)
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
    monkeypatch.setattr(pipeline, "build_corridor", lambda start_latlon, dest_latlon: box(7.9, 47.9, 8.1, 48.1))

    def _outside(start_latlon, dest_latlon, graph_dir):
        raise OutOfCoverageError("Route is outside the prebuilt graph coverage")

    monkeypatch.setattr(pipeline, "_assert_within_coverage", _outside)
    with pytest.raises(OutOfCoverageError, match="coverage"):
        pipeline.plan_route(origin="Start", destination="End", params=DEFAULT_PARAMS)


def test_plan_route_no_route_raises_no_route_error(tmp_path: Path, monkeypatch):
    graph = make_line_graph()
    graph.add_node(99, x=20.0, y=60.0, elevation=100.0)  # isolated target
    _wire_offline(monkeypatch, tmp_path, graph)
    monkeypatch.setattr(pipeline, "check_strongly_connected", lambda graph: None)  # bypass (disconnected)
    monkeypatch.setattr(pipeline, "snap_endpoints", lambda graph, start_latlon, dest_latlon: (1, 99))
    with pytest.raises(NoRouteError):
        pipeline.plan_route(origin="Start", destination="End", params=DEFAULT_PARAMS)


def test_resolve_endpoints_geocodes_box_text_and_snaps(monkeypatch):
    # Whatever text is passed IS what's geocoded (the web app hands the box text here),
    # then each result snaps to the nearest node's (lat, lon, elevation).
    monkeypatch.setattr(pipeline, "make_geocode_fn", lambda: lambda place: None)
    monkeypatch.setattr(
        pipeline, "geocode_endpoint", lambda place, label, geocode_fn: (48.0, 8.0) if label == "Start" else (48.5, 8.5)
    )
    monkeypatch.setattr(pipeline, "snap_to_node", lambda lat, lon: (lat + 0.001, lon + 0.001, 200.0))
    start, end = pipeline.resolve_endpoints(origin="Freudenstadt", destination="Pforzheim")
    assert start == (48.001, 8.001, 200.0)
    assert end == (48.501, 8.501, 200.0)
