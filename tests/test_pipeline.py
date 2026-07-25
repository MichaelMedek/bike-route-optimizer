"""Pipeline orchestration test — plan_route wired with mocks (no network/raster).

Monkeypatches the network-bound steps (geocode, Overpass graph build, DEM) with
the tiny in-memory line graph + mock DEM so the whole flow runs offline. Asserts a
single route is produced with real artifacts (GPX, PNG, Maps URL) and stats.
"""

from pathlib import Path

import pytest
from shapely.geometry import box

from bike_router import pipeline
from bike_router.geocoding import GeocodeError
from tests.conftest import DEFAULT_PARAMS, MockDEMService, make_line_graph


def _wire_offline(monkeypatch, tmp_path, graph):
    monkeypatch.setattr(pipeline, "make_geocode_fn", lambda: (lambda place: None))
    monkeypatch.setattr(
        pipeline, "geocode_endpoint", lambda place, label, geocode_fn: (48.0, 8.0) if label == "Start" else (48.0, 8.2)
    )
    monkeypatch.setattr(pipeline, "DEMService", lambda dem_path: MockDEMService(base_elevation=100.0))
    monkeypatch.setattr(pipeline, "build_corridor", lambda start_latlon, dest_latlon: box(7.9, 47.9, 8.1, 48.1))
    monkeypatch.setattr(pipeline, "corridor_within_dem", lambda polygon, dem_bounds: True)
    monkeypatch.setattr(pipeline, "build_bike_graph", lambda polygon, progress: (graph, 100))
    monkeypatch.setattr(pipeline, "snap_endpoints", lambda graph, start_latlon, dest_latlon: (1, 3))
    monkeypatch.setattr(pipeline, "enrich_elevations", lambda graph, dem: None)
    monkeypatch.setattr(
        pipeline, "route_output_paths", lambda origin, destination: (tmp_path / "r.gpx", tmp_path / "r.png")
    )


def test_plan_route_end_to_end_offline(tmp_path: Path, monkeypatch):
    _wire_offline(monkeypatch, tmp_path, make_line_graph())
    result = pipeline.plan_route(origin="Start", destination="End", dem_path=Path("unused.tif"), params=DEFAULT_PARAMS)

    assert result.gmaps_url.startswith("https://www.google.com/maps/dir/?api=1")
    assert result.gpx_path.exists() and result.gpx_path.stat().st_size > 0
    assert result.png_path.exists() and result.png_path.stat().st_size > 0
    assert result.track.distance_km > 0
    assert result.track.duration_min > 0
    assert result.track.ascent_m >= 0 and result.track.descent_m >= 0


def test_plan_route_rejects_too_short_trip(monkeypatch):
    monkeypatch.setattr(pipeline, "make_geocode_fn", lambda: (lambda place: None))
    monkeypatch.setattr(
        pipeline,
        "geocode_endpoint",
        lambda place, label, geocode_fn: (48.0, 8.0) if label == "Start" else (48.001, 8.0),
    )
    with pytest.raises(SystemExit):
        pipeline.plan_route(origin="A", destination="B", dem_path=Path("unused.tif"), params=DEFAULT_PARAMS)


def test_plan_route_rejects_too_far_trip(monkeypatch):
    # ~330 km apart (48.0 → 51.0 lat) → beyond MAX_TRIP_KM
    monkeypatch.setattr(pipeline, "make_geocode_fn", lambda: (lambda place: None))
    monkeypatch.setattr(
        pipeline, "geocode_endpoint", lambda place, label, geocode_fn: (48.0, 8.0) if label == "Start" else (51.0, 8.0)
    )
    with pytest.raises(SystemExit):
        pipeline.plan_route(origin="A", destination="B", dem_path=Path("unused.tif"), params=DEFAULT_PARAMS)


def test_plan_route_propagates_geocode_error(monkeypatch):
    def _boom(place, label, geocode_fn):
        raise GeocodeError("no such place")

    monkeypatch.setattr(pipeline, "make_geocode_fn", lambda: (lambda place: None))
    monkeypatch.setattr(pipeline, "geocode_endpoint", _boom)
    with pytest.raises(GeocodeError):
        pipeline.plan_route(
            origin="Nowhere", destination="Elsewhere", dem_path=Path("unused.tif"), params=DEFAULT_PARAMS
        )


def test_plan_route_propagates_overpass_error(monkeypatch):
    monkeypatch.setattr(pipeline, "make_geocode_fn", lambda: (lambda place: None))
    monkeypatch.setattr(
        pipeline, "geocode_endpoint", lambda place, label, geocode_fn: (48.0, 8.0) if label == "Start" else (48.0, 8.2)
    )
    monkeypatch.setattr(pipeline, "DEMService", lambda dem_path: MockDEMService(base_elevation=100.0))
    monkeypatch.setattr(pipeline, "build_corridor", lambda start_latlon, dest_latlon: box(7.9, 47.9, 8.1, 48.1))
    monkeypatch.setattr(pipeline, "corridor_within_dem", lambda polygon, dem_bounds: True)

    def _overpass_down(polygon, progress):
        raise RuntimeError("Overpass API HTTP 429")

    monkeypatch.setattr(pipeline, "build_bike_graph", _overpass_down)
    with pytest.raises(RuntimeError, match="Overpass"):
        pipeline.plan_route(origin="Start", destination="End", dem_path=Path("unused.tif"), params=DEFAULT_PARAMS)


def test_plan_route_no_route_raises_systemexit(tmp_path: Path, monkeypatch):
    graph = make_line_graph()
    graph.add_node(99, x=20.0, y=60.0, elevation=100.0)  # isolated target
    _wire_offline(monkeypatch, tmp_path, graph)
    monkeypatch.setattr(pipeline, "check_strongly_connected", lambda graph: None)  # bypass (disconnected)
    monkeypatch.setattr(pipeline, "snap_endpoints", lambda graph, start_latlon, dest_latlon: (1, 99))
    with pytest.raises(SystemExit):
        pipeline.plan_route(origin="Start", destination="End", dem_path=Path("unused.tif"), params=DEFAULT_PARAMS)
