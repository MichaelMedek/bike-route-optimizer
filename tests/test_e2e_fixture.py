"""End-to-end route tests against the REAL committed Schwarzwald fixture.

These exercise the full production ``plan_route`` on the prebuilt graph in
tests/fixtures/dach_graph — the ONLY data source allowed here. Nothing about
routing, the graph, elevation, or output is mocked; only ``geocode_endpoint`` is
stubbed to fixed lat/lons so the tests don't hit the Nominatim network.
"""

from pathlib import Path

import pytest

from bike_router import pipeline
from bike_router.constants import PARAM_SPECS, RoutingParams
from tests.conftest import FIXTURE_GRAPH_DIR

# Two real points inside the fixture coverage (Schwarzwald, ~18 km apart, net downhill).
_SOUTH = (48.4503, 8.4608)
_NORTH = (48.5601, 8.3981)


def _params(**overrides: float) -> RoutingParams:
    values = {spec.field: spec.default for spec in PARAM_SPECS}
    values.update(overrides)
    return RoutingParams(**values)


def _stub_geocode(monkeypatch, start: tuple[float, float], end: tuple[float, float]) -> None:
    """Stub ONLY the network geocoder; everything else runs for real."""
    monkeypatch.setattr(pipeline, "make_geocode_fn", lambda: lambda place: None)
    monkeypatch.setattr(
        pipeline, "geocode_endpoint", lambda place, label, geocode_fn: start if label == "Start" else end
    )


def _plan(monkeypatch, tmp_path, start, end, **overrides):
    _stub_geocode(monkeypatch, start, end)
    monkeypatch.setattr(
        pipeline, "route_output_paths", lambda origin, destination: (tmp_path / "r.gpx", tmp_path / "r.png")
    )
    return pipeline.plan_route(
        origin="Start", destination="End", params=_params(**overrides), graph_dir=FIXTURE_GRAPH_DIR
    )


def test_e2e_real_route_produced_from_fixture(tmp_path: Path, monkeypatch):
    """Full pipeline on the real fixture yields a plausible bike route + real artifacts."""
    result = _plan(monkeypatch, tmp_path, _SOUTH, _NORTH)
    track = result.track

    # Real GPX + PNG written to the temp folder.
    assert result.gpx_path.exists() and result.gpx_path.stat().st_size > 0
    assert result.png_path.exists() and result.png_path.stat().st_size > 0
    assert result.gmaps_url.startswith("https://www.google.com/maps/dir/?api=1")

    # Plausible Schwarzwald route: sane distance, positive time, real elevation change.
    assert 8.0 < track.distance_km < 60.0
    assert track.duration_min > 0
    assert track.ascent_m > 0 and track.descent_m > 0
    assert len(track.points) > 50  # densified along the real 3D polyline
    assert all(p.elevation_m > 0 for p in track.points)  # baked elevations, no DEM at inference

    # Composition covers the pedalled distance by surface + road class.
    assert sum(result.composition.by_surface_km.values()) == pytest.approx(track.distance_km, rel=0.05)
    assert result.composition.by_mode_km  # at least the bike mode present


def test_e2e_flat_hater_still_routes(tmp_path: Path, monkeypatch):
    """A high uphill penalty must still produce a valid (longer/flatter) route."""
    result = _plan(monkeypatch, tmp_path, _SOUTH, _NORTH, extra_km_per_uphill_100m=50.0)
    assert result.track.distance_km > 0
    assert result.gpx_path.exists()


def test_e2e_out_of_coverage_raises(tmp_path: Path, monkeypatch):
    """Endpoints outside the fixture bbox fail loud (ValueError), not silently."""
    _stub_geocode(monkeypatch, (52.5, 13.4), (52.6, 13.5))  # Berlin — outside Schwarzwald fixture
    with pytest.raises(ValueError, match="coverage"):
        pipeline.plan_route(origin="Start", destination="End", params=_params(), graph_dir=FIXTURE_GRAPH_DIR)
