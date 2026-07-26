"""End-to-end route tests against the REAL committed Schwarzwald fixture.

These exercise the full production ``plan_route`` on the prebuilt graph in
tests/fixtures/dach_graph — the ONLY data source allowed here. Nothing about
routing, the graph, elevation, or output is mocked; only ``geocode_endpoint`` is
stubbed to fixed lat/lons so the tests don't hit the Nominatim network.
"""

from pathlib import Path

import pytest

from bike_router import pipeline
from bike_router.constants import PARAM_SPECS, Mode, RoutingParams
from bike_router.errors import OutOfCoverageError
from tests.conftest import FIXTURE_GRAPH_DIR

# Two real points inside the fixture coverage (Schwarzwald, ~18 km apart, net downhill).
_SOUTH = (48.4503, 8.4608)
_NORTH = (48.5601, 8.3981)
# Baiersbronn → Freudenstadt: both inside the fixture, on the same rail line ~7 km apart.
_BAIERSBRONN = (48.5057, 8.3703)
_FREUDENSTADT = (48.4634, 8.4111)


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
    assert len(result.gmaps_urls) >= 1  # at least one pedalled leg
    assert all(u.startswith("https://www.google.com/maps/dir/?api=1") for u in result.gmaps_urls)

    # Plausible Schwarzwald route: sane distance, positive time, real elevation change.
    assert 8.0 < track.distance_km < 60.0
    assert track.duration_min > 0
    assert track.ascent_m > 0 and track.descent_m > 0
    assert len(track.points) > 50  # densified along the real 3D polyline
    assert all(p.elevation_m > 0 for p in track.points)  # baked elevations, no DEM at inference

    # Composition covers the pedalled distance by surface + road class: the surface
    # tally sums to exactly the bike-mode km (rail/station legs carry no surface).
    assert sum(result.composition.by_surface_km.values()) == pytest.approx(result.composition.by_mode_km["bike"])
    assert result.composition.by_mode_km  # at least the bike mode present


def test_e2e_flat_hater_still_routes(tmp_path: Path, monkeypatch):
    """A high uphill penalty must still produce a valid (longer/flatter) route."""
    result = _plan(monkeypatch, tmp_path, _SOUTH, _NORTH, extra_km_per_uphill_100m=50.0)
    assert result.track.distance_km > 0
    assert result.gpx_path.exists()


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


def test_e2e_baiersbronn_to_freudenstadt_takes_one_train_and_two_bike_legs(tmp_path: Path, monkeypatch):
    """With rail made cheap, the same-line pair rides exactly ONE train, split into TWO bike legs.

    Baiersbronn and Freudenstadt sit on one rail line ~7 km apart. Cheap boarding/rail sliders
    make the train worth it: the route is bike → board → ride → alight → bike, i.e. one
    contiguous rail ride bracketed by two pedalled legs — so gmaps_urls has exactly 2 entries.
    """
    result = _plan(
        monkeypatch, tmp_path, _BAIERSBRONN, _FREUDENSTADT, extra_km_per_boarding=0.2, extra_km_per_rail_km=0.05
    )
    # Exactly one train ride, and rail km actually accumulated.
    assert _rail_ride_count(result.track) == 1
    assert result.composition.by_mode_km[Mode.RAIL] > 0
    # Two pedalled legs (one before boarding, one after alighting) → two bike-only Maps URLs.
    assert len(result.gmaps_urls) == 2
    assert all(u.startswith("https://www.google.com/maps/dir/?api=1") for u in result.gmaps_urls)
    # Station-access hops appear (board + alight), but they are short (≤ radius each).
    assert result.composition.by_mode_km[Mode.STATION] > 0


def test_e2e_default_sliders_pure_bike_no_train(tmp_path: Path, monkeypatch):
    """At default sliders the same pair is cheaper by bike: no train, no station km, one leg."""
    result = _plan(monkeypatch, tmp_path, _BAIERSBRONN, _FREUDENSTADT)
    assert _rail_ride_count(result.track) == 0
    assert Mode.RAIL not in result.composition.by_mode_km
    assert Mode.STATION not in result.composition.by_mode_km  # no station touched → no cut-through
    assert len(result.gmaps_urls) == 1


def test_e2e_out_of_coverage_raises(tmp_path: Path, monkeypatch):
    """Endpoints outside the fixture bbox fail loud (OutOfCoverageError), not silently."""
    _stub_geocode(monkeypatch, (52.5, 13.4), (52.6, 13.5))  # Berlin — outside Schwarzwald fixture
    with pytest.raises(OutOfCoverageError, match="coverage"):
        pipeline.plan_route(origin="Start", destination="End", params=_params(), graph_dir=FIXTURE_GRAPH_DIR)
