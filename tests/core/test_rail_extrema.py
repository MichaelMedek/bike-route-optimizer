"""rail_extrema tests — 2-pass topological station extrema (candidacy → prominence).

One test_<fn> per production symbol (exact-name mirror). Pure helpers use tiny synthetic graphs;
load_stations/station_extrema run against the committed FIXTURE_GRAPH_DIR (+ a real-dataset e2e).
"""

import pandas as pd
import pytest

from bike_router.core.constants import GeoConfig, GraphConfig, RailConfig
from bike_router.core.rail_extrema import (
    StationExtrema,
    branch_confirms,
    extremum_candidates,
    extremum_stations,
    is_confirmed_extremum,
    load_stations,
    station_extrema,
    station_line_degrees,
    station_markers,
    station_rail_neighbors,
)
from tests.conftest import FIXTURE_GRAPH_DIR

# A synthetic line Low(500) — Mid(505) — High(700); Mid stays in-band under the 100 m prominence.
_STATIONS = pd.DataFrame(
    {
        "osmid": [1, 2, 3],
        "lat": [48.00, 48.05, 48.10],
        "lon": [8.0, 8.0, 8.0],
        "elevation_m": [500.0, 505.0, 700.0],
        "station_name": ["Low", "Mid", "High"],
        "node_type": ["rail", "rail", "rail"],
    }
)
# Station-level graph for the line + a synthetic mainline where a bottom needs a monotone up-walk.
_STATION_GRAPH = {1: {2}, 2: {1, 3}, 3: {2}}
_ELEV = {1: 500.0, 2: 505.0, 3: 700.0}


class TestStationExtrema:
    def test_fields(self) -> None:
        # The 🚞 payload: green local-max + red local-min station markers.
        payload = StationExtrema(maxima=[(48.0, 8.0, 700.0, "H")], minima=[(48.1, 8.0, 500.0, "L")])
        assert payload.maxima[0][3] == "H" and payload.minima[0][3] == "L"


def test_load_stations():
    # Every named rail station in the fixture, with elevation baked in.
    stations = load_stations(graph_dir=FIXTURE_GRAPH_DIR)
    assert not stations.empty and stations["station_name"].notna().all()
    assert "Freudenstadt Stadt" in set(stations["station_name"])


def test_station_rail_neighbors():
    # Track nodes (negative ids) are walked THROUGH, stations STOP the walk → the station-level graph.
    # Line: station 1 — track -9 — station 2 — track -8 — station 3 (both directed rows present).
    edges = pd.DataFrame(
        {
            "from_node": [1, -9, -9, 2, 2, -8, -8, 3],
            "to_node": [-9, 1, 2, -9, -8, 2, 3, -8],
        }
    )
    graph = station_rail_neighbors(edges_df=edges, station_ids={1, 2, 3})
    assert graph == {1: {2}, 2: {1, 3}, 3: {2}}


def test_extremum_candidates():
    # PASS 1, direction-only: High is a top candidate (dead-end, neighbour lower), Low a bottom candidate;
    # Mid (a junction with one lower + one higher) is neither. No prominence applied yet.
    tops = extremum_candidates(station_graph=_STATION_GRAPH, elev_by_id=_ELEV, want_high=True)
    bottoms = extremum_candidates(station_graph=_STATION_GRAPH, elev_by_id=_ELEV, want_high=False)
    assert tops == {3} and bottoms == {1}


def test_extremum_candidates_both():
    # A junction with 2 lower AND 2 higher branches is candidate for BOTH top and bottom (a saddle hub).
    graph = {0: {1, 2, 3, 4}, 1: {0}, 2: {0}, 3: {0}, 4: {0}}
    elev = {0: 500.0, 1: 400.0, 2: 400.0, 3: 600.0, 4: 600.0}
    assert 0 in extremum_candidates(station_graph=graph, elev_by_id=elev, want_high=True)
    assert 0 in extremum_candidates(station_graph=graph, elev_by_id=elev, want_high=False)


def test_extremum_candidates_needs_two():
    # A junction with only ONE lower branch (the other higher) is NOT a top candidate (needs ≥2).
    graph = {0: {1, 2}, 1: {0}, 2: {0}}
    elev = {0: 500.0, 1: 400.0, 2: 600.0}
    assert 0 not in extremum_candidates(station_graph=graph, elev_by_id=elev, want_high=True)


def test_branch_confirms():
    # A top's downhill branch confirms once it has dropped ≥100 m; the in-band Mid (−0 from a bottom's
    # view) is walked THROUGH. High→Mid→Low drops 200 m → confirms; the same branch can't confirm a bottom.
    assert branch_confirms(station_graph=_STATION_GRAPH, elev_by_id=_ELEV, source=3, first=2, want_high=True)
    assert not branch_confirms(station_graph=_STATION_GRAPH, elev_by_id=_ELEV, source=3, first=2, want_high=False)


def test_branch_confirms_walks_past_in_band():
    # A bottom's uphill branch must WALK PAST a near-elevation stop to a station ≥100 m higher.
    # 0(400) — 1(410, in-band) — 2(560, +160): the branch confirms only by continuing past 1.
    graph = {0: {1}, 1: {0, 2}, 2: {1}}
    elev = {0: 400.0, 1: 410.0, 2: 560.0}
    assert branch_confirms(station_graph=graph, elev_by_id=elev, source=0, first=1, want_high=False)


def test_is_confirmed_extremum():
    # PASS 2: a dead-end confirms on its one branch. High confirms as a top, Low as a bottom, neither inverts.
    assert is_confirmed_extremum(station_graph=_STATION_GRAPH, elev_by_id=_ELEV, source=3, want_high=True)
    assert is_confirmed_extremum(station_graph=_STATION_GRAPH, elev_by_id=_ELEV, source=1, want_high=False)
    assert not is_confirmed_extremum(station_graph=_STATION_GRAPH, elev_by_id=_ELEV, source=1, want_high=True)


def test_extremum_stations():
    # End to end on the synthetic line: High is the sole top, Low the sole bottom, Mid neither.
    highs = extremum_stations(stations_df=_STATIONS, station_graph=_STATION_GRAPH, want_high=True)
    lows = extremum_stations(stations_df=_STATIONS, station_graph=_STATION_GRAPH, want_high=False)
    assert list(highs["station_name"]) == ["High"]
    assert list(lows["station_name"]) == ["Low"]


def test_station_markers():
    # (lat, lon, elev, name) markers nudged +lat for tops, −lat for bottoms so a both-station splits.
    d_lat = RailConfig.EXTREMUM_MARKER_OFFSET_M / GeoConfig.METERS_PER_DEGREE_EQUATOR
    tops = station_markers(stations_df=_STATIONS.iloc[[2]], want_high=True)
    bottoms = station_markers(stations_df=_STATIONS.iloc[[0]], want_high=False)
    assert tops[0][0] == 48.10 + d_lat and tops[0][3] == "High"  # nudged north
    assert bottoms[0][0] == 48.00 - d_lat and bottoms[0][3] == "Low"  # nudged south


def test_station_extrema():
    # The fixture scan surfaces tops + bottoms (Freudenstadt Stadt a top, Röt a bottom).
    payload = station_extrema(graph_dir=FIXTURE_GRAPH_DIR)
    assert "Freudenstadt Stadt" in {m[3] for m in payload.maxima}
    assert "Röt" in {m[3] for m in payload.minima}
    assert payload.maxima and payload.minima


# --- FULL e2e against the real DACH dataset (skipped when only the fixture is present) ---------

# Real stations whose known topology fixes their class:
_REAL_MINIMA = ["Horb", "Scuol-Tarasp", "Calw"]
_REAL_MAXIMA = ["Freudenstadt Stadt", "Bad Wildbad Bahnhof", "Hochdorf", "St. Georgen(Schwarzw)"]
# Stations that must NOT be extrema:
_REAL_NOT_EXTREMA = ["Bondorf", "Dallau", "Delmenhorst", "Hodenhagen", "Bad Bergzabern", "Mühlen", "Nagold Stadtmitte"]


@pytest.mark.skipif(
    not (GraphConfig.GRAPH_DIR / GraphConfig.META_FILENAME).exists(),
    reason="real dataset not present in data/ (only the committed fixture is available)",
)
def test_station_extrema_real_classification() -> None:
    """FULL e2e: known summits/valleys classify right, and in-between/flat stops are NOT extrema."""
    payload = station_extrema(graph_dir=GraphConfig.GRAPH_DIR)
    minima = {m[3] for m in payload.minima}
    maxima = {m[3] for m in payload.maxima}
    extrema = minima | maxima
    assert all(name in minima for name in _REAL_MINIMA)
    assert all(name in maxima for name in _REAL_MAXIMA)
    assert not any(name in extrema for name in _REAL_NOT_EXTREMA)


@pytest.mark.skipif(
    not (GraphConfig.GRAPH_DIR / GraphConfig.META_FILENAME).exists(),
    reason="real dataset not present in data/ (only the committed fixture is available)",
)
def test_station_extrema_real_proportions() -> None:
    """FULL e2e: extrema stay SELECTIVE — ≤5% of all stations, tops and bottoms balanced within ±20%."""
    n_stations = len(load_stations(graph_dir=GraphConfig.GRAPH_DIR))
    payload = station_extrema(graph_dir=GraphConfig.GRAPH_DIR)
    n_tops, n_bottoms = len(payload.maxima), len(payload.minima)
    assert n_tops + n_bottoms <= 0.05 * n_stations, "extrema must be at most 5% of all stations"
    assert n_tops and n_bottoms, "both classes must be non-empty to compare"
    assert 1 / 1.2 <= n_bottoms / n_tops <= 1.2, "bottoms must be within ±20% of tops"


# NON-NEGOTIABLE ground-truth station line-degree
_REAL_LINE_DEGREE = {
    "Horb": 4,
    "Eutingen": 3,
    "Eutingen Nord": 3,
    "Mühlen": 2,
    "Eyach": 3,
    "Bad Wildbad Bahnhof": 1,
    "Hochdorf": 3,
    "Nagold Stadtmitte": 2,
    "Calw": 3,
    "St. Georgen(Schwarzw)": 2,
    "Bondorf": 2,
    "Scuol-Tarasp": 1,
    "Aulendorf": 4,
    "Friedrichshafen Stadt": 4,
    "Lindau-Insel": 2,
}


@pytest.mark.skipif(
    not (GraphConfig.GRAPH_DIR / GraphConfig.META_FILENAME).exists(),
    reason="real dataset not present in data/ (only the committed fixture is available)",
)
def test_station_line_degree_real() -> None:
    """FULL e2e: the resolved line-degree matches the hand-read ground truth for every known station."""
    degrees = station_line_degrees(graph_dir=GraphConfig.GRAPH_DIR)
    for name, want in _REAL_LINE_DEGREE.items():
        got = degrees[name]
        assert got == want, f"{name}: line-degree {got} != ground-truth {want}"
