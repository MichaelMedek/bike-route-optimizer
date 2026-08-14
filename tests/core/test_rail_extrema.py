"""rail_extrema tests — extrema over the clean station↔station graph baked by preprocessing.

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
    key_col_prominence,
    load_stations,
    station_extrema,
    station_line_degrees,
    station_markers,
    station_track_graph,
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
_STATION_GRAPH = {1: {2}, 2: {1, 3}, 3: {2}}
_ELEV = {1: 500.0, 2: 505.0, 3: 700.0}


class TestStationExtrema:
    def test_fields(self) -> None:
        # The 🚞 payload: green local-max + red local-min station markers.
        payload = StationExtrema(maxima=[(48.0, 8.0, 700.0, "H")], minima=[(48.1, 8.0, 500.0, "L")])
        assert payload.maxima[0][3] == "H" and payload.minima[0][3] == "L"


class TestStationGraph:
    def test_fields(self) -> None:
        # The one cached read: (stations_df, name→neighbours graph, name→elevation) unpacks in order.
        result = station_track_graph(graph_dir=FIXTURE_GRAPH_DIR)
        assert result.stations_df is result[0] and result.graph is result[1] and result.elev_by_name is result[2]
        assert isinstance(result.graph, dict) and isinstance(result.elev_by_name, dict)


def test_load_stations():
    # Every named rail station in the fixture, with elevation baked in.
    stations = load_stations(graph_dir=FIXTURE_GRAPH_DIR)
    assert not stations.empty and stations["station_name"].notna().all()
    assert "Freudenstadt Stadt" in set(stations["station_name"])


def test_station_track_graph():
    # ONE cached read → (stations_df, name graph, name→elevation); the graph is symmetric + self-loop-free.
    result = station_track_graph(graph_dir=FIXTURE_GRAPH_DIR)
    stations_df, graph, elev_by_name = result
    assert not stations_df.empty and "Freudenstadt Stadt" in graph and "Freudenstadt Stadt" in elev_by_name
    for a, neighbours in graph.items():
        assert isinstance(neighbours, set) and a not in neighbours
        assert all(a in graph[b] for b in neighbours)  # symmetric


def test_station_line_degrees():
    # The line-degree is exactly the neighbour-set size of the track graph (one source of truth).
    graph = station_track_graph(graph_dir=FIXTURE_GRAPH_DIR).graph
    degrees = station_line_degrees(graph_dir=FIXTURE_GRAPH_DIR)
    assert degrees == {name: len(neighbours) for name, neighbours in graph.items()}


def test_extremum_candidates():
    # PASS 1, direction-only: High is a top candidate (dead-end, neighbour lower), Low a bottom candidate.
    tops = extremum_candidates(station_graph=_STATION_GRAPH, elev_by_id=_ELEV, want_high=True)
    bottoms = extremum_candidates(station_graph=_STATION_GRAPH, elev_by_id=_ELEV, want_high=False)
    assert tops == {3} and bottoms == {1}


def test_branch_confirms():
    # High→Mid→Low drops 200 m → confirms a top on that branch; the same branch can't confirm a bottom.
    assert branch_confirms(station_graph=_STATION_GRAPH, elev_by_id=_ELEV, source=3, first=2, want_high=True)
    assert not branch_confirms(station_graph=_STATION_GRAPH, elev_by_id=_ELEV, source=3, first=2, want_high=False)


def test_is_confirmed_extremum():
    # PASS 2: a dead-end confirms on its one branch. High confirms as a top, Low as a bottom, neither inverts.
    assert is_confirmed_extremum(station_graph=_STATION_GRAPH, elev_by_id=_ELEV, source=3, want_high=True)
    assert is_confirmed_extremum(station_graph=_STATION_GRAPH, elev_by_id=_ELEV, source=1, want_high=False)
    assert not is_confirmed_extremum(station_graph=_STATION_GRAPH, elev_by_id=_ELEV, source=1, want_high=True)


def test_key_col_prominence():
    # Low(500)—Mid(505)—High(700): High is the summit → infinite max-prominence; Low the valley →
    # infinite min-prominence. Mid escapes to lower/higher ground immediately (its key col is itself → 0).
    assert key_col_prominence(station_graph=_STATION_GRAPH, elev_by_id=_ELEV, source=3, want_high=True) == float("inf")
    assert key_col_prominence(station_graph=_STATION_GRAPH, elev_by_id=_ELEV, source=1, want_high=False) == float("inf")
    assert key_col_prominence(station_graph=_STATION_GRAPH, elev_by_id=_ELEV, source=2, want_high=True) == 0.0


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
    # The fixture scan surfaces tops + bottoms (Freudenstadt Stadt the summit, Horb the valley hub).
    payload = station_extrema(graph_dir=FIXTURE_GRAPH_DIR)
    assert "Freudenstadt Stadt" in {m[3] for m in payload.maxima}
    assert "Horb" in {m[3] for m in payload.minima}
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


# --- FORMAL INVARIANTS of the station track-graph (name -> set of directly-connected neighbour names).
# "Directly connected" = one real-track edge with NO other station between; parallel rails / switches /
# multi-track throats between the same two stations collapse to ONE edge. The final DACH graph is the input.
@pytest.fixture(scope="module")
def _real_station_graph() -> dict[str, set[str]]:
    # Read straight from the final DACH graph — the build already baked the clean station↔station edges.
    return station_track_graph(graph_dir=GraphConfig.GRAPH_DIR).graph


@pytest.mark.skipif(
    not (GraphConfig.GRAPH_DIR / GraphConfig.META_FILENAME).exists(),
    reason="real dataset not present in data/ (only the committed fixture is available)",
)
def test_station_track_graph_symmetric(_real_station_graph: dict[str, set[str]]) -> None:
    """I1 — edges are bidirectional: B in G[A] <=> A in G[B]."""
    for a, neighbours in _real_station_graph.items():
        for b in neighbours:
            assert a in _real_station_graph[b], f"asymmetric edge {a}->{b}"


@pytest.mark.skipif(
    not (GraphConfig.GRAPH_DIR / GraphConfig.META_FILENAME).exists(),
    reason="real dataset not present in data/ (only the committed fixture is available)",
)
def test_station_track_graph_no_self_loops(_real_station_graph: dict[str, set[str]]) -> None:
    """I2 — no A->A self-loop."""
    for a, neighbours in _real_station_graph.items():
        assert a not in neighbours, f"self-loop at {a}"


@pytest.mark.skipif(
    not (GraphConfig.GRAPH_DIR / GraphConfig.META_FILENAME).exists(),
    reason="real dataset not present in data/ (only the committed fixture is available)",
)
def test_station_track_graph_merged_within_threshold(_real_station_graph: dict[str, set[str]]) -> None:
    """I3 — co-located platforms within 50 m are ONE node: the split-name is absent, merged-name present."""
    assert "Eyach" in _real_station_graph  # shorter name kept
    assert "Eyach HzL" not in _real_station_graph  # merged away (32.8 m apart)


@pytest.mark.skipif(
    not (GraphConfig.GRAPH_DIR / GraphConfig.META_FILENAME).exists(),
    reason="real dataset not present in data/ (only the committed fixture is available)",
)
def test_station_track_graph_simple(_real_station_graph: dict[str, set[str]]) -> None:
    """I4 — at most ONE edge between any two stations (parallel rails/switches/throat collapse to one)."""
    for a, neighbours in _real_station_graph.items():
        assert isinstance(neighbours, set), f"{a}: neighbours must be a set (no multi-edges)"
        assert len(neighbours) == len(set(neighbours)), f"{a}: duplicate edge to same neighbour"


@pytest.mark.skipif(
    not (GraphConfig.GRAPH_DIR / GraphConfig.META_FILENAME).exists(),
    reason="real dataset not present in data/ (only the committed fixture is available)",
)
def test_station_track_graph_degree(_real_station_graph: dict[str, set[str]]) -> None:
    """I5 — the node degree (edge count) equals the hand-read ground truth exactly (I4 simple by set)."""
    for name, want in _REAL_LINE_DEGREE.items():
        assert name in _real_station_graph, f"{name} missing from station graph"
        got = len(_real_station_graph[name])
        assert got == want, f"{name}: degree {got} (neighbours={sorted(_real_station_graph[name])}) != {want}"
