"""rail_ascent tests — station extrema (max/min), rail BFS, grade filter, ascent-polyline assembly.

One test_<fn> per production symbol (exact-name mirror). Pure helpers use tiny synthetic inputs;
the tile-reading orchestrators run against the committed FIXTURE_GRAPH_DIR.
"""

import math

import pandas as pd
import pytest

from bike_router.core.constants import RailConfig
from bike_router.core.rail_ascent import (
    RailAscent,
    StationExtrema,
    ascent_grade,
    ascent_polyline,
    bfs_extremum_pairs,
    extremum_stations,
    interpolate_z_by_distance,
    load_stations,
    rail_adjacency,
    rail_ascents,
    station_extrema,
    station_markers,
)
from tests.conftest import FIXTURE_GRAPH_DIR

# A tiny synthetic station frame: a clear high, a clear low, and mid stations between them.
_STATIONS = pd.DataFrame(
    {
        "osmid": [1, 2, 3, 4],
        "lat": [48.0, 48.01, 48.02, 48.03],
        "lon": [8.0, 8.0, 8.0, 8.0],
        "elevation_m": [500.0, 560.0, 640.0, 720.0],
        "station_name": ["Low", "Mid1", "Mid2", "High"],
        "node_type": ["rail", "rail", "rail", "rail"],
    }
)


class TestRailAscent:
    def test_fields(self) -> None:
        # Frozen carrier: low/high names, gain, direct_km, grade, and the [lon, lat, z] polyline.
        ascent = RailAscent(
            low_name="A", high_name="B", gain_m=100.0, direct_km=2.0, grade=0.05, points=[[8.0, 48.0, 500.0]]
        )
        assert ascent.high_name == "B" and ascent.grade == 0.05 and ascent.points[0][2] == 500.0


class TestStationExtrema:
    def test_fields(self) -> None:
        # The whole 🚞 payload: green maxima, red minima, purple ascent legs.
        payload = StationExtrema(maxima=[(48.0, 8.0, 700.0, "H")], minima=[(48.1, 8.0, 500.0, "L")], ascents=[])
        assert payload.maxima[0][3] == "H" and payload.minima[0][3] == "L" and payload.ascents == []


def test_load_stations():
    # Every named rail station in the fixture (13 of them), with elevation baked in.
    stations = load_stations(graph_dir=FIXTURE_GRAPH_DIR)
    assert not stations.empty and stations["station_name"].notna().all()
    assert "Freudenstadt Stadt" in set(stations["station_name"])


def test_extremum_stations():
    # ONE pass, both directions: want_high picks the highest (sorted high-first), want_low the lowest.
    highs = extremum_stations(stations_df=_STATIONS, want_high=True)
    lows = extremum_stations(stations_df=_STATIONS, want_high=False)
    assert list(highs["station_name"]) == ["High"] and highs["elevation_m"].iloc[0] == 720.0
    assert list(lows["station_name"]) == ["Low"] and lows["elevation_m"].iloc[0] == 500.0


def test_station_markers():
    # (lat, lon, elevation_m, name) tuples — the map-layer shape.
    markers = station_markers(stations_df=_STATIONS)
    assert markers[0] == (48.0, 8.0, 500.0, "Low") and len(markers) == 4


def test_rail_adjacency():
    # Undirected adjacency from a (from_node, to_node) frame; both directed rows fold into neighbour lists.
    edges = pd.DataFrame({"from_node": [1, 2, 2, 3], "to_node": [2, 1, 3, 2]})
    adjacency = rail_adjacency(edges_df=edges)
    assert adjacency[1] == [2] and sorted(adjacency[2]) == [1, 3]


def test_bfs_extremum_pairs():
    # Line 10(max)-11-12(min)-13-14(min2): max reaches 12 (through no extremum) but NOT 14 (12 absorbs).
    adjacency = {10: [11], 11: [10, 12], 12: [11, 13], 13: [12, 14], 14: [13]}
    pairs = bfs_extremum_pairs(adjacency=adjacency, sources={10}, absorbing={10, 12, 14}, targets={12, 14})
    assert [(s, t) for s, t, _p in pairs] == [(10, 12)]
    assert pairs[0][2] == [10, 11, 12]  # reconstructed source→target path


def test_ascent_grade():
    # Direct-line rise/run: 244 m over 9.85 km ≈ 2.48%; a zero direct distance fails loud.
    assert ascent_grade(gain_m=244.0, direct_km=9.85) == pytest.approx(0.02477, abs=1e-4)
    with pytest.raises(AssertionError, match="direct-line distance must be positive"):
        ascent_grade(gain_m=100.0, direct_km=0.0)


def test_interpolate_z_by_distance():
    # Linear z by cumulative distance: endpoints exact, the equidistant midpoint halfway between.
    points = [(8.0, 48.0), (8.0, 48.01), (8.0, 48.02)]
    zs = interpolate_z_by_distance(points_2d=points, z_start=100.0, z_end=200.0)
    assert zs[0] == 100.0 and zs[-1] == 200.0 and zs[1] == pytest.approx(150.0, abs=1.0)


def test_ascent_polyline():
    # A geometry-bearing hop uses its oriented WKT; a geometry-less hop straight-lines the endpoints.
    node_seq = [(1, 8.0, 48.0, 500.0), (2, 8.01, 48.0, 560.0), (3, 8.02, 48.0, 640.0)]
    wkt_by_hop = {(1, 2): "LINESTRING (8.0 48.0, 8.005 48.0, 8.01 48.0)"}
    points = ascent_polyline(node_seq=node_seq, wkt_by_hop=wkt_by_hop)
    assert points[0][:2] == [8.0, 48.0] and points[-1][:2] == [8.02, 48.0]
    assert points[0][2] == 500.0 and points[-1][2] == 640.0  # z climbs end to end
    assert all(not math.isnan(p[2]) for p in points)


def test_rail_ascents():
    # Fixture Röt(495)→Freudenstadt Stadt(739) is a ~2.48% direct-line climb: kept at 0.02, filtered at 0.03.
    stations = load_stations(graph_dir=FIXTURE_GRAPH_DIR)
    highs = extremum_stations(stations_df=stations, want_high=True)
    lows = extremum_stations(stations_df=stations, want_high=False)
    kept = rail_ascents(
        graph_dir=FIXTURE_GRAPH_DIR, stations_df=stations, maxima_df=highs, minima_df=lows, grade_threshold=0.02
    )
    assert len(kept) == 1
    ascent = kept[0]
    assert ascent.low_name == "Röt" and ascent.high_name == "Freudenstadt Stadt"
    assert ascent.points[0][2] == 495.0 and ascent.points[-1][2] == 739.0  # low→high, z climbs
    assert (
        rail_ascents(
            graph_dir=FIXTURE_GRAPH_DIR, stations_df=stations, maxima_df=highs, minima_df=lows, grade_threshold=0.03
        )
        == []
    )
    empty = stations.iloc[0:0]
    assert (
        rail_ascents(
            graph_dir=FIXTURE_GRAPH_DIR, stations_df=stations, maxima_df=empty, minima_df=lows, grade_threshold=0.02
        )
        == []
    )


def test_station_extrema():
    # The end-to-end 🚞 payload from one scan: the fixture's sole max + min, ascent gated by the threshold.
    payload = station_extrema(graph_dir=FIXTURE_GRAPH_DIR, grade_threshold=RailConfig.MIN_ASCENT_GRADE)
    assert [m[3] for m in payload.maxima] == ["Freudenstadt Stadt"]
    assert [m[3] for m in payload.minima] == ["Röt"]
    assert payload.ascents == []  # 2.48% is below the 3% default


def test_bfs_extremum_pairs_branch():
    # A branch: max 10 reaches BOTH mins 12 and 22 (each through no other extremum) → two pairs.
    adjacency = {10: [11, 21], 11: [10, 12], 12: [11], 21: [10, 22], 22: [21]}
    pairs = bfs_extremum_pairs(adjacency=adjacency, sources={10}, absorbing={10, 12, 22}, targets={12, 22})
    assert sorted((s, t) for s, t, _p in pairs) == [(10, 12), (10, 22)]


def test_bfs_extremum_pairs_unreachable():
    # A min with no rail path from the max yields no pair (disconnected components).
    adjacency = {10: [11], 11: [10], 12: [13], 13: [12]}
    assert bfs_extremum_pairs(adjacency=adjacency, sources={10}, absorbing={10, 12}, targets={12}) == []


def test_extremum_stations_flat_region():
    # All stations within the prominence band → neither a max nor a min clears Schartenhöhe (empty).
    flat = _STATIONS.assign(elevation_m=[500.0, 501.0, 502.0, 503.0])
    assert extremum_stations(stations_df=flat, want_high=True).empty
    assert extremum_stations(stations_df=flat, want_high=False).empty


def test_ascent_polyline_all_access_hops():
    # Every hop geometry-less (station access edges) → straight endpoint segments, z still climbs.
    node_seq = [(1, 8.0, 48.0, 500.0), (2, 8.02, 48.0, 700.0)]
    points = ascent_polyline(node_seq=node_seq, wkt_by_hop={})
    assert points == [[8.0, 48.0, 500.0], [8.02, 48.0, 700.0]]


def test_rail_adjacency_bidirectional():
    # Only one directed row present still yields a directed neighbour (adjacency mirrors the frame as-is).
    edges = pd.DataFrame({"from_node": [5], "to_node": [6]})
    assert rail_adjacency(edges_df=edges) == {5: [6]}
