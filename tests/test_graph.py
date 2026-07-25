"""Graph enrichment tests — enrich_elevations against the mock DEM (no network).

build_bike_graph / snap_endpoints wrap Overpass + OSMnx, so we mock the osmnx
calls to cover our wrapping logic without hitting the network.
"""

import networkx as nx
import numpy as np

from bike_router import graph as graph_mod
from bike_router.graph import (
    _contract_interstitial_nodes,
    _drop_excluded_surface_edges,
    build_bike_graph,
    enrich_elevations,
    snap_endpoints,
)
from tests.conftest import MockDEMService, make_line_graph


def test_drop_excluded_surface_edges_removes_tier2():
    # a paved edge and a mud (tier-2) edge; only the mud edge is dropped.
    graph = nx.MultiDiGraph()
    for node in (1, 2, 3):
        graph.add_node(node, x=float(node), y=0.0)
    graph.add_edge(1, 2, key=0, length=100.0, surface="asphalt", highway="residential")
    graph.add_edge(2, 3, key=0, length=100.0, surface="mud", highway="path")
    _drop_excluded_surface_edges(graph)
    assert graph.has_edge(1, 2)
    assert not graph.has_edge(2, 3)  # mud edge removed
    assert 3 not in graph  # orphaned node pruned


def test_contract_interstitial_nodes_removes_passthrough_keeps_length():
    # a→b→c chain (b is a degree-2 pass-through) with a real intersection at c→d.
    graph = nx.MultiDiGraph()
    for node in (1, 2, 3, 4):
        graph.add_node(node, x=float(node), y=0.0)
    for left, right, length in [(1, 2, 100.0), (2, 3, 150.0), (3, 4, 80.0)]:
        graph.add_edge(left, right, key=0, length=length, surface="asphalt", highway="residential")
        graph.add_edge(right, left, key=0, length=length, surface="asphalt", highway="residential")
    # node 3 also branches to 4 so it is NOT a pass-through; only node 2 is.
    graph.add_edge(3, 4, key=1, length=80.0, surface="asphalt", highway="residential")

    result = _contract_interstitial_nodes(graph)
    assert 2 not in result  # interstitial node removed
    assert 1 in result and 3 in result  # endpoints of the contracted run kept
    # the new 1↔3 edge carries the summed run length (100 + 150)
    assert min(d["length"] for d in result.get_edge_data(1, 3).values()) == 250.0


def test_contract_interstitial_preserves_dead_end():
    # 1↔2 with 2 a dead-end (degree 1) → nothing to contract
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=1.0, y=0.0)
    graph.add_edge(1, 2, key=0, length=100.0)
    graph.add_edge(2, 1, key=0, length=100.0)
    result = _contract_interstitial_nodes(graph)
    assert set(result.nodes) == {1, 2}


def test_contract_reports_progress():
    graph = make_line_graph()
    seen: list[tuple[int, int]] = []
    _contract_interstitial_nodes(graph, progress=lambda done, total: seen.append((done, total)))
    assert seen  # progress was driven
    assert seen[-1][0] == seen[-1][1]  # ends at done == total


def _bare_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=0.0, y=1.0)  # 1° north
    return graph


def test_enrich_elevations_populates_all_nodes():
    graph = _bare_graph()
    dem = MockDEMService(base_elevation=1000.0, slope_ns_pct=10.0, slope_ew_pct=0.0)
    enrich_elevations(graph=graph, dem=dem)
    assert graph.nodes[1]["elevation"] == 1000.0
    assert graph.nodes[2]["elevation"] > graph.nodes[1]["elevation"]  # 1° north on 10% slope


def test_enrich_elevations_fills_nodata_with_mean():
    graph = _bare_graph()

    class _PartialNanDEM(MockDEMService):
        def get_elevations(self, lons, lats):  # noqa: ANN001, ANN201
            return np.array([500.0, np.nan])

    dem = _PartialNanDEM(base_elevation=0.0)
    enrich_elevations(graph=graph, dem=dem)
    assert graph.nodes[1]["elevation"] == 500.0
    assert np.isfinite(graph.nodes[2]["elevation"])  # nodata filled, never NaN


def test_build_bike_graph_builds_once_then_caches(tmp_path, monkeypatch):
    from shapely.geometry import box

    monkeypatch.setattr(graph_mod.OutputConfig, "CACHE_DIR", tmp_path)
    core = make_line_graph()
    calls = {"n": 0}

    def _from_polygon(polygon, network_type, simplify, retain_all):
        calls["n"] += 1
        assert simplify is False, "must download UN-simplified, then contract in memory"
        assert retain_all is True, "must skip OSMnx's internal largest_component copy"
        return make_line_graph()

    monkeypatch.setattr(graph_mod.ox, "graph_from_polygon", _from_polygon)
    monkeypatch.setattr(graph_mod, "_drop_excluded_surface_edges", lambda graph: None)
    monkeypatch.setattr(graph_mod, "_contract_interstitial_nodes", lambda graph, progress: make_line_graph())
    monkeypatch.setattr(graph_mod.ox.truncate, "largest_component", lambda graph, strongly: core)

    corridor = box(8.0, 48.0, 8.1, 48.1)
    # first call: cache miss → downloads once, builds, caches
    result, raw_count = build_bike_graph(polygon=corridor)
    assert result.number_of_nodes() == core.number_of_nodes()
    assert raw_count == core.number_of_nodes()
    assert calls["n"] == 1

    # second call: cache hit → no further download, same raw_count restored
    cached, cached_count = build_bike_graph(polygon=corridor)
    assert calls["n"] == 1  # NOT downloaded again
    assert cached_count == raw_count
    assert cached.number_of_nodes() == core.number_of_nodes()


def test_snap_endpoints_maps_latlon_to_nodes(monkeypatch):
    calls = []

    def _fake_nearest(graph, X, Y):  # noqa: N803 — osmnx kwarg convention
        calls.append((X, Y))
        return 1 if X == 8.0 else 3

    monkeypatch.setattr(graph_mod.ox.distance, "nearest_nodes", _fake_nearest)
    source, target = snap_endpoints(graph=make_line_graph(), start_latlon=(48.0, 8.0), dest_latlon=(48.0, 8.02))
    assert (source, target) == (1, 3)
    assert calls[0] == (8.0, 48.0)  # X is longitude, Y is latitude
