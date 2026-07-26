"""graph_ops tests — shared transforms, pyrosm normalization, consolidation.

Consolidation is exercised on a small synthetic projected round-trip; the heavy
osmnx call is real (fast on a tiny graph), proving the project→consolidate→
unproject pipeline works and preserves routing attrs.
"""

import networkx as nx
import numpy as np

from bike_router import graph_ops
from bike_router.graph_ops import (
    consolidate_graph,
    contract_interstitial_nodes,
    drop_excluded_surface_edges,
    enrich_elevations,
    normalize_pyrosm_graph,
    snap_endpoints,
)
from tests.conftest import MockDEMService, make_line_graph


def test_normalize_pyrosm_graph_strips_colliding_attrs():
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=8.0, y=48.0, osmid=1, geometry="pt", tags=None, street_count=3)
    graph.add_node(2, x=8.1, y=48.0, osmid=2, geometry="pt")
    graph.add_edge(1, 2, key=0, length=100.0, osmid=1, u=1, v=2, geometry="line", osm_type="way")
    normalize_pyrosm_graph(graph)
    assert "osmid" not in graph.nodes[1] and "geometry" not in graph.nodes[1]
    assert "x" in graph.nodes[1] and "y" in graph.nodes[1]  # routing attrs kept
    edge = next(iter(graph.edges(keys=True, data=True)))[3]
    assert "osmid" not in edge and "u" not in edge  # index-colliding attrs stripped
    assert edge["geometry"] == "line"  # real edge polyline is KEPT (drives 3D path/drape)
    assert edge["length"] == 100.0


def test_drop_excluded_surface_edges_removes_tier2():
    graph = nx.MultiDiGraph()
    for node in (1, 2, 3):
        graph.add_node(node, x=float(node), y=0.0)
    graph.add_edge(1, 2, key=0, length=100.0, surface="asphalt", highway="residential")
    graph.add_edge(2, 3, key=0, length=100.0, surface="mud", highway="path")
    drop_excluded_surface_edges(graph)
    assert graph.has_edge(1, 2)
    assert not graph.has_edge(2, 3)  # mud edge removed
    assert 3 not in graph  # orphaned node pruned


def test_contract_interstitial_nodes_removes_passthrough_keeps_length():
    graph = nx.MultiDiGraph()
    for node in (1, 2, 3, 4):
        graph.add_node(node, x=float(node), y=0.0)
    for left, right, length in [(1, 2, 100.0), (2, 3, 150.0), (3, 4, 80.0)]:
        graph.add_edge(left, right, key=0, length=length, surface="asphalt", highway="residential")
        graph.add_edge(right, left, key=0, length=length, surface="asphalt", highway="residential")
    graph.add_edge(3, 4, key=1, length=80.0, surface="asphalt", highway="residential")  # node 3 branches → kept

    result = contract_interstitial_nodes(graph)
    assert 2 not in result  # interstitial node removed
    assert 1 in result and 3 in result
    assert min(d["length"] for d in result.get_edge_data(1, 3).values()) == 250.0  # summed run


def test_consolidate_graph_noop_at_zero_tolerance():
    graph = make_line_graph()
    result = consolidate_graph(graph=graph, tolerance_m=0.0)
    assert result is graph  # 0 → untouched


def test_consolidate_graph_merges_close_nodes():
    # Two clusters ~2 km apart, each a tight knot of 3 near-identical nodes (~5 m).
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    coords = {
        1: (8.0000, 48.0),
        2: (8.00005, 48.0),
        3: (8.0001, 48.0),  # cluster A
        4: (8.0200, 48.0),
        5: (8.02005, 48.0),
        6: (8.0201, 48.0),  # cluster B
    }
    for nid, (lon, lat) in coords.items():
        graph.add_node(nid, x=lon, y=lat)
    for a, b in [(3, 4), (4, 3)]:  # link the clusters
        graph.add_edge(a, b, key=0, length=1500.0, surface="asphalt", highway="residential")
    for a, b in [(1, 2), (2, 3), (2, 1), (3, 2), (4, 5), (5, 6), (5, 4), (6, 5)]:
        graph.add_edge(a, b, key=0, length=5.0, surface="asphalt", highway="residential")
    result = consolidate_graph(graph=graph, tolerance_m=25.0)
    assert result.number_of_nodes() < graph.number_of_nodes()  # tight knots merged
    assert result.graph["crs"] == "EPSG:4326" or "4326" in str(result.graph.get("crs"))


def test_enrich_elevations_populates_all_nodes():
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=0.0, y=1.0)  # 1° north
    dem = MockDEMService(base_elevation=1000.0, slope_ns_pct=10.0)
    enrich_elevations(graph=graph, dem=dem)
    assert graph.nodes[1]["elevation"] == 1000.0
    assert graph.nodes[2]["elevation"] > graph.nodes[1]["elevation"]


def test_enrich_elevations_fills_nodata_with_mean():
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=0.0, y=1.0)

    class _PartialNanDEM(MockDEMService):
        def get_elevations(self, lons, lats):  # noqa: ANN001, ANN201
            return np.array([500.0, np.nan])

    enrich_elevations(graph=graph, dem=_PartialNanDEM(base_elevation=0.0))
    assert graph.nodes[1]["elevation"] == 500.0
    assert np.isfinite(graph.nodes[2]["elevation"])


def test_snap_endpoints_maps_latlon_to_nodes(monkeypatch):
    calls = []

    def _fake_nearest(graph, X, Y):  # noqa: N803 — osmnx kwarg convention
        calls.append((X, Y))
        return 1 if X == 8.0 else 3

    monkeypatch.setattr(graph_ops.ox.distance, "nearest_nodes", _fake_nearest)
    source, target = snap_endpoints(graph=make_line_graph(), start_latlon=(48.0, 8.0), dest_latlon=(48.0, 8.02))
    assert (source, target) == (1, 3)
    assert calls[0] == (8.0, 48.0)  # X is longitude, Y is latitude
