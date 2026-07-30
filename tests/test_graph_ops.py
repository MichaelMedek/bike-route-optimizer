"""graph_ops tests — shared transforms, pyrosm normalization, consolidation.

Consolidation is exercised on a small synthetic projected round-trip; the heavy
osmnx call is real (fast on a tiny graph), proving the project→consolidate→
unproject pipeline works and preserves routing attrs.
"""

import networkx as nx
import numpy as np

from bike_router.preprocessing import graph_ops
from bike_router.preprocessing.graph_ops import (
    consolidate_graph,
    drop_disallowed_edges,
    enrich_elevations,
    normalize_pyrosm_graph,
)
from tests.conftest import (
    MockDEMService,
    make_surface_mix_graph,
    make_two_cluster_graph,
)


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


def test_drop_disallowed_edges_surface_and_highway_allowlist():
    graph = make_surface_mix_graph()
    drop_disallowed_edges(graph)
    assert graph.has_edge(1, 2)  # asphalt / residential allowlisted
    assert graph.has_edge(2, 3)  # untagged surface kept (DEFAULT_TIER)
    assert not graph.has_edge(3, 4)  # sand surface removed
    assert not graph.has_edge(4, 5)  # gravel;dirt removed (names a disallowed surface)
    assert not graph.has_edge(5, 6)  # motorway removed (disallowed highway — no bikes)
    assert 6 not in graph  # orphaned node pruned


def test_consolidate_graph_rejects_nonpositive_tolerance():
    # Strict: tolerance is always a positive build constant; a 0/negative value is a bug, not a no-op.
    import pytest

    graph = make_two_cluster_graph()
    with pytest.raises(AssertionError, match="tolerance must be positive"):
        consolidate_graph(graph=graph, tolerance_m=0.0)


def test_consolidate_graph_merges_close_nodes():
    graph = make_two_cluster_graph()
    result = consolidate_graph(graph=graph, tolerance_m=25.0)
    assert result.number_of_nodes() == 2  # each tight 3-node knot collapses to one
    assert "4326" in str(result.graph.get("crs"))  # unprojected back to lat/lon


def test_consolidate_graph_keeps_dead_ends(monkeypatch):
    # REGRESSION: consolidation MUST pass dead_ends=True. dead_ends=False pruned dead-end nodes, which
    # cascaded up low-connectivity rail branches → missing lines (Oberzent/Aglasterhausen) + orphaned
    # stations. A dead-end is a valid destination (terminus, branch halt, cul-de-sac) and may stitch to
    # a neighbour region later. osmnx only prunes in complex real topologies, so we assert the arg here.
    captured = {}
    real = graph_ops.ox.simplification.consolidate_intersections

    def _spy(G, **kwargs):  # noqa: ANN001, ANN003, N803
        captured.update(kwargs)
        return real(G, **kwargs)

    monkeypatch.setattr(graph_ops.ox.simplification, "consolidate_intersections", _spy)
    consolidate_graph(graph=make_two_cluster_graph(), tolerance_m=25.0)
    assert captured["dead_ends"] is True, "consolidation must KEEP dead-ends (valid destinations)"


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


def test_enrich_elevations_all_nodata_falls_back_to_zero():
    # Edge case: a region ENTIRELY outside DEM coverage → every sample NaN. np.nanmean would
    # return NaN + RuntimeWarning on an all-NaN slice; the fill must guard that and use 0.0 so
    # the "all elevations finite" assertion never trips (proven: numpy nanmean all-NaN → NaN).
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=0.0, y=1.0)

    class _AllNanDEM(MockDEMService):
        def get_elevations(self, lons, lats):  # noqa: ANN001, ANN201
            return np.full(len(lons), np.nan)

    enrich_elevations(graph=graph, dem=_AllNanDEM(base_elevation=0.0))
    assert graph.nodes[1]["elevation"] == 0.0 and graph.nodes[2]["elevation"] == 0.0
    assert all(np.isfinite(d["elevation"]) for _n, d in graph.nodes(data=True))
