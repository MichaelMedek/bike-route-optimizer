"""graph_ops tests — shared transforms, pyrosm normalization, consolidation, elevation bake.

One test_<fn> per production symbol (exact-name mirror). Consolidation runs the real (fast on a
tiny graph) osmnx project→consolidate→unproject; elevation is sampled via a duck-typed MockDEM.
"""

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from bike_router.preprocessing import graph_ops
from bike_router.preprocessing.graph_ops import (
    _fill_nan_with_mean,
    bake_edge_geometry_elevations,
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


def test_normalize_pyrosm_graph():
    # Strips index-colliding node/edge attrs (osmid/u/v/geometry-on-node) but KEEPS routing attrs
    # (x/y/length) and the real edge polyline (drives the 3D path/drape).
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


def test_drop_disallowed_edges():
    # Drops edges naming a disallowed surface/highway (sand, gravel;dirt, motorway); keeps allowlisted
    # + untagged (DEFAULT_TIER); prunes the node left orphaned by the drop.
    graph = make_surface_mix_graph()
    drop_disallowed_edges(graph)
    assert graph.has_edge(1, 2)  # asphalt / residential allowlisted
    assert graph.has_edge(2, 3)  # untagged surface kept (DEFAULT_TIER)
    assert not graph.has_edge(3, 4)  # sand surface removed
    assert not graph.has_edge(4, 5)  # gravel;dirt removed (names a disallowed surface)
    assert not graph.has_edge(5, 6)  # motorway removed (disallowed highway — no bikes)
    assert 6 not in graph  # orphaned node pruned


def test_consolidate_graph(monkeypatch):
    # Merges within-tolerance node knots, unprojects back to lat/lon, and MUST pass dead_ends=True
    # (a dead-end is a valid terminus/branch halt); a non-positive tolerance is a bug and fails loud.
    result = consolidate_graph(graph=make_two_cluster_graph(), tolerance_m=25.0)
    assert result.number_of_nodes() == 2  # each tight 3-node knot collapses to one
    assert "4326" in str(result.graph.get("crs"))  # unprojected back to lat/lon

    with pytest.raises(AssertionError, match="tolerance must be positive"):
        consolidate_graph(graph=make_two_cluster_graph(), tolerance_m=0.0)

    # REGRESSION: dead_ends=False pruned dead-end nodes → cascaded up rail branches (missing lines +
    # orphaned stations). A dead-end is a valid destination and may stitch to a neighbour region later.
    captured = {}
    real = graph_ops.ox.simplification.consolidate_intersections

    def _spy(G, **kwargs):  # noqa: ANN001, ANN003, N803
        captured.update(kwargs)
        return real(G, **kwargs)

    monkeypatch.setattr(graph_ops.ox.simplification, "consolidate_intersections", _spy)
    consolidate_graph(graph=make_two_cluster_graph(), tolerance_m=25.0)
    assert captured["dead_ends"] is True, "consolidation must KEEP dead-ends (valid destinations)"


def test_fill_nan_with_mean():
    # Neutral-fills NaN samples with the finite mean, returning (filled, nan_count); an all-NaN slice
    # falls back to 0.0 (numpy nanmean would return NaN); no NaN → passthrough with count 0.
    filled, count = _fill_nan_with_mean(values=np.array([500.0, np.nan, 700.0]))
    assert count == 1 and filled[1] == pytest.approx(600.0)  # mean of the two finite values
    all_nan, count = _fill_nan_with_mean(values=np.array([np.nan, np.nan]))
    assert count == 2 and list(all_nan) == [0.0, 0.0]  # all-NaN → 0.0, never NaN
    clean, count = _fill_nan_with_mean(values=np.array([1.0, 2.0]))
    assert count == 0 and list(clean) == [1.0, 2.0]  # nothing to fill


def test_enrich_elevations():
    # Attaches a finite elevation to EVERY node from one bulk DEM sample; a partial-NaN region
    # neutral-fills the gaps; an entirely-out-of-coverage region falls back to 0.0 (never NaN).
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=0.0, y=1.0)  # 1° north
    enrich_elevations(graph=graph, dem=MockDEMService(base_elevation=1000.0, slope_ns_pct=10.0))
    assert graph.nodes[1]["elevation"] == 1000.0
    assert graph.nodes[2]["elevation"] > graph.nodes[1]["elevation"]  # slope rises northward

    class _PartialNanDEM(MockDEMService):
        def get_elevations(self, lons, lats):  # noqa: ANN001, ANN201
            return np.array([500.0, np.nan])

    partial = nx.MultiDiGraph()
    partial.add_node(1, x=0.0, y=0.0)
    partial.add_node(2, x=0.0, y=1.0)
    enrich_elevations(graph=partial, dem=_PartialNanDEM(base_elevation=0.0))
    assert partial.nodes[1]["elevation"] == 500.0 and np.isfinite(partial.nodes[2]["elevation"])

    class _AllNanDEM(MockDEMService):
        def get_elevations(self, lons, lats):  # noqa: ANN001, ANN201
            return np.full(len(lons), np.nan)

    barren = nx.MultiDiGraph()
    barren.add_node(1, x=0.0, y=0.0)
    barren.add_node(2, x=0.0, y=1.0)
    enrich_elevations(graph=barren, dem=_AllNanDEM(base_elevation=0.0))
    assert all(d["elevation"] == 0.0 and np.isfinite(d["elevation"]) for _n, d in barren.nodes(data=True))


def test_bake_edge_geometry_elevations():
    # Replaces each edge's 2D polyline with a 3D one (lon, lat, elev) sampled at every vertex; a
    # geometry-less edge is left untouched; an empty graph is a no-op.
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=8.0, y=48.0)
    graph.add_node(2, x=8.02, y=48.0)
    graph.add_edge(1, 2, key=0, length=1500.0, geometry=LineString([(8.0, 48.0), (8.01, 48.0), (8.02, 48.0)]))
    graph.add_edge(1, 2, key=1, length=1500.0, geometry=None)  # straight hop, no polyline
    bake_edge_geometry_elevations(graph=graph, dem=MockDEMService(base_elevation=300.0))
    baked = graph.get_edge_data(1, 2)[0]["geometry"]
    assert baked.has_z and all(z == pytest.approx(300.0) for _x, _y, z in baked.coords)  # 3D now
    assert graph.get_edge_data(1, 2)[1]["geometry"] is None  # geometry-less edge untouched

    empty = nx.MultiDiGraph()
    bake_edge_geometry_elevations(graph=empty, dem=MockDEMService(base_elevation=0.0))  # no-op, no raise
    assert empty.number_of_edges() == 0
