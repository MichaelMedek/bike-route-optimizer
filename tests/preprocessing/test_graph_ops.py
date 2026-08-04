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
    _densify_coords,
    _fill_nan_with_mean,
    _worst_band_vertex,
    bake_edge_geometry_elevations,
    consolidate_graph,
    densify_edge_geometry,
    drop_bike_self_loops,
    drop_disallowed_edges,
    enrich_elevations,
    normalize_pyrosm_graph,
    split_bike_edges_at_extrema,
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


def test_densify_coords():
    # Inserts evenly-spaced points so no segment exceeds max_spacing; existing vertices kept, order preserved.
    # ~743 m single segment @ 100 m spacing → 8 sub-segments (7 inserts) → 9 points, all endpoints intact.
    out = _densify_coords([(8.0, 48.0), (8.01, 48.0)], max_spacing_m=100.0)
    assert out[0] == (8.0, 48.0) and out[-1] == (8.01, 48.0)  # endpoints preserved
    from bike_router.core.geo import haversine_distance_m

    gaps = [
        haversine_distance_m(lat_a=out[i][1], lon_a=out[i][0], lat_b=out[i + 1][1], lon_b=out[i + 1][0])
        for i in range(len(out) - 1)
    ]
    assert max(gaps) <= 100.0  # every sub-gap within spacing
    # an already-dense pair is left as-is (no inserts)
    assert _densify_coords([(8.0, 48.0), (8.0005, 48.0)], max_spacing_m=100.0) == [(8.0, 48.0), (8.0005, 48.0)]
    # REGRESSION: a LONG DIAGONAL segment (lon+lat both change) must stay STRICTLY under the cap despite
    # lon/lat-linear interpolation vs great-circle measurement — the 100.0m-boundary build failure.
    diag = _densify_coords([(8.0, 48.0), (8.03, 48.9)], max_spacing_m=100.0)
    dgaps = [
        haversine_distance_m(lat_a=diag[i][1], lon_a=diag[i][0], lat_b=diag[i + 1][1], lon_b=diag[i + 1][0])
        for i in range(len(diag) - 1)
    ]
    assert max(dgaps) < 100.0  # STRICT: no sub-gap lands exactly at (or above) the cap


def test_densify_edge_geometry():
    # Every edge polyline is densified so no vertex gap exceeds max_spacing; a geometry-less edge is skipped.
    from bike_router.core.constants import Schema
    from bike_router.core.geo import haversine_distance_m

    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, **{Schema.GEOMETRY: LineString([(8.0, 48.0), (8.02, 48.0)])})  # ~1486 m
    graph.add_edge(3, 4, key=0, **{Schema.GEOMETRY: None})  # skipped, no raise
    densify_edge_geometry(graph=graph, max_spacing_m=100.0)
    coords = list(graph.edges[1, 2, 0][Schema.GEOMETRY].coords)
    gaps = [
        haversine_distance_m(lat_a=coords[i][1], lon_a=coords[i][0], lat_b=coords[i + 1][1], lon_b=coords[i + 1][0])
        for i in range(len(coords) - 1)
    ]
    assert max(gaps) <= 100.0 and coords[0] == (8.0, 48.0) and coords[-1] == (8.02, 48.0)


def test_drop_bike_self_loops():
    # Removes BIKE self-loops (u==u); keeps normal bike edges AND rail self-loops. Returns #removed.
    from bike_router.core.constants import Mode, Schema

    graph = nx.MultiDiGraph()
    graph.add_edge(1, 1, key=0, **{Schema.MODE: Mode.BIKE})  # bike self-loop → dropped
    graph.add_edge(1, 2, key=0, **{Schema.MODE: Mode.BIKE})  # normal bike → kept
    graph.add_edge(3, 3, key=0, **{Schema.MODE: Mode.RAIL})  # rail self-loop → kept (harmless)
    n = drop_bike_self_loops(graph=graph)
    assert n == 1
    assert not graph.has_edge(1, 1) and graph.has_edge(1, 2) and graph.has_edge(3, 3)


def test_worst_band_vertex():
    # Returns the interior vertex farthest OUTSIDE the [endpoint z] band past the margin; None if all in.
    # Endpoints z=100,200 → band [100,200]. A midpoint at 260 is 60 m over → index 1 (margin 30).
    assert _worst_band_vertex([(0, 0, 100), (0, 0, 260), (0, 0, 200)], margin_m=30.0) == 1
    # a dip to 40 (60 m under) also flagged
    assert _worst_band_vertex([(0, 0, 100), (0, 0, 40), (0, 0, 200)], margin_m=30.0) == 1
    # within margin → None; endpoints are never chosen even if "outside" (they define the band)
    assert _worst_band_vertex([(0, 0, 100), (0, 0, 215), (0, 0, 200)], margin_m=30.0) is None
    assert _worst_band_vertex([(0, 0, 100), (0, 0, 200)], margin_m=30.0) is None  # < 3 vertices


def test_split_bike_edges_at_extrema():
    # A bike edge with a mid crest 60 m above its endpoint band is split there into two in-band sub-edges;
    # the crest becomes a new node carrying its z. A flat edge is left untouched (one edge, no new nodes).
    from shapely.geometry import LineString as LS

    from bike_router.core.constants import Mode, Schema

    graph = nx.MultiDiGraph()
    graph.add_node(1, x=8.0, y=48.0, elevation=100.0, node_type="bike", station_name=None)
    graph.add_node(2, x=8.002, y=48.0, elevation=100.0, node_type="bike", station_name=None)
    graph.add_edge(
        1,
        2,
        key=0,
        length=150.0,
        mode=Mode.BIKE,
        **{Schema.GEOMETRY: LS([(8.0, 48.0, 100.0), (8.001, 48.0, 200.0), (8.002, 48.0, 100.0)])},
    )
    next_id = split_bike_edges_at_extrema(graph=graph, margin_m=30.0, next_node_id=3)
    assert next_id == 4 and graph.number_of_nodes() == 3  # one crest node minted (id 3)
    assert graph.nodes[3]["elevation"] == pytest.approx(200.0)  # new node carries the crest z
    assert graph.number_of_edges() == 2  # edge split in two at the crest
    # every resulting bike edge is now in-band (max |z - endpoint band| ≤ margin)
    for _u, _v, d in graph.edges(data=True):
        z = [c[2] for c in d[Schema.GEOMETRY].coords]
        lo, hi = min(z[0], z[-1]), max(z[0], z[-1])
        assert max(max(zz - hi, 0.0) + max(lo - zz, 0.0) for zz in z) <= 30.0


def test_split_bike_edges_at_extrema_iterates_multiple_extrema():
    # A monotone-looking edge hiding a dip THEN a crest needs two cuts: after the first split one sub-edge
    # still holds an out-of-band extremum, so the queue re-checks and splits again. All sub-edges end in band.
    from shapely.geometry import LineString as LS

    from bike_router.core.constants import Mode, Schema

    graph = nx.MultiDiGraph()
    graph.add_node(1, x=8.0, y=48.0, elevation=0.0, node_type="bike", station_name=None)
    graph.add_node(2, x=8.004, y=48.0, elevation=300.0, node_type="bike", station_name=None)
    # endpoints 0→300 (band [0,300]); a dip to -100 and a spike to 400 are BOTH out of band by 100 m
    coords = [(8.0, 48.0, 0.0), (8.001, 48.0, -100.0), (8.002, 48.0, 150.0), (8.003, 48.0, 400.0), (8.004, 48.0, 300.0)]
    graph.add_edge(1, 2, key=0, length=300.0, mode=Mode.BIKE, **{Schema.GEOMETRY: LS(coords)})
    split_bike_edges_at_extrema(graph=graph, margin_m=30.0, next_node_id=3)
    assert graph.number_of_edges() >= 3  # dip + crest → at least two splits
    for _u, _v, d in graph.edges(data=True):
        z = [c[2] for c in d[Schema.GEOMETRY].coords]
        lo, hi = min(z[0], z[-1]), max(z[0], z[-1])
        assert max(max(zz - hi, 0.0) + max(lo - zz, 0.0) for zz in z) <= 30.0  # all in band


def test_split_bike_edges_preserves_edge_attrs():
    # Sub-edges inherit the parent's mode/surface/highway tags (only geometry + length are recomputed).
    from shapely.geometry import LineString as LS

    from bike_router.core.constants import Mode, Schema

    graph = nx.MultiDiGraph()
    graph.add_node(1, x=8.0, y=48.0, elevation=100.0, node_type="bike", station_name=None)
    graph.add_node(2, x=8.002, y=48.0, elevation=100.0, node_type="bike", station_name=None)
    geom = LS([(8.0, 48.0, 100.0), (8.001, 48.0, 200.0), (8.002, 48.0, 100.0)])
    graph.add_edge(
        1, 2, key=0, length=150.0, mode=Mode.BIKE, surface="asphalt", highway="residential", **{Schema.GEOMETRY: geom}
    )
    split_bike_edges_at_extrema(graph=graph, margin_m=30.0, next_node_id=3)
    for _u, _v, d in graph.edges(data=True):
        assert d["mode"] == Mode.BIKE and d["surface"] == "asphalt" and d["highway"] == "residential"
        assert d[Schema.LENGTH] == pytest.approx(d[Schema.GEOMETRY].length)  # length matches new geometry


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
