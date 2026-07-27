"""Builder tests — the shared pyrosm→graph builder, rail track graph, station merge.

Uses pyrosm's bundled tiny test.osm.pbf (no network) to prove the ONE shared
``_network_graph`` builder scopes bike vs rail correctly, and synthetic bike + rail
graphs for the station-merge logic so the bidirectional rail and station-access edges
are checked deterministically.
"""

from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pyrosm
import pytest
from shapely.geometry import Point

from bike_router.builder import (
    BIKE_LAYER,
    RAIL_LAYER,
    _merge_bike_rail,
    _network_graph,
    _open_osm,
    _station_entrances,
    _station_points,
    _tag_layer,
    build_layer_graph,
    dedup_by_geometry,
    reindex_region,
    remap_contiguous,
)
from bike_router.constants import Mode, NodeType, RailConfig
from bike_router.geo import haversine_distance_m
from bike_router.graph_store import graph_to_tables, read_full_graph
from tests.conftest import FIXTURE_GRAPH_DIR, MockDEMService

_TEST_PBF = Path(pyrosm.get_data("test_pbf"))


class _FakeOSM:
    """OSM stand-in serving ONLY station points via get_data_by_custom_criteria.

    The rail track graph is now passed to _merge_bike_rail directly as a MultiDiGraph, so the
    fake only needs to answer the station-points query (railway∈STATION_TAGS).
    """

    def __init__(self, stations: gpd.GeoDataFrame | None) -> None:
        self._stations = stations

    def get_data_by_custom_criteria(self, custom_filter, **_kwargs):  # noqa: ANN001, ANN003
        return self._stations


class _SpyOSM:
    """Wraps a REAL OSM (bundled pbf) and records every fetch call, to prove layer scoping.

    Delegates to the real pyrosm machinery (so to_graph works) while capturing which
    get_network / get_data_by_custom_criteria filters each builder function issues.
    """

    def __init__(self) -> None:
        self._real = _open_osm(pbf_path=_TEST_PBF, bbox=None)
        self.network_calls: list[tuple[str, object, object]] = []
        self.data_filters: list[dict] = []
        self.to_graph_bidirectional: list[bool] = []

    def get_network(self, *, network_type, custom_filter, filter_type, nodes):  # noqa: ANN001, ANN003, ANN201
        self.network_calls.append((network_type, custom_filter, filter_type))
        return self._real.get_network(
            network_type=network_type, custom_filter=custom_filter, filter_type=filter_type, nodes=nodes
        )

    def to_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.to_graph_bidirectional.append(kwargs.get("force_bidirectional"))
        return self._real.to_graph(*args, **kwargs)

    def get_data_by_custom_criteria(self, custom_filter, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.data_filters.append(custom_filter)
        return self._real.get_data_by_custom_criteria(custom_filter=custom_filter, **kwargs)


def _synth_rail_graph(lines: list[list[tuple[float, float]]]) -> nx.MultiDiGraph:
    """Synthetic rail track graph from (lat, lon) polylines: RAIL nodes, bidirectional RAIL edges.

    Stands in for what _rail_graph produces (a normalized MultiDiGraph), so _merge_bike_rail can be
    tested deterministically without a pbf. Track nodes carry NO elevation (like the real pyrosm
    graph) — it is baked later by the single enrich pass. Shared vertices coalesce by rounded coord.
    """
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    coord_id: dict[tuple[float, float], int] = {}

    def node_for(lat: float, lon: float) -> int:
        key = (round(lat, 6), round(lon, 6))
        if key not in coord_id:
            coord_id[key] = len(coord_id)
            graph.add_node(coord_id[key], x=lon, y=lat, node_type=NodeType.RAIL, station_name=None)
        return coord_id[key]

    for line in lines:
        for (lat_a, lon_a), (lat_b, lon_b) in zip(line[:-1], line[1:], strict=True):
            a, b = node_for(lat_a, lon_a), node_for(lat_b, lon_b)
            dist = haversine_distance_m(lat_a=lat_a, lon_a=lon_a, lat_b=lat_b, lon_b=lon_b)
            graph.add_edge(a, b, length=dist, mode=Mode.RAIL)
            graph.add_edge(b, a, length=dist, mode=Mode.RAIL)
    return graph


def _empty_rail_graph() -> nx.MultiDiGraph:
    """A rail graph with no track (station-only region)."""
    return nx.MultiDiGraph(crs="EPSG:4326")


def test_build_region_graph_shim_and_modes():
    # The real committed fixture is a full bike+rail region (the toy pbf has no usable rail). Every
    # node has baked elevation + coords + a node_type; every edge a mode in {bike, rail, station}.
    graph = read_full_graph(graph_dir=FIXTURE_GRAPH_DIR)
    assert graph.number_of_nodes() > 0 and graph.number_of_edges() > 0
    assert all("elevation" in d and "x" in d and "y" in d for _n, d in graph.nodes(data=True))
    assert all(d["node_type"] in {NodeType.BIKE, NodeType.RAIL} for _n, d in graph.nodes(data=True))
    modes = {d["mode"] for _u, _v, _k, d in graph.edges(keys=True, data=True)}
    assert modes == {Mode.BIKE, Mode.RAIL, Mode.STATION}  # a real region exercises all three


def test_build_region_graph_no_dangling_nodes():
    # The per-region artifact has ZERO dangling nodes — every node is in the single strongly-connected
    # component, even after station wiring (the SCC filter runs AFTER the merge in build_region_graph).
    graph = read_full_graph(graph_dir=FIXTURE_GRAPH_DIR)
    assert nx.is_strongly_connected(graph)  # no node unreachable to/from the rest


def test_remap_contiguous_then_reindex_two_regions_disjoint():
    # Regression for the node-id COLLISION bug: two regions each remapped to 0..N-1 (which would
    # collide), then reindexed with a running offset → globally disjoint, collision-free ids.
    from bike_router.graph_store import read_region_tables

    tables = read_region_tables(region_dir=FIXTURE_GRAPH_DIR)
    a_nodes, a_edges = remap_contiguous(nodes_df=tables[0], edges_df=tables[1])
    b_nodes, b_edges = remap_contiguous(nodes_df=tables[0], edges_df=tables[1])  # same region twice = worst case
    assert set(a_nodes["osmid"]) == set(b_nodes["osmid"])  # both start 0..N-1 → would collide
    b_nodes, b_edges = reindex_region(nodes_df=b_nodes, edges_df=b_edges, offset=len(a_nodes))
    assert set(a_nodes["osmid"]).isdisjoint(set(b_nodes["osmid"]))  # after offset: no overlap
    # schema preserved end-to-end
    assert {"osmid", "lat", "lon", "elevation_m", "node_type", "station_name"} == set(a_nodes.columns)
    assert {
        "from_node",
        "to_node",
        "key",
        "length_m",
        "height_diff_m",
        "surface",
        "highway",
        "mode",
        "geometry_wkt",
    } == set(a_edges.columns)


_NODE_COLS = ["osmid", "lat", "lon", "elevation_m", "node_type", "station_name"]
_EDGE_COLS = ["from_node", "to_node", "key", "length_m", "height_diff_m", "surface", "highway", "mode", "geometry_wkt"]


def _nodes(rows: list[tuple]) -> gpd.GeoDataFrame:  # noqa: ANN001
    """(osmid, lat, lon) rows → a node frame with the standard schema (all bike)."""
    return gpd.GeoDataFrame([(i, lat, lon, 0.0, "bike", None) for i, lat, lon in rows], columns=_NODE_COLS)


def _typed_nodes(rows: list[tuple]) -> gpd.GeoDataFrame:  # noqa: ANN001
    """(osmid, lat, lon, node_type) rows → a node frame; lets tests place bike + rail at one coord."""
    return gpd.GeoDataFrame([(i, lat, lon, 0.0, nt, None) for i, lat, lon, nt in rows], columns=_NODE_COLS)


def _edges(rows: list[tuple]) -> gpd.GeoDataFrame:  # noqa: ANN001
    """(from, to, key, mode, geometry_wkt) rows → an edge frame with the standard schema."""
    return gpd.GeoDataFrame(
        [(f, t, k, 1.0, 0.0, "asphalt", "residential", m, g) for f, t, k, m, g in rows], columns=_EDGE_COLS
    )


def test_dedup_no_duplicates_is_identity():
    # Distinct nodes + distinct edges → nothing dropped, ids unchanged.
    nodes = _nodes([(0, 48.0, 8.0), (1, 48.1, 8.1), (2, 48.2, 8.2)])
    edges = _edges([(0, 1, 0, "bike", None), (1, 2, 0, "bike", None)])
    kn, ke = dedup_by_geometry(nodes_df=nodes, edges_df=edges)
    assert sorted(kn["osmid"]) == [0, 1, 2]
    assert len(ke) == 2
    assert set(zip(ke["from_node"], ke["to_node"], strict=True)) == {(0, 1), (1, 2)}


def test_dedup_node_tiebreak_keeps_lowest_id_and_repoints():
    # Three coincident nodes (5, 2, 9) at the same lat/lon collapse to the LOWEST id (2);
    # an edge from the duplicate 9 is repointed onto 2.
    nodes = _nodes([(5, 48.0, 8.0), (2, 48.0, 8.0), (9, 48.0, 8.0), (1, 48.5, 8.5)])
    edges = _edges([(9, 1, 0, "bike", None)])
    kn, ke = dedup_by_geometry(nodes_df=nodes, edges_df=edges)
    assert sorted(kn["osmid"]) == [1, 2]  # the coincident trio → just node 2 survives
    assert list(ke["from_node"]) == [2] and list(ke["to_node"]) == [1]  # 9 repointed to 2


def test_dedup_parallel_edges_both_survive():
    # Two edges between the SAME nodes with DIFFERENT geometry are genuinely parallel roads.
    g1 = "LINESTRING (8.0 48.0, 8.05 48.05, 8.1 48.1)"
    g2 = "LINESTRING (8.0 48.0, 8.05 48.02, 8.1 48.1)"
    nodes = _nodes([(0, 48.0, 8.0), (1, 48.1, 8.1)])
    edges = _edges([(0, 1, 0, "bike", g1), (0, 1, 1, "bike", g2)])
    _, ke = dedup_by_geometry(nodes_df=nodes, edges_df=edges)
    assert len(ke) == 2  # distinct geometry → both kept


def test_dedup_true_duplicate_edge_dropped_after_repoint():
    # Edge 2→3 duplicates 0→1 (nodes 2,3 coincide with 0,1 AND geometry matches) → collapses.
    g = "LINESTRING (8.0 48.0, 8.1 48.1)"
    nodes = _nodes([(0, 48.0, 8.0), (1, 48.1, 8.1), (2, 48.0, 8.0), (3, 48.1, 8.1)])
    edges = _edges([(0, 1, 0, "bike", g), (2, 3, 0, "bike", g)])
    kn, ke = dedup_by_geometry(nodes_df=nodes, edges_df=edges)
    assert sorted(kn["osmid"]) == [0, 1]
    assert len(ke) == 1 and list(ke["from_node"]) == [0] and list(ke["to_node"]) == [1]


def test_dedup_null_geometry_edges_dedup_on_endpoints_and_mode():
    # Null-geometry hops (rail/station) with the SAME endpoints+mode collapse; a DIFFERENT mode
    # between the same nodes is kept (station vs rail are distinct edges).
    nodes = _nodes([(0, 48.0, 8.0), (1, 48.1, 8.1), (2, 48.0, 8.0), (3, 48.1, 8.1)])
    edges = _edges(
        [
            (0, 1, 0, "rail", None),
            (2, 3, 0, "rail", None),  # duplicate of the rail edge (endpoints coincide) → dropped
            (0, 1, 0, "station", None),  # same endpoints, different mode → kept
        ]
    )
    _, ke = dedup_by_geometry(nodes_df=nodes, edges_df=edges)
    assert len(ke) == 2
    assert set(ke["mode"]) == {"rail", "station"}


def test_dedup_beyond_precision_not_merged():
    # Coords differing beyond COORD_PRECISION (6 dp ≈ 0.1 m) are DISTINCT nodes, not merged.
    nodes = _nodes([(0, 48.0, 8.0), (1, 48.001, 8.0)])  # ~111 m apart
    edges = _edges([(0, 1, 0, "bike", None)])
    kn, ke = dedup_by_geometry(nodes_df=nodes, edges_df=edges)
    assert sorted(kn["osmid"]) == [0, 1] and len(ke) == 1


def test_dedup_bike_and_rail_at_same_coord_NOT_merged():
    # REGRESSION (the Phase-3 corruption bug): a bike node and a rail node at the SAME rounded
    # coordinate are DIFFERENT nodes and must NOT merge — node_type is part of the dedup key. Merging
    # them would leave a bike edge pointing at a rail node, breaking the type invariant.
    nodes = _typed_nodes([(0, 48.0, 8.0, "bike"), (1, 48.0, 8.0, "rail")])  # identical coord, diff type
    edges = _edges([(0, 0, 0, "bike", None), (1, 1, 0, "rail", None)])  # self-loops just to reference each
    kn, _ke = dedup_by_geometry(nodes_df=nodes, edges_df=edges)
    assert len(kn) == 2, "bike + rail at one coord must stay TWO separate nodes"
    assert set(kn["node_type"]) == {"bike", "rail"}
    assert sorted(kn["osmid"]) == [0, 1]  # neither dropped


def test_dedup_same_type_at_same_coord_still_merges():
    # The node_type key must NOT over-separate: two BIKE nodes at one coord still collapse to one.
    nodes = _typed_nodes([(0, 48.0, 8.0, "bike"), (1, 48.0, 8.0, "bike")])
    edges = _edges([(1, 0, 0, "bike", None)])
    kn, ke = dedup_by_geometry(nodes_df=nodes, edges_df=edges)
    assert list(kn["osmid"]) == [0]  # merged to the lower id
    assert list(ke["from_node"]) == [0] and list(ke["to_node"]) == [0]  # edge repointed onto survivor


def test_dedup_two_rail_at_same_coord_merge_but_not_with_bike():
    # Mixed: at one coord sit two RAIL nodes + one BIKE node. The two rail merge into one; the bike
    # stays separate. Result: exactly 2 nodes (one rail, one bike), types intact.
    nodes = _typed_nodes([(0, 48.0, 8.0, "rail"), (1, 48.0, 8.0, "rail"), (2, 48.0, 8.0, "bike")])
    edges = _edges([(1, 2, 0, "station", None)])  # rail(1)↔bike(2) station edge
    kn, ke = dedup_by_geometry(nodes_df=nodes, edges_df=edges)
    assert len(kn) == 2 and set(kn["node_type"]) == {"rail", "bike"}
    assert sorted(kn["osmid"]) == [0, 2]  # rail 1→0 (merged), bike 2 kept
    assert list(ke["from_node"]) == [0] and list(ke["to_node"]) == [2]  # station edge repointed rail→0


def test_dedup_coincident_bike_rail_keeps_edges_type_consistent_through_rebuild():
    # END-TO-END: coincident bike+rail + their edges survive dedup AND graph_from_tables' node/edge
    # type-consistency assertion (which raised "bike edge has rail endpoint" before the fix).
    from bike_router.graph_store import graph_from_tables

    nodes = _typed_nodes([(0, 48.0, 8.0, "bike"), (1, 48.0, 8.0, "rail"), (2, 48.05, 8.0, "bike")])
    edges = _edges(
        [
            (0, 2, 0, "bike", "LINESTRING (8.0 48.0, 8.0 48.05)"),  # bike↔bike
            (2, 0, 0, "bike", "LINESTRING (8.0 48.05, 8.0 48.0)"),
            (0, 1, 0, "station", None),  # bike↔rail (the coincident pair)
            (1, 0, 0, "station", None),
        ]
    )
    kn, ke = dedup_by_geometry(nodes_df=nodes, edges_df=edges)
    assert len(kn) == 3  # nothing wrongly merged
    graph = graph_from_tables(nodes_df=kn, edges_df=ke)  # asserts type consistency internally — must not raise
    assert graph.number_of_nodes() == 3 and graph.number_of_edges() == 4


def test_dedup_preserves_rail_station_types_through_graph_rebuild():
    # A realistic mixed-mode region (bike node + rail station + station/rail edges) must survive
    # dedup AND graph_from_tables' node/edge-type consistency assertions — the rail path the
    # bike-only dedup tests don't exercise. node_type is kept, edges stay type-consistent.
    from bike_router.graph_store import graph_from_tables

    nodes = gpd.GeoDataFrame(
        [
            (0, 48.0, 8.0, 0.0, "bike", None),
            (1, 48.0, 8.0009, 0.0, "rail", "Stn"),  # ~70 m E — a station node
            (2, 48.1, 8.1, 0.0, "rail", "Stn2"),
        ],
        columns=_NODE_COLS,
    )
    edges = _edges(
        [
            (0, 1, 0, "station", None),  # bike↔station access (null geometry)
            (1, 0, 0, "station", None),
            (1, 2, 0, "rail", "LINESTRING (8.0009 48.0, 8.1 48.1)"),  # rail↔rail with geometry
            (2, 1, 0, "rail", "LINESTRING (8.1 48.1, 8.0009 48.0)"),
        ]
    )
    kn, ke = dedup_by_geometry(nodes_df=nodes, edges_df=edges)
    assert sorted(kn["node_type"]) == ["bike", "rail", "rail"]  # types preserved, nothing merged
    assert sorted(ke["mode"]) == ["rail", "rail", "station", "station"]
    graph = graph_from_tables(nodes_df=kn, edges_df=ke)  # asserts node/edge-type consistency internally
    assert graph.number_of_nodes() == 3 and graph.number_of_edges() == 4


def test_tag_layer_bike_and_rail():
    # _tag_layer stamps mode + node_type for EITHER layer (single tagging source, both modes).
    g = nx.MultiDiGraph()
    g.add_node(1, x=0.0, y=0.0)
    g.add_node(2, x=1.0, y=0.0)
    g.add_edge(1, 2, key=0, length=10.0)
    _tag_layer(graph=g, mode=Mode.BIKE, node_type=NodeType.BIKE)
    assert g.get_edge_data(1, 2)[0]["mode"] == Mode.BIKE
    assert all(d["node_type"] == NodeType.BIKE for _n, d in g.nodes(data=True))
    _tag_layer(graph=g, mode=Mode.RAIL, node_type=NodeType.RAIL)  # same fn, other layer
    assert g.get_edge_data(1, 2)[0]["mode"] == Mode.RAIL
    assert all(d["node_type"] == NodeType.RAIL for _n, d in g.nodes(data=True))


def _bike_graph(nodes: list[tuple[int, float, float]], edges: list[tuple[int, int]]) -> nx.MultiDiGraph:
    """A tagged bike graph from (id, lat, lon) nodes and bidirectional (a, b) edges."""
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    for node_id, lat, lon in nodes:
        graph.add_node(node_id, x=lon, y=lat, elevation=100.0, node_type=NodeType.BIKE)
    for a, b in edges:
        graph.add_edge(a, b, length=40.0, mode=Mode.BIKE)
        graph.add_edge(b, a, length=40.0, mode=Mode.BIKE)
    return graph


# ============================= ONE shared builder: bike + rail =============================


def test_layer_specs_differ_only_in_declared_config():
    # The whole point of the refactor: bike and rail are ONE pipeline; only LayerSpec fields differ.
    assert BIKE_LAYER.custom_filter is None and BIKE_LAYER.surface_allowlist is True
    assert RAIL_LAYER.custom_filter == '["railway"~"rail"]' and RAIL_LAYER.surface_allowlist is False
    assert BIKE_LAYER.mode == Mode.BIKE and BIKE_LAYER.node_type == NodeType.BIKE
    assert RAIL_LAYER.mode == Mode.RAIL and RAIL_LAYER.node_type == NodeType.RAIL


def test_build_layer_graph_bike_is_roads_only_one_component():
    # BIKE via the shared build_layer_graph: only highway edges (zero railway), tagged BIKE, and the
    # output is exactly ONE weakly-connected component (consolidate + largest-component guarantee it).
    g = build_layer_graph(osm=_open_osm(pbf_path=_TEST_PBF, bbox=None), layer=BIKE_LAYER, tolerance_m=25.0)
    assert g.number_of_edges() > 0
    assert all(d["mode"] == Mode.BIKE for _u, _v, _k, d in g.edges(keys=True, data=True))
    assert all(d["node_type"] == NodeType.BIKE for _n, d in g.nodes(data=True))
    assert all(d.get("railway") is None for _u, _v, _k, d in g.edges(keys=True, data=True))  # no rail leaked
    assert nx.number_weakly_connected_components(g) == 1


def test_rail_layer_forms_one_connected_network_on_real_fixture():
    # The rail layer's contract — ONE connected rail network — proven on the committed real fixture
    # (the toy pbf has no usable rail). Every train reaches every other; rail nodes are RAIL-typed.
    graph = read_full_graph(graph_dir=FIXTURE_GRAPH_DIR)
    rail = nx.Graph()
    rail.add_nodes_from(n for n, d in graph.nodes(data=True) if d["node_type"] == NodeType.RAIL)
    for u, v, d in graph.edges(data=True):
        if d["mode"] == Mode.RAIL:
            rail.add_edge(u, v)
    assert rail.number_of_nodes() > 0
    assert nx.number_connected_components(rail) == 1  # one rail network, no islands


def test_filter_proof_rail_only_rail_bike_only_roads_on_raw_tags():
    # THE filter proof, on RAW tags (before tagging overwrites mode): the rail filter yields ONLY
    # railway=rail ways (never a highway road — the OOM regression); the bike preset yields ONLY
    # highway roads (never a railway). Symmetric, and inspects the actual OSM tags the query returned.
    osm = _open_osm(pbf_path=_TEST_PBF, bbox=None)
    rail_raw = _network_graph(osm=osm, custom_filter='["railway"~"rail"]', filter_type="keep")
    rail_vals = {d.get("railway") for _u, _v, _k, d in rail_raw.edges(keys=True, data=True)}
    assert rail_vals == {"rail"}, f"rail filter must return ONLY railway=rail, got {rail_vals}"
    assert all(d.get("highway") is None for _u, _v, _k, d in rail_raw.edges(keys=True, data=True))  # no roads

    bike_raw = _network_graph(osm=osm, custom_filter=None, filter_type=None)
    assert bike_raw.number_of_edges() > 0
    assert all(d.get("highway") is not None for _u, _v, _k, d in bike_raw.edges(keys=True, data=True))  # all roads
    assert all(d.get("railway") is None for _u, _v, _k, d in bike_raw.edges(keys=True, data=True))  # no rail


def test_build_layer_graph_scopes_via_spy_one_shared_path():
    # PROOF both layers go through the SAME get_network(nodes=True) path with force_bidirectional=True,
    # differing only by the filter the LayerSpec carries.
    spy = _SpyOSM()
    build_layer_graph(osm=spy, layer=BIKE_LAYER, tolerance_m=25.0)
    with pytest.raises(ValueError, match="no nodes"):  # toy-pbf rail collapses at consolidation (expected)
        build_layer_graph(osm=spy, layer=RAIL_LAYER, tolerance_m=25.0)
    assert spy.network_calls == [
        ("cycling", None, None),  # bike: preset selects ways
        ("cycling", '["railway"~"rail"]', "keep"),  # rail: bracket filter selects ways
    ]
    assert spy.to_graph_bidirectional == [True, True]  # both forced bidirectional


def test_build_layer_graph_bike_edges_all_bidirectional():
    # force_bidirectional: every bike edge has its reverse (ride any road up OR down).
    g = build_layer_graph(osm=_open_osm(pbf_path=_TEST_PBF, bbox=None), layer=BIKE_LAYER, tolerance_m=25.0)
    directed = set(g.edges())
    assert directed and all((v, u) in directed for u, v in directed)


def test_station_points_queries_only_station_tags():
    # PROOF the station extractor pulls ONLY railway in STATION_TAGS via get_data_by_custom_criteria.
    spy = _SpyOSM()
    _station_points(osm=spy)
    assert spy.data_filters == [{"railway": list(RailConfig.STATION_TAGS)}]  # exactly one query, stations only
    assert spy.network_calls == []  # stations never go through the network builder


def test_station_points_none_when_empty():
    assert _station_points(_FakeOSM(stations=None)) == []


def test_station_points_reads_names_and_centroids():
    stations = gpd.GeoDataFrame(
        {"name": ["Freudenstadt", None]},
        geometry=[Point(8.41, 48.46), Point(8.70, 48.89)],
        crs="EPSG:4326",
    )
    pts = _station_points(_FakeOSM(stations=stations))
    assert len(pts) == 2
    assert pts[0][0] == "Freudenstadt" and pts[1][0] == "station"  # unnamed → "station"
    assert pts[0][1:] == (48.46, 8.41)


# ============================= station merge (SVG contract) =============================


def _svg_invariant_holds(graph: nx.MultiDiGraph) -> bool:
    """The frozen graph_model.svg footer: BIKE↔BIKE, RAIL↔RAIL, STATION = exactly one of each."""
    for u, v, d in graph.edges(data=True):
        tu, tv = graph.nodes[u]["node_type"], graph.nodes[v]["node_type"]
        if d["mode"] == Mode.BIKE and not (tu == NodeType.BIKE and tv == NodeType.BIKE):
            return False
        if d["mode"] == Mode.RAIL and not (tu == NodeType.RAIL and tv == NodeType.RAIL):
            return False
        if d["mode"] == Mode.STATION and {tu, tv} != {NodeType.BIKE, NodeType.RAIL}:
            return False
    return True


def test_merge_wires_station_edges_and_keeps_rail_bidirectional():
    # SVG contract: independent bike graph + rail track graph merged only at stations. Station is a
    # SEPARATE rail node; bike entrance ↔ station is a STATION edge; track stays RAIL↔RAIL bidirectional.
    bike = _bike_graph([(100, 48.0, 8.001)], [])  # one bike node ~74 m E of station A
    rail = _synth_rail_graph([[(48.0, 8.00), (48.0, 8.025), (48.0, 8.05)]])  # track through A..B
    stations = gpd.GeoDataFrame({"name": ["A", "B"]}, geometry=[Point(8.00, 48.0), Point(8.05, 48.0)], crs="EPSG:4326")
    n = _merge_bike_rail(bike_graph=bike, rail_graph=rail, osm=_FakeOSM(stations=stations))
    assert n == 2  # two stations
    modes = {d["mode"] for _u, _v, d in bike.edges(data=True)}
    assert Mode.STATION in modes and Mode.RAIL in modes
    # each station is a SEPARATE rail node (negative id, RAIL type)
    assert all(bike.nodes[nid]["node_type"] == NodeType.RAIL for nid in (-1, -2))
    # every RAIL edge has its reverse (bidirectional track — trains run both ways)
    rail_edges = {(u, v) for u, v, d in bike.edges(data=True) if d["mode"] == Mode.RAIL}
    assert all((v, u) in rail_edges for u, v in rail_edges)
    # station edges bridge BIKE↔RAIL; no time baked on any edge
    assert all("rail_seconds" not in d for _u, _v, d in bike.edges(data=True))
    assert _svg_invariant_holds(bike)  # the frozen SVG's footer contract holds


def test_merge_leaves_track_and_station_nodes_for_later_elevation_bake():
    # REGRESSION: _merge_bike_rail adds rail TRACK + STATION nodes with NO elevation (baked later by
    # ONE enrich_elevations pass over the whole graph). Baking before the merge left the 4016 track
    # nodes with no "elevation" → graph_to_tables raised KeyError. Here: after merge those nodes lack
    # elevation; after enrich EVERY node has it (bike + track + station), so graph_to_tables succeeds.
    from bike_router.graph_ops import enrich_elevations

    bike = _bike_graph([(100, 48.0, 8.001)], [])
    rail = _synth_rail_graph([[(48.0, 8.00), (48.0, 8.025), (48.0, 8.05)]])
    stations = gpd.GeoDataFrame({"name": ["A"]}, geometry=[Point(8.0, 48.0)], crs="EPSG:4326")
    _merge_bike_rail(bike_graph=bike, rail_graph=rail, osm=_FakeOSM(stations=stations))
    track_and_station = [n for n, d in bike.nodes(data=True) if d["node_type"] == NodeType.RAIL]
    assert track_and_station, "expected rail track + station nodes"
    assert all("elevation" not in bike.nodes[n] for n in track_and_station)  # not baked yet
    enrich_elevations(graph=bike, dem=MockDEMService(base_elevation=100.0))
    assert all("elevation" in d for _n, d in bike.nodes(data=True))  # ALL nodes now have it
    nodes_df, _edges_df = graph_to_tables(graph=bike)  # the call that KeyError'd before
    assert len(nodes_df) == bike.number_of_nodes()


def test_merge_links_multiple_nearby_entrances():
    # A station connects to the nearest N bike nodes INSIDE the radius (SVG: several entrances per node).
    bike = _bike_graph([(201, 48.0, 8.0005), (202, 48.0, 8.0010), (203, 48.0, 8.0015)], [])  # ~37/74/111 m E
    stations = gpd.GeoDataFrame({"name": ["A"]}, geometry=[Point(8.0000, 48.0)], crs="EPSG:4326")
    _merge_bike_rail(
        bike_graph=bike,
        rail_graph=_empty_rail_graph(),
        osm=_FakeOSM(stations=stations),
    )
    linked = {u for u, v, d in bike.edges(data=True) if d["mode"] == Mode.STATION and v < 0}
    assert linked == {201, 202, 203}  # all three in-radius bike nodes


def test_merge_caps_entrances_and_respects_radius():
    # More candidates than the cap → only the nearest MAX_ENTRANCES; a far node is never linked.
    nodes = [(300 + i, 48.0, 8.0000 + i * 0.00005) for i in range(RailConfig.STATION_MAX_ENTRANCES + 5)]
    nodes.append((999, 48.0, 8.02))  # ~1.5 km east — outside the 200 m radius
    bike = _bike_graph(nodes, [])
    stations = gpd.GeoDataFrame({"name": ["A"]}, geometry=[Point(8.0000, 48.0)], crs="EPSG:4326")
    _merge_bike_rail(
        bike_graph=bike,
        rail_graph=_empty_rail_graph(),
        osm=_FakeOSM(stations=stations),
    )
    linked = {u for u, v, d in bike.edges(data=True) if d["mode"] == Mode.STATION and v < 0}
    assert len(linked) == RailConfig.STATION_MAX_ENTRANCES  # capped
    assert 999 not in linked  # out-of-radius node never linked


def test_merge_no_stations_returns_zero():
    bike = _bike_graph([(1, 48.0, 8.0)], [])
    assert _merge_bike_rail(bike_graph=bike, rail_graph=_empty_rail_graph(), osm=_FakeOSM(stations=None)) == 0


def test_station_edges_touch_bike_nodes_not_each_other():
    # Regression: two stations closer to EACH OTHER than to the single bike node must still link to
    # the BIKE node (a STATION edge is always BIKE↔RAIL — never RAIL↔RAIL between two stations).
    bike = _bike_graph([(500, 48.0, 8.0015)], [])  # only bike node, ~110 m east
    stations = gpd.GeoDataFrame(
        {"name": ["A", "B"]}, geometry=[Point(8.0000, 48.0), Point(8.0003, 48.0)], crs="EPSG:4326"
    )
    _merge_bike_rail(
        bike_graph=bike,
        rail_graph=_empty_rail_graph(),
        osm=_FakeOSM(stations=stations),
    )
    station_edges = [(u, v) for u, v, d in bike.edges(data=True) if d["mode"] == Mode.STATION]
    assert station_edges, "expected station edges within radius"
    for u, v in station_edges:
        assert 500 in (u, v), f"station edge {u}->{v} does not touch the bike node"


def test_station_entrances_empty_bike_graph_returns_none():
    # Edge case: a station with ZERO candidate bike nodes (empty arrays) must return [] cleanly.
    empty = np.array([], dtype=np.float64)
    got = _station_entrances(node_ids=np.array([], dtype=np.int64), node_lats=empty, node_lons=empty, lat=48.0, lon=8.0)
    assert got == []


def test_merge_orphaned_station_dropped_by_scc():
    # Stations with NO bike node in radius AND no track get zero edges → isolated island. The SCC
    # filter (run after merge in build_region_graph) drops them, leaving zero dangling rail nodes.
    bike = _bike_graph([(0, 48.0, 8.50), (1, 48.0, 8.5001), (2, 48.0005, 8.50005)], [(0, 1), (1, 2), (0, 2)])
    stations = gpd.GeoDataFrame({"name": ["A", "B"]}, geometry=[Point(8.0, 48.0), Point(8.05, 48.0)], crs="EPSG:4326")
    _merge_bike_rail(
        bike_graph=bike,
        rail_graph=_empty_rail_graph(),
        osm=_FakeOSM(stations=stations),
    )
    assert not [u for u, v, d in bike.edges(data=True) if d["mode"] == Mode.STATION]  # no access edges
    survivors = ox.truncate.largest_component(bike, strongly=True)
    assert all(d["node_type"] == NodeType.BIKE for _n, d in survivors.nodes(data=True))  # island dropped


def test_merge_passthrough_station_without_road_access_is_kept():
    # LEGITIMATE (user's rule): a mid-chain station B (A—B—C) where the train stops but NO road is
    # near. B has zero station edges yet MUST survive — reachable bike→A→(rail)→B→(rail)→C→bike via
    # bidirectional track. Only a rail island touching NO road ANYWHERE is dropped.
    bike = _bike_graph([(1000, 48.0, 8.0005), (1001, 48.0, 8.1005)], [(1000, 1001)])  # bike near A and C only
    # continuous track through A(8.00) B(8.05) C(8.10); B has no bike node nearby
    rail = _synth_rail_graph([[(48.0, 8.00), (48.0, 8.05), (48.0, 8.10)]])
    stations = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"]}, geometry=[Point(8.0, 48.0), Point(8.05, 48.0), Point(8.10, 48.0)], crs="EPSG:4326"
    )
    _merge_bike_rail(bike_graph=bike, rail_graph=rail, osm=_FakeOSM(stations=stations))
    # Only A and C got bike access; B is a genuine no-road pass-through stop.
    access = {
        bike.nodes[u if u < 0 else v]["station_name"] for u, v, d in bike.edges(data=True) if d["mode"] == Mode.STATION
    }
    assert access == {"A", "C"}
    survivors = ox.truncate.largest_component(bike, strongly=True)
    survivor_stations = {d["station_name"] for _n, d in survivors.nodes(data=True) if d["node_type"] == NodeType.RAIL}
    assert {"A", "B", "C"} <= survivor_stations  # pass-through B KEPT (bidirectional rail anchors it)
    assert nx.is_strongly_connected(survivors)  # whole graph is ONE connected component
    assert _svg_invariant_holds(survivors)
