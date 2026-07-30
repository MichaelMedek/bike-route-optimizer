"""Builder tests — the shared pyrosm→graph builder, rail track graph, station merge.

Uses pyrosm's bundled tiny test.osm.pbf (no network) to prove the ONE shared
``_network_graph`` builder scopes bike vs rail correctly, and synthetic bike + rail
graphs for the station-merge logic so the bidirectional rail and station-access edges
are checked deterministically.
"""

import shutil
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pyrosm
import pytest
from shapely.geometry import Point

from bike_router.core.constants import Mode, NodeType, RailConfig
from bike_router.core.geo import haversine_distance_m
from bike_router.preprocessing.builder import (
    BIKE_LAYER,
    RAIL_LAYER,
    _merge_bike_rail,
    _nearest_tracks,
    _network_graph,
    _open_osm,
    _station_entrances,
    _station_points,
    _tag_layer,
    build_layer_graph,
    build_region_graph_clipped,
    dedup_by_geometry,
    reindex_region,
    remap_contiguous,
    stage_pbf,
)
from bike_router.preprocessing.graph_writer import graph_to_tables, read_full_graph
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
        self._real = _open_osm(pbf_path=_TEST_PBF)
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


def test_build_region_graph():
    # The real committed fixture is a full bike+rail region (the toy pbf has no usable rail). Every
    # node has baked elevation + coords + a node_type; every edge a mode in {bike, rail, station};
    # and it has ZERO dangling nodes — one strongly-connected component even after station wiring.
    graph = read_full_graph(graph_dir=FIXTURE_GRAPH_DIR)
    assert graph.number_of_nodes() > 0 and graph.number_of_edges() > 0
    assert all("elevation" in d and "x" in d and "y" in d for _n, d in graph.nodes(data=True))
    assert all(d["node_type"] in {NodeType.BIKE, NodeType.RAIL} for _n, d in graph.nodes(data=True))
    modes = {d["mode"] for _u, _v, _k, d in graph.edges(keys=True, data=True)}
    assert modes == {Mode.BIKE, Mode.RAIL, Mode.STATION}  # a real region exercises all three
    assert nx.is_strongly_connected(graph)  # no node unreachable to/from the rest


def test_remap_contiguous():
    # Regression for the node-id COLLISION bug: two regions each remapped to 0..N-1 (which would
    # collide) then reindexed with a running offset → globally disjoint ids. Schema preserved.
    from bike_router.preprocessing.graph_writer import read_region_tables

    tables = read_region_tables(region_dir=FIXTURE_GRAPH_DIR)
    a_nodes, a_edges = remap_contiguous(nodes_df=tables[0], edges_df=tables[1])
    b_nodes, b_edges = remap_contiguous(nodes_df=tables[0], edges_df=tables[1])  # same region twice = worst case
    assert set(a_nodes["osmid"]) == set(b_nodes["osmid"])  # both start 0..N-1 → would collide
    b_nodes, b_edges = reindex_region(nodes_df=b_nodes, edges_df=b_edges, offset=len(a_nodes))
    assert set(a_nodes["osmid"]).isdisjoint(set(b_nodes["osmid"]))  # after offset: no overlap
    assert {"osmid", "lat", "lon", "elevation_m", "node_type", "station_name"} == set(a_nodes.columns)
    assert set(_EDGE_COLS) == set(a_edges.columns)

    # osmnx leaves gapped 0-based ids + our station code adds negatives; remap → dense 0..N-1, and
    # edges follow the SAME mapping (-2→0, 0→1, 5→2, 40→3).
    gapped_nodes = gpd.GeoDataFrame(
        {
            "osmid": [-2, 0, 5, 40],  # gapped + negative
            "lat": [48.0, 48.1, 48.2, 48.3],
            "lon": [8.0, 8.1, 8.2, 8.3],
            "elevation_m": [0.0, 0.0, 0.0, 0.0],
            "node_type": ["rail", "bike", "bike", "bike"],
            "station_name": ["S", None, None, None],
        }
    )
    gapped_edges = gpd.GeoDataFrame({"from_node": [-2, 5], "to_node": [0, 40]})
    n2, e2 = remap_contiguous(nodes_df=gapped_nodes, edges_df=gapped_edges)
    assert sorted(n2["osmid"]) == [0, 1, 2, 3]  # dense, contiguous, no negatives
    assert n2["osmid"].max() == len(n2) - 1  # n_nodes == max_id + 1
    assert list(e2["from_node"]) == [0, 2] and list(e2["to_node"]) == [1, 3]


def test_reindex_region():
    # Shifts EVERY node id and edge endpoint by a fixed offset (so a second region can't collide);
    # nothing else changes — counts and schema intact.
    from bike_router.preprocessing.graph_writer import read_region_tables

    nodes_df, edges_df = read_region_tables(region_dir=FIXTURE_GRAPH_DIR)
    base_nodes, base_edges = remap_contiguous(nodes_df=nodes_df, edges_df=edges_df)
    offset = 10_000
    shifted_nodes, shifted_edges = reindex_region(nodes_df=base_nodes, edges_df=base_edges, offset=offset)
    assert list(shifted_nodes["osmid"]) == [i + offset for i in base_nodes["osmid"]]
    assert set(shifted_edges["from_node"]) == {i + offset for i in base_edges["from_node"]}
    assert len(shifted_edges) == len(base_edges)  # only the ids shifted


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


def test_dedup_by_geometry():
    # Distinct nodes + distinct edges → nothing dropped, ids unchanged (the identity baseline).
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
    from bike_router.preprocessing.graph_writer import graph_from_tables

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
    from bike_router.preprocessing.graph_writer import graph_from_tables

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


def test_tag_layer():
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
        graph.add_node(node_id, x=lon, y=lat, elevation=100.0, node_type=NodeType.BIKE, station_name=None)
    for a, b in edges:
        graph.add_edge(a, b, length=40.0, mode=Mode.BIKE)
        graph.add_edge(b, a, length=40.0, mode=Mode.BIKE)
    return graph


# ============================= ONE shared builder: bike + rail =============================


class TestLayerSpec:
    def test_specs_differ_only_in_declared_config(self):
        # The whole point of the refactor: bike and rail are ONE pipeline; only LayerSpec fields differ.
        assert BIKE_LAYER.custom_filter is None and BIKE_LAYER.surface_allowlist is True
        assert (
            RAIL_LAYER.custom_filter == '["railway"~"^(rail|light_rail|narrow_gauge)$"]'
            and RAIL_LAYER.surface_allowlist is False
        )
        assert BIKE_LAYER.mode == Mode.BIKE and BIKE_LAYER.node_type == NodeType.BIKE
        assert RAIL_LAYER.mode == Mode.RAIL and RAIL_LAYER.node_type == NodeType.RAIL


def test_network_graph():
    # THE filter proof, on RAW tags (before tagging overwrites mode): the rail filter yields ONLY
    # railway=rail ways (never a highway road — the OOM regression); the bike preset yields ONLY
    # highway roads (never a railway). Symmetric, inspecting the actual OSM tags the query returned.
    osm = _open_osm(pbf_path=_TEST_PBF)
    rail_raw = _network_graph(
        osm=osm, custom_filter='["railway"~"^(rail|light_rail|narrow_gauge)$"]', filter_type="keep"
    )
    rail_vals = {d.get("railway") for _u, _v, _k, d in rail_raw.edges(keys=True, data=True)}
    assert rail_vals == {"rail"}, f"rail filter must return ONLY railway=rail, got {rail_vals}"
    assert all(d.get("highway") is None for _u, _v, _k, d in rail_raw.edges(keys=True, data=True))  # no roads

    bike_raw = _network_graph(osm=osm, custom_filter=None, filter_type=None)
    assert bike_raw.number_of_edges() > 0
    assert all(d.get("highway") is not None for _u, _v, _k, d in bike_raw.edges(keys=True, data=True))  # all roads
    assert all(d.get("railway") is None for _u, _v, _k, d in bike_raw.edges(keys=True, data=True))  # no rail


def test_open_osm():
    # Opens a pbf into a pyrosm OSM handle usable by the network/data queries.
    osm = _open_osm(pbf_path=_TEST_PBF)
    assert osm is not None
    assert osm.get_network(network_type="cycling", custom_filter=None, filter_type=None, nodes=True) is not None


def test_build_layer_graph():
    # BIKE via the shared build_layer_graph: only highway edges (zero railway), tagged BIKE, ALL
    # components kept (a region is a clip); every edge bidirectional; BOTH layers go through the SAME
    # get_network(nodes=True) path with force_bidirectional=True, differing only by the LayerSpec filter.
    g = build_layer_graph(osm=_open_osm(pbf_path=_TEST_PBF), layer=BIKE_LAYER, tolerance_m=25.0)
    assert g.number_of_edges() > 0
    assert all(d["mode"] == Mode.BIKE for _u, _v, _k, d in g.edges(keys=True, data=True))
    assert all(d["node_type"] == NodeType.BIKE for _n, d in g.nodes(data=True))
    assert all(d.get("railway") is None for _u, _v, _k, d in g.edges(keys=True, data=True))  # no rail leaked
    assert nx.number_weakly_connected_components(g) >= 1  # NOT truncated per-region anymore
    directed = set(g.edges())
    assert directed and all((v, u) in directed for u, v in directed)  # every bike edge bidirectional

    spy = _SpyOSM()
    build_layer_graph(osm=spy, layer=BIKE_LAYER, tolerance_m=25.0)
    build_layer_graph(osm=spy, layer=RAIL_LAYER, tolerance_m=25.0)
    assert spy.network_calls == [
        ("cycling", None, None),  # bike: preset selects ways
        ("cycling", '["railway"~"^(rail|light_rail|narrow_gauge)$"]', "keep"),  # rail: bracket filter selects ways
    ]
    assert spy.to_graph_bidirectional == [True, True]  # both forced bidirectional


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


def test_station_points():
    # Pulls ONLY railway in STATION_TAGS via get_data_by_custom_criteria (never the network builder);
    # reads each station's name + centroid; an unnamed station falls back to "station"; empty → [].
    spy = _SpyOSM()
    _station_points(osm=spy)
    assert spy.data_filters == [{"railway": list(RailConfig.STATION_TAGS)}]  # exactly one query, stations only
    assert spy.network_calls == []  # stations never go through the network builder

    assert _station_points(_FakeOSM(stations=None)) == []
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


def test_merge_bike_rail():
    # SVG contract: independent bike graph + rail track graph merged only at stations. Station is a
    # SEPARATE rail node; bike entrance ↔ station is a STATION edge; track stays RAIL↔RAIL bidirectional.
    bike = _bike_graph([(100, 48.0, 8.001), (101, 48.0, 8.051)], [])  # a bike node near EACH station A, B
    rail = _synth_rail_graph([[(48.0, 8.00), (48.0, 8.025), (48.0, 8.05)]])  # track through A..B
    stations = gpd.GeoDataFrame({"name": ["A", "B"]}, geometry=[Point(8.00, 48.0), Point(8.05, 48.0)], crs="EPSG:4326")
    n = _merge_bike_rail(bike_graph=bike, rail_graph=rail, osm=_FakeOSM(stations=stations))
    assert n == 2  # two stations, both kept (on rail + each has a bike entrance)
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
    from bike_router.preprocessing.graph_ops import enrich_elevations

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


def test_merge_wires_station_when_track_nodes_far_but_line_near():
    # REGRESSION (63% of stations orphaned): the rail LINE passes right by the station, but its nearest
    # track NODE is >200 m away (consolidation thins track vertices). Snapping to the nearest NODE with a
    # 200 m gate wrongly leaves the station with NO rail edge → unreachable by train. Snapping to the
    # nearest EDGE (the line) must always wire it. Here track nodes are ~1.6 km apart; station sits on
    # the line ~800 m from either node — a node-snap fails, an edge-snap succeeds.
    bike = _bike_graph([(100, 48.0, 8.0)], [])  # a bike node AT the station (entrance in range)
    rail = _synth_rail_graph([[(48.0, 7.99), (48.0, 8.01)]])  # ONE straight segment, nodes ~1.6 km apart
    stations = gpd.GeoDataFrame({"name": ["Mid"]}, geometry=[Point(8.0, 48.0)], crs="EPSG:4326")  # midpoint
    _merge_bike_rail(bike_graph=bike, rail_graph=rail, osm=_FakeOSM(stations=stations))
    station_id = next(n for n, d in bike.nodes(data=True) if d.get("station_name") == "Mid")
    rail_edges_at_station = [
        (u, v) for u, v, d in bike.edges(data=True) if d["mode"] == Mode.RAIL and station_id in (u, v)
    ]
    assert rail_edges_at_station, "station on the rail line MUST get a RAIL edge even if nearest node >200 m"


def test_merge_warns_and_keeps_train_only_station_with_no_bike_entrance(caplog):
    # A station on the rail line but with NO bike node within STATION_RADIUS_M (only bike ~1.6 km away)
    # is kept as a TRAIN-ONLY stop with a WARNING — real rural halts (Langen(Han) 494 m) sit on rail with
    # the nearest mapped road 200–500 m off (OSM sparsity, not a build bug). It still gets its RAIL edge.
    import logging

    far_lon = 8.0 + 0.02  # ~1.5 km east of the station at lon 8.0
    bike = _bike_graph([(100, 48.0, far_lon)], [])
    rail = _synth_rail_graph([[(48.0, 7.99), (48.0, 8.01)]])
    stations = gpd.GeoDataFrame({"name": ["Lonely"]}, geometry=[Point(8.0, 48.0)], crs="EPSG:4326")
    with caplog.at_level(logging.WARNING, logger="bike_router.preprocessing.builder"):
        n = _merge_bike_rail(bike_graph=bike, rail_graph=rail, osm=_FakeOSM(stations=stations))
    assert n == 1  # kept
    assert any("no bike node within" in r.message for r in caplog.records)  # warned, not raised
    station_id = next(nid for nid, d in bike.nodes(data=True) if d.get("station_name") == "Lonely")
    assert [d for _u, _v, d in bike.edges(station_id, data=True) if d["mode"] == Mode.RAIL]  # on rail
    assert not [d for _u, _v, d in bike.edges(station_id, data=True) if d["mode"] == Mode.STATION]  # no entrance


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


def test_station_entrances():
    # Edge case: a station with ZERO candidate bike nodes (empty arrays) must return [] cleanly.
    empty = np.array([], dtype=np.float64)
    got = _station_entrances(node_ids=np.array([], dtype=np.int64), node_lats=empty, node_lons=empty, lat=48.0, lon=8.0)
    assert got == []


def test_merge_no_track_station_without_entrance_warns_and_keeps(caplog):
    # No rail track + a station with NO bike node in range: kept with a WARNING (train-only, and here
    # not even on track). Exercises the snap-is-None + no-entrance path — no raise.
    import logging

    bike = _bike_graph([(0, 48.0, 8.50), (1, 48.0, 8.5001), (2, 48.0005, 8.50005)], [(0, 1), (1, 2), (0, 2)])
    stations = gpd.GeoDataFrame({"name": ["Far"]}, geometry=[Point(8.0, 48.0)], crs="EPSG:4326")  # ~37 km away
    with caplog.at_level(logging.WARNING, logger="bike_router.preprocessing.builder"):
        n = _merge_bike_rail(bike_graph=bike, rail_graph=_empty_rail_graph(), osm=_FakeOSM(stations=stations))
    assert n == 1 and any("no bike node within" in r.message for r in caplog.records)


def test_merge_midline_station_without_entrance_warns_and_keeps(caplog):
    # Mid-chain station B (A—B—C on continuous track) a train passes through but NO road reaches: kept as
    # a train-only stop (on rail, no STATION edge) with a WARNING naming B — real rural halts do this.
    import logging

    bike = _bike_graph([(1000, 48.0, 8.0005), (1001, 48.0, 8.1005)], [(1000, 1001)])  # bike near A and C only
    rail = _synth_rail_graph([[(48.0, 8.00), (48.0, 8.05), (48.0, 8.10)]])  # continuous track A-B-C
    stations = gpd.GeoDataFrame(
        {"name": ["A", "B", "C"]}, geometry=[Point(8.0, 48.0), Point(8.05, 48.0), Point(8.10, 48.0)], crs="EPSG:4326"
    )
    with caplog.at_level(logging.WARNING, logger="bike_router.preprocessing.builder"):
        n = _merge_bike_rail(bike_graph=bike, rail_graph=rail, osm=_FakeOSM(stations=stations))
    assert n == 3  # all three kept (all on rail)
    assert any("'B'" in r.message and "no bike node within" in r.message for r in caplog.records)
    b_id = next(nid for nid, d in bike.nodes(data=True) if d.get("station_name") == "B")
    assert not [d for _u, _v, d in bike.edges(b_id, data=True) if d["mode"] == Mode.STATION]  # B train-only


def test_merge_drops_station_off_rail_and_counts_only_kept():
    # A station >200 m (point-to-line) from ANY track is off the routable-rail network (tram/funicular) →
    # DROPPED (not orphaned): no node, no edges, and n_wired counts only the kept on-rail station.
    bike = _bike_graph([(100, 48.0, 8.00), (101, 48.0, 8.05)], [])  # entrances for the on-rail station only
    rail = _synth_rail_graph([[(48.0, 8.00), (48.0, 8.05)]])  # track along lat 48.0
    stations = gpd.GeoDataFrame(
        {"name": ["OnRail", "OffRail"]},
        geometry=[Point(8.00, 48.0), Point(8.00, 48.5)],  # OffRail ~55 km NORTH of the line
        crs="EPSG:4326",
    )
    n = _merge_bike_rail(bike_graph=bike, rail_graph=rail, osm=_FakeOSM(stations=stations))
    assert n == 1  # only OnRail wired
    names = {d["station_name"] for _n, d in bike.nodes(data=True) if d.get("station_name")}
    assert names == {"OnRail"} and "OffRail" not in names  # OffRail dropped entirely


def test_nearest_tracks():
    # Snaps a point to its nearest track EDGE (projected graph): a point on the line midway between
    # two far nodes has line_dist≈0 but a large node_dist, and returns a real endpoint node. Full
    # offset/endpoint/vectorized cases live in TestNearestTracks below.
    import osmnx as ox

    rail = _synth_rail_graph([[(48.0, 8.00), (48.0, 8.02)]])  # ~1.5 km segment, two nodes
    proj = ox.projection.project_graph(rail)
    node, node_dist, line_dist = _nearest_tracks(
        rail_graph=rail, rail_proj=proj, lats=np.array([48.0]), lons=np.array([8.01])
    )[0]
    assert line_dist < 5.0  # on the line
    assert 600 < node_dist < 900  # ~half of ~1.5 km to the nearer endpoint
    assert node in rail.nodes


class TestNearestTracks:
    """_nearest_tracks snaps MANY points to their nearest track EDGE in ONE vectorized nearest_edges
    call, returning per point (endpoint_node, node_dist_m, line_dist_m). line_dist is the TRUE
    perpendicular distance to the rail LINE (on-network gate); node_dist is to the edge's nearer endpoint
    (wired RAIL-edge length). The query runs on a PROJECTED graph (Euclidean nearest_edges mis-picks on
    lat/lon at ~48°N). Exercised on straight segments, offsets, endpoints, and endpoint selection.
    """

    @staticmethod
    def _snap(rail, lat, lon):  # noqa: ANN001, ANN205
        import osmnx as ox

        proj = ox.projection.project_graph(rail)
        return _nearest_tracks(rail_graph=rail, rail_proj=proj, lats=np.array([lat]), lons=np.array([lon]))[0]

    def test_midpoint_of_long_segment_line_dist_near_zero(self):
        # Station on the line midway between two far-apart nodes: on the LINE (line_dist≈0) but ~half the
        # segment from either node (node_dist large) — the exact consolidation-thinning case.
        rail = _synth_rail_graph([[(48.0, 8.00), (48.0, 8.02)]])  # ~1.5 km segment, two nodes
        node, node_dist, line_dist = self._snap(rail, 48.0, 8.01)  # midpoint
        assert line_dist < 5.0  # on the line
        assert 600 < node_dist < 900  # ~half of ~1.5 km to the nearer endpoint
        assert node in rail.nodes

    def test_at_endpoint_both_distances_near_zero(self):
        rail = _synth_rail_graph([[(48.0, 8.00), (48.0, 8.02)]])
        _node, node_dist, line_dist = self._snap(rail, 48.0, 8.00)  # on node
        assert node_dist < 5.0 and line_dist < 5.0

    def test_perpendicular_offset_sets_line_dist(self):
        # A point offset NORTH of the line: line_dist = the perpendicular offset (~111 m per 0.001° lat).
        rail = _synth_rail_graph([[(48.0, 8.00), (48.0, 8.02)]])
        _node, _node_dist, line_dist = self._snap(rail, 48.001, 8.01)  # 0.001° N
        assert 100 < line_dist < 125  # ~111 m

    def test_picks_nearer_endpoint(self):
        # Station near the 8.00 end → returns that endpoint (smaller node_dist), not the far one.
        rail = _synth_rail_graph([[(48.0, 8.00), (48.0, 8.02)]])
        node, _node_dist, _line_dist = self._snap(rail, 48.0, 8.002)
        assert abs(rail.nodes[node]["x"] - 8.00) < abs(rail.nodes[node]["x"] - 8.02)

    def test_vectorized_matches_per_point(self):
        # One call for MANY points returns the SAME result as snapping each individually (correctness of
        # the vectorized path, which is ~500× faster than a per-point loop).
        import osmnx as ox

        rail = _synth_rail_graph([[(48.0, 8.00), (48.0, 8.02), (48.0, 8.04)]])
        proj = ox.projection.project_graph(rail)
        lats = np.array([48.0, 48.0, 48.001])
        lons = np.array([8.005, 8.03, 8.01])
        batch = _nearest_tracks(rail_graph=rail, rail_proj=proj, lats=lats, lons=lons)
        each = [self._snap(rail, la, lo) for la, lo in zip(lats, lons, strict=True)]
        assert [b[0] for b in batch] == [e[0] for e in each]  # same endpoint nodes


def test_stage_pbf(tmp_path):
    # bbox=None (whole region) → plain byte-for-byte copy into staging_dir, no osmium needed.
    raw = tmp_path / "raw.osm.pbf"
    raw.write_bytes(b"\x00PBF-BYTES\x01")
    staging = tmp_path / "stage"
    staging.mkdir()
    staged = stage_pbf(raw_pbf=raw, bbox=None, staging_dir=staging)
    assert staged == staging / "raw.osm.pbf"  # path derived once from the raw name
    assert staged.read_bytes() == b"\x00PBF-BYTES\x01"  # verbatim copy


def test_stage_pbf_bbox_invokes_osmium_extract(monkeypatch, tmp_path):
    # bbox → shells out to `osmium extract` writing to the SAME path it returns (no drift).
    from bike_router.preprocessing import builder

    calls = []
    monkeypatch.setattr(builder.subprocess, "run", lambda cmd, check: calls.append((cmd, check)))
    staging = tmp_path / "stage"
    staging.mkdir()
    staged = stage_pbf(raw_pbf=tmp_path / "at.osm.pbf", bbox=(9.4, 46.3, 13.85, 49.1), staging_dir=staging)
    assert staged == staging / "at.osm.pbf"
    (cmd, check) = calls[0]
    assert cmd[:3] == ["osmium", "extract", "-b"] and cmd[3] == "9.4,46.3,13.85,49.1"
    assert "complete_ways" in cmd and check is True  # reference-complete + fail-fast
    assert str(staged) in cmd  # osmium writes to exactly the returned path


def test_stage_pbf_bbox_really_restricts_output_end_to_end(tmp_path):
    # END-TO-END with REAL osmium: clipping the bundled pbf to a sub-bbox yields a physically smaller
    # pbf (fewer bytes on disk) than the whole — proof the bbox selection actually reaches disk.
    if shutil.which("osmium") is None:
        pytest.skip("osmium CLI not installed")
    # the bundled pbf spans 26.93–26.97 E, 60.52–60.54 N; clip to its western ~half
    staged = stage_pbf(raw_pbf=_TEST_PBF, bbox=(26.93, 60.52, 26.95, 60.54), staging_dir=tmp_path)
    assert 0 < staged.stat().st_size < _TEST_PBF.stat().st_size  # osmium clip really shrank the file


def test_build_region_graph_clipped(monkeypatch, tmp_path):
    # The workflow: stage_pbf (into a temp dir) → build_region_graph on the STAGED path. Verify it
    # stages with the given bbox and builds exactly what stage returned, temp dir gone afterward.
    from bike_router.preprocessing import builder

    staged_seen = {}
    fake_graph = nx.MultiDiGraph()

    def _fake_stage(*, raw_pbf, bbox, staging_dir):  # noqa: ANN001, ANN202
        staged_seen["bbox"] = bbox
        p = staging_dir / raw_pbf.name  # a real temp path inside the workflow's TemporaryDirectory
        p.write_bytes(b"clipped")
        staged_seen["dir"] = staging_dir
        return p

    def _fake_build(*, pbf_path, dem, tolerance_m):  # noqa: ANN001, ANN202
        assert pbf_path.read_bytes() == b"clipped"  # builds the STAGED file, not the raw
        return fake_graph

    monkeypatch.setattr(builder, "stage_pbf", _fake_stage)
    monkeypatch.setattr(builder, "build_region_graph", _fake_build)
    out = build_region_graph_clipped(
        raw_pbf=Path("/x/austria.osm.pbf"), dem=MockDEMService(), tolerance_m=25.0, bbox=(9.4, 46.3, 13.85, 49.1)
    )
    assert out is fake_graph
    assert staged_seen["bbox"] == (9.4, 46.3, 13.85, 49.1)  # region bbox forwarded to the clip
    assert not staged_seen["dir"].exists()  # TemporaryDirectory auto-removed after build
