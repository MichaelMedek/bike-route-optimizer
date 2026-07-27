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
from shapely.geometry import Point

from bike_router.builder import (
    _merge_bike_rail,
    _network_graph,
    _open_osm,
    _rail_graph,
    _station_entrances,
    _station_points,
    _tag_bike_defaults,
    build_region_graph,
    dedup_by_geometry,
    reindex_region,
    remap_contiguous,
)
from bike_router.constants import Mode, NodeType, RailConfig
from bike_router.geo import haversine_distance_m
from bike_router.graph_store import graph_to_tables
from tests.conftest import MockDEMService

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
    dem = MockDEMService(base_elevation=300.0, slope_ns_pct=5.0)
    graph = build_region_graph(pbf_path=_TEST_PBF, dem=dem, tolerance_m=25.0)
    assert graph.number_of_nodes() > 0 and graph.number_of_edges() > 0
    # every node has baked elevation + coords + a node_type; every edge has a mode
    assert all("elevation" in d and "x" in d and "y" in d for _n, d in graph.nodes(data=True))
    assert all(d["node_type"] in {NodeType.BIKE, NodeType.RAIL} for _n, d in graph.nodes(data=True))
    modes = {d["mode"] for _u, _v, _k, d in graph.edges(keys=True, data=True)}
    assert modes <= {Mode.BIKE, Mode.RAIL, Mode.STATION}
    assert Mode.BIKE in modes  # the bundled extract has cycling ways


def test_build_region_graph_no_dangling_nodes():
    # Requirement: the per-region artifact has ZERO dangling nodes — every node is in the single
    # strongly-connected component, even after railway wiring (the filter runs AFTER the rail merge).
    dem = MockDEMService(base_elevation=300.0, slope_ns_pct=5.0)
    graph = build_region_graph(pbf_path=_TEST_PBF, dem=dem, tolerance_m=25.0)
    assert nx.is_strongly_connected(graph)  # no node unreachable to/from the rest


def test_remap_contiguous_then_reindex_two_regions_disjoint():
    # Regression for the node-id COLLISION bug: two regions each remapped to 0..N-1 (which would
    # collide), then reindexed with a running offset → globally disjoint, collision-free ids.
    dem = MockDEMService(base_elevation=300.0)
    tables = graph_to_tables(graph=build_region_graph(pbf_path=_TEST_PBF, dem=dem, tolerance_m=25.0))
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
    """(osmid, lat, lon) rows → a node frame with the standard schema."""
    return gpd.GeoDataFrame([(i, lat, lon, 0.0, "bike", None) for i, lat, lon in rows], columns=_NODE_COLS)


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


def test_tag_bike_defaults_sets_mode_and_node_type():
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=1.0, y=0.0)
    graph.add_edge(1, 2, key=0, length=10.0)
    _tag_bike_defaults(graph=graph)
    assert graph.get_edge_data(1, 2)[0]["mode"] == Mode.BIKE
    assert all(d["node_type"] == NodeType.BIKE for _n, d in graph.nodes(data=True))


def _bike_graph(nodes: list[tuple[int, float, float]], edges: list[tuple[int, int]]) -> nx.MultiDiGraph:
    """A tagged bike graph from (id, lat, lon) nodes and bidirectional (a, b) edges."""
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    for node_id, lat, lon in nodes:
        graph.add_node(node_id, x=lon, y=lat, elevation=100.0, node_type=NodeType.BIKE)
    for a, b in edges:
        graph.add_edge(a, b, length=40.0, mode=Mode.BIKE)
        graph.add_edge(b, a, length=40.0, mode=Mode.BIKE)
    return graph


# ============================= shared-builder scoping proofs =============================


def test_network_graph_bike_and_rail_from_ONE_builder_scope_correctly():
    # THE unification proof: bike and rail come from the SAME _network_graph, only the filter differs.
    # Bike (network_type="cycling") → only highway edges, ZERO railway. Rail (bracket railway filter,
    # keep) → a graph that is NOT the whole road net. Both issued against the SAME real pbf via a spy.
    spy = _SpyOSM()
    bike = _network_graph(osm=spy, network_type="cycling", custom_filter=None, filter_type=None)
    rail = _network_graph(osm=spy, network_type="cycling", custom_filter='["railway"~"rail"]', filter_type="keep")
    # both routed through get_network(nodes=True) — one shared code path, two filters
    assert spy.network_calls == [
        ("cycling", None, None),
        ("cycling", '["railway"~"rail"]', "keep"),
    ]
    # bike side: every edge is a road (highway), none is rail
    assert all(d.get("highway") is not None for _u, _v, _k, d in bike.edges(keys=True, data=True))
    assert all(d.get("railway") is None for _u, _v, _k, d in bike.edges(keys=True, data=True))
    # rail side is a DIFFERENT, far smaller graph than the full road net (the OOM regression)
    assert rail.number_of_nodes() < bike.number_of_nodes()


def test_rail_graph_tags_rail_and_never_returns_road_net():
    # _rail_graph uses the shared builder with the rail filter, then tags RAIL. On the bundled pbf it
    # is a handful of nodes — NOT the 3.6M road net the plain-dict filter returned (the crash).
    rail = _rail_graph(osm=_open_osm(pbf_path=_TEST_PBF, bbox=None))
    assert all(d["mode"] == Mode.RAIL for _u, _v, _k, d in rail.edges(keys=True, data=True))
    assert all(d["node_type"] == NodeType.RAIL for _n, d in rail.nodes(data=True))
    bike = _network_graph(
        osm=_open_osm(pbf_path=_TEST_PBF, bbox=None), network_type="cycling", custom_filter=None, filter_type=None
    )
    assert rail.number_of_nodes() < bike.number_of_nodes()  # rail is the small layer, not the road net


def test_both_layers_forced_bidirectional():
    # Unified rule: BOTH bike and rail go through _network_graph with force_bidirectional=True → every
    # edge exists both ways (same length, opposite elevation delta). A bike may ride any road up OR
    # down; trains run both ways. pyrosm network_type="cycling" alone would honour oneway (2.5% of
    # cycling ways) and make those one-directional — force_bidirectional overrides that for both.
    spy = _SpyOSM()
    _rail_graph(osm=spy)
    assert spy.to_graph_bidirectional == [True]  # rail forced bidirectional
    spy2 = _SpyOSM()
    _network_graph(osm=spy2, network_type="cycling", custom_filter=None, filter_type=None)
    assert spy2.to_graph_bidirectional == [True]  # bike ALSO forced bidirectional (ride up or down)
    # every directed edge has its reverse (no one-way edges survive in either layer)
    bike = _network_graph(osm=spy2, network_type="cycling", custom_filter=None, filter_type=None)
    directed = set(bike.edges())
    assert all((v, u) in directed for u, v in directed), "every bike edge must be traversable both ways"


def test_bike_graph_is_roads_only_and_never_returns_rail():
    # SYMMETRIC opposite of the rail test: the bike graph (network_type="cycling") is roads ONLY —
    # every edge carries a highway tag and NONE carries a railway tag, so no rail leaks into the
    # bike layer (mirror of _rail_graph never returning the road net).
    bike = _network_graph(
        osm=_open_osm(pbf_path=_TEST_PBF, bbox=None), network_type="cycling", custom_filter=None, filter_type=None
    )
    assert bike.number_of_edges() > 0
    assert all(d.get("highway") is not None for _u, _v, _k, d in bike.edges(keys=True, data=True))  # all roads
    assert all(d.get("railway") is None for _u, _v, _k, d in bike.edges(keys=True, data=True))  # zero rail leaked in


def test_station_points_queries_only_station_tags():
    # PROOF the station extractor pulls ONLY railway∈STATION_TAGS via get_data_by_custom_criteria.
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
    from bike_router.graph_store import graph_to_tables

    bike = _bike_graph([(100, 48.0, 8.001)], [])
    rail = _synth_rail_graph([[(48.0, 8.00), (48.0, 8.025), (48.0, 8.05)]])
    stations = gpd.GeoDataFrame({"name": ["A"]}, geometry=[Point(8.0, 48.0)], crs="EPSG:4326")
    _tag_bike_defaults(graph=bike)  # bike nodes tagged BIKE (elevation still unbaked here)
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
