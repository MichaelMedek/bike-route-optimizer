"""Builder tests — pyrosm→graph shim, rail/station wiring.

Uses pyrosm's bundled tiny test.osm.pbf (no network) for the graph shim, and
synthetic inputs for the rail/station logic so the bidirectional rail edges and
station-access edges are checked deterministically.
"""

from pathlib import Path

import geopandas as gpd
import networkx as nx
import pyrosm
from shapely.geometry import LineString, MultiLineString, Point

from bike_router.builder import (
    _add_railway,
    _connect_stations_along_lines,
    _rail_lines,
    _station_points,
    _tag_bike_defaults,
    build_region_graph,
    merge_region_tables,
)
from bike_router.constants import Mode, NodeType
from bike_router.graph_store import graph_to_tables
from tests.conftest import MockDEMService

_TEST_PBF = Path(pyrosm.get_data("test_pbf"))


class _FakeOSM:
    """Minimal OSM stand-in returning fixed station/rail GeoDataFrames."""

    def __init__(self, stations: gpd.GeoDataFrame | None, rails: gpd.GeoDataFrame | None) -> None:
        self._stations = stations
        self._rails = rails

    def get_data_by_custom_criteria(self, custom_filter, **_kwargs):  # noqa: ANN001, ANN003
        return self._stations if "station" in str(custom_filter) or "halt" in str(custom_filter) else self._rails


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


def test_merge_region_tables_returns_schema_tables():
    # One region built + merged → the exact node/edge schema graph_store persists (the shim
    # from build_region_graph → graph_to_tables → merge_region_tables the build script uses).
    dem = MockDEMService(base_elevation=300.0)
    tables = graph_to_tables(graph=build_region_graph(pbf_path=_TEST_PBF, dem=dem, tolerance_m=25.0))
    nodes_df, edges_df = merge_region_tables(regions=[tables])
    assert len(nodes_df) > 0 and len(edges_df) > 0
    assert {"osmid", "lat", "lon", "elevation_m", "node_type", "station_name"} == set(nodes_df.columns)
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
    } == set(edges_df.columns)


def test_tag_bike_defaults_sets_mode_and_node_type():
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=1.0, y=0.0)
    graph.add_edge(1, 2, key=0, length=10.0)
    _tag_bike_defaults(graph=graph)
    assert graph.get_edge_data(1, 2)[0]["mode"] == Mode.BIKE
    assert all(d["node_type"] == NodeType.BIKE for _n, d in graph.nodes(data=True))


def test_rail_edges_are_bidirectional():
    # Two stations on one line get a rail edge in BOTH directions.
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(-1, x=8.00, y=48.0, elevation=200.0, node_type=NodeType.RAIL, station_name="A")
    graph.add_node(-2, x=8.05, y=48.0, elevation=500.0, node_type=NodeType.RAIL, station_name="B")
    line = [(48.0, 8.00), (48.0, 8.025), (48.0, 8.05)]  # (lat, lon) vertices near both
    station_nodes = [(-1, "A", 48.0, 8.00), (-2, "B", 48.0, 8.05)]
    _connect_stations_along_lines(graph=graph, station_nodes=station_nodes, lines=[line])
    rail = {(u, v) for u, v, d in graph.edges(data=True) if d["mode"] == Mode.RAIL}
    assert rail == {(-1, -2), (-2, -1)}  # both directions
    assert graph.get_edge_data(-1, -2)[0]["length"] > 0  # real along-track length stored
    assert graph.get_edge_data(-2, -1)[0]["length"] > 0


def test_rail_edge_stores_length_not_time():
    # Rail edges carry only length; ride time is DERIVED at route time (build_track).
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(-1, x=8.0, y=48.0, elevation=100.0, node_type=NodeType.RAIL, station_name="A")
    graph.add_node(-2, x=8.1, y=48.0, elevation=400.0, node_type=NodeType.RAIL, station_name="B")
    line = [(48.0, 8.0), (48.0, 8.1)]
    _connect_stations_along_lines(graph=graph, station_nodes=[(-1, "A", 48.0, 8.0), (-2, "B", 48.0, 8.1)], lines=[line])
    data = graph.get_edge_data(-1, -2)[0]
    assert "rail_seconds" not in data  # no time baked on the edge
    assert data["length"] > 0


def test_station_points_none_when_empty():
    assert _station_points(_FakeOSM(stations=None, rails=None)) == []


def test_station_points_reads_names_and_centroids():
    stations = gpd.GeoDataFrame(
        {"name": ["Freudenstadt", None]},
        geometry=[Point(8.41, 48.46), Point(8.70, 48.89)],
        crs="EPSG:4326",
    )
    pts = _station_points(_FakeOSM(stations=stations, rails=None))
    assert len(pts) == 2
    assert pts[0][0] == "Freudenstadt" and pts[1][0] == "station"  # unnamed → "station"
    assert pts[0][1:] == (48.46, 8.41)


def test_rail_lines_flattens_multilinestring():
    rails = gpd.GeoDataFrame(
        geometry=[
            LineString([(8.0, 48.0), (8.1, 48.0)]),
            MultiLineString([[(8.2, 48.0), (8.3, 48.0)], [(8.4, 48.0), (8.5, 48.0)]]),
        ],
        crs="EPSG:4326",
    )
    lines = _rail_lines(_FakeOSM(stations=None, rails=rails))
    assert len(lines) == 3  # 1 single + 2 parts
    assert lines[0][0] == (48.0, 8.0)  # (lat, lon) order


def test_add_railway_wires_station_edges_and_bidirectional_rail():
    # A bike graph with one node; two stations near it on one line.
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(100, x=8.001, y=48.0, elevation=210.0, node_type=NodeType.BIKE)  # ~74 m E of station A
    stations = gpd.GeoDataFrame(
        {"name": ["A", "B"]},
        geometry=[Point(8.00, 48.0), Point(8.05, 48.0)],
        crs="EPSG:4326",
    )
    rails = gpd.GeoDataFrame(geometry=[LineString([(8.00, 48.0), (8.025, 48.0), (8.05, 48.0)])], crs="EPSG:4326")
    osm = _FakeOSM(stations=stations, rails=rails)
    dem = MockDEMService(base_elevation=200.0, slope_ew_pct=-20.0)  # elevation rises with lon (east)
    n = _add_railway(graph=graph, osm=osm, dem=dem)
    assert n == 2  # two stations added
    modes = [d["mode"] for _u, _v, d in graph.edges(data=True)]
    assert Mode.STATION in modes and Mode.RAIL in modes
    # each station is a SEPARATE rail node (never the bike node)
    assert all(graph.nodes[n]["node_type"] == NodeType.RAIL for n in (-1, -2))
    # rail runs both directions between the two stations
    rail = {(u, v) for u, v, d in graph.edges(data=True) if d["mode"] == Mode.RAIL}
    assert len(rail) == 2
    (u, v) = next(iter(rail))
    assert (v, u) in rail  # every rail edge has its reverse
    # station edges link the bike node to a rail node both ways (boarding wait is applied at
    # route time from node_type, not stored on the edge).
    station_targets = {(u, v) for u, v, d in graph.edges(data=True) if d["mode"] == Mode.STATION}
    assert any(graph.nodes[v]["node_type"] == NodeType.RAIL for _u, v in station_targets)  # a leg enters a station
    assert all("rail_seconds" not in d for _u, _v, d in graph.edges(data=True))  # no time baked anywhere


def test_add_railway_links_multiple_nearby_entrances():
    # A station must connect to the nearest N bike nodes INSIDE the radius (not just the
    # closest one) so the station is reachable wherever a ride ends nearby. Here 3 bike
    # nodes sit within ~150 m of the station → all three become bidirectional entrances.
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    for node_id, lon in ((201, 8.0005), (202, 8.0010), (203, 8.0015)):  # ~37/74/111 m east
        graph.add_node(node_id, x=lon, y=48.0, elevation=100.0, node_type=NodeType.BIKE)
    stations = gpd.GeoDataFrame({"name": ["A"]}, geometry=[Point(8.0000, 48.0)], crs="EPSG:4326")
    _add_railway(graph=graph, osm=_FakeOSM(stations=stations, rails=None), dem=MockDEMService(base_elevation=100.0))
    linked = {u for u, v, d in graph.edges(data=True) if d["mode"] == Mode.STATION and v < 0}
    assert linked == {201, 202, 203}  # all three in-radius bike nodes, both directions


def test_add_railway_caps_entrances_and_respects_radius():
    # More candidates than the cap → only the nearest MAX_ENTRANCES are linked; a far node
    # outside the radius is never linked regardless of the cap.
    from bike_router.constants import RailConfig

    graph = nx.MultiDiGraph(crs="EPSG:4326")
    for i in range(RailConfig.STATION_MAX_ENTRANCES + 5):  # more than the cap, all in-radius
        graph.add_node(300 + i, x=8.0000 + i * 0.00005, y=48.0, elevation=100.0, node_type=NodeType.BIKE)  # ~3.7 m
    graph.add_node(999, x=8.02, y=48.0, elevation=100.0, node_type=NodeType.BIKE)  # ~1.5 km east — outside 200 m
    stations = gpd.GeoDataFrame({"name": ["A"]}, geometry=[Point(8.0000, 48.0)], crs="EPSG:4326")
    _add_railway(graph=graph, osm=_FakeOSM(stations=stations, rails=None), dem=MockDEMService(base_elevation=100.0))
    linked = {u for u, v, d in graph.edges(data=True) if d["mode"] == Mode.STATION and v < 0}
    assert len(linked) == RailConfig.STATION_MAX_ENTRANCES  # capped
    assert 999 not in linked  # out-of-radius node never linked
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(1, x=8.0, y=48.0, elevation=100.0, node_type=NodeType.BIKE)
    assert _add_railway(graph=graph, osm=_FakeOSM(stations=None, rails=None), dem=MockDEMService()) == 0


def test_station_edges_touch_bike_nodes_not_each_other():
    # Regression: two stations closer to EACH OTHER than to the single bike node must
    # still snap to the BIKE node — otherwise they island off and get dropped by the
    # strongly-connected filter (previously all stations vanished from the corridor).
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(500, x=8.0015, y=48.0, elevation=100.0, node_type=NodeType.BIKE)  # only bike node, ~110 m east
    stations = gpd.GeoDataFrame(
        {"name": ["A", "B"]},
        geometry=[Point(8.0000, 48.0), Point(8.0003, 48.0)],  # ~22 m apart, ~90-110 m from bike node
        crs="EPSG:4326",
    )
    osm = _FakeOSM(stations=stations, rails=None)
    _add_railway(graph=graph, osm=osm, dem=MockDEMService(base_elevation=100.0))
    # every station edge must touch the bike node 500, never link two stations
    station_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["mode"] == Mode.STATION]
    assert station_edges, "expected station edges within radius"
    for u, v in station_edges:
        assert 500 in (u, v), f"station edge {u}->{v} does not touch the bike node"
