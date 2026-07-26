"""Offline builder: a whole-region bike+rail graph from Geofabrik .osm.pbf extracts.

Reads the cycling network + railway with pyrosm, normalizes to an OSMnx-shaped
graph, simplifies/consolidates, bakes DEM elevation, then adds transfer + rail
hops. Returns node/edge tables; the routing sliders decide if A* uses rail.
"""

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyrosm import OSM

from bike_router.constants import (
    Mode,
    RailConfig,
)
from bike_router.elevation import DEMService
from bike_router.geo import haversine_distance_m
from bike_router.graph_ops import (
    bake_edge_geometry_elevations,
    consolidate_graph,
    contract_interstitial_nodes,
    drop_excluded_surface_edges,
    enrich_elevations,
    normalize_pyrosm_graph,
)
from bike_router.graph_store import graph_to_tables
from bike_router.sanity import check_simplify_shrunk

logger = logging.getLogger(__name__)


def _cycling_graph(osm: OSM) -> nx.MultiDiGraph:
    """Read the cycling network from a pbf as a normalized OSMnx-shaped graph."""
    nodes, edges = osm.get_network(network_type="cycling", nodes=True)
    assert nodes is not None and edges is not None, "pbf has no cycling network"
    graph: nx.MultiDiGraph = osm.to_graph(nodes, edges, graph_type="networkx", osmnx_compatible=True, retain_all=True)
    normalize_pyrosm_graph(graph=graph)
    return graph


def _open_osm(pbf_path: Path, bbox: tuple[float, float, float, float] | None) -> OSM:
    """Open a pbf, optionally clipped to a (west, south, east, north) bbox.

    Clipping keeps region tests fast (parse only the corridor area) without
    changing the build logic; full-country builds pass bbox=None.
    """
    if bbox is None:
        return OSM(str(pbf_path))
    from shapely.geometry import box

    west, south, east, north = bbox
    return OSM(str(pbf_path), bounding_box=box(west, south, east, north))


def _station_points(osm: OSM) -> list[tuple[str, float, float]]:
    """Extract (name, lat, lon) for railway stations/halts; empty list if none.

    Polygon/way stations are reduced to their centroid so every stop is one point.
    """
    stations = osm.get_data_by_custom_criteria(
        custom_filter={"railway": list(RailConfig.STATION_TAGS)},
        filter_type="keep",
        keep_nodes=True,
        keep_ways=True,
        keep_relations=False,
    )
    if stations is None or stations.empty:
        return []
    points = stations.geometry.representative_point()
    names = stations["name"] if "name" in stations.columns else [None] * len(stations)
    return [
        (name if isinstance(name, str) else "station", float(pt.y), float(pt.x))
        for name, pt in zip(names, points, strict=True)
    ]


def _rail_lines(osm: OSM) -> list[list[tuple[float, float]]]:
    """Extract rail lines as lists of (lat, lon) vertices (each Line/MultiLine part)."""
    rails = osm.get_data_by_custom_criteria(
        custom_filter={"railway": list(RailConfig.RAIL_TAGS)},
        filter_type="keep",
        keep_nodes=False,
        keep_ways=True,
        keep_relations=False,
    )
    if rails is None or rails.empty:
        return []
    lines: list[list[tuple[float, float]]] = []
    for geom in rails.geometry:
        parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            coords = [(lat, lon) for lon, lat in part.coords]
            if len(coords) >= 2:
                lines.append(coords)
    return lines


def _nearest_bike_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> tuple[int, float]:
    """Nearest bike node id + its distance (m) to (lat, lon)."""
    node = int(ox.distance.nearest_nodes(graph, X=lon, Y=lat))
    d = haversine_distance_m(lat_a=lat, lon_a=lon, lat_b=graph.nodes[node]["y"], lon_b=graph.nodes[node]["x"])
    return node, d


def _add_railway(graph: nx.MultiDiGraph, osm: OSM, dem: DEMService) -> int:
    """Add station nodes, transfer edges, and rail edges. Returns #stations.

    Stations become DEM-elevated nodes linked to their nearest bike node by
    bidirectional transfers; consecutive stations on a line get a bidirectional
    rail edge.
    """
    stations = _station_points(osm=osm)
    if not stations:
        logger.warning("No railway stations in extract — rail edges skipped")
        return 0
    lines = _rail_lines(osm=osm)

    # Snap stations to the BIKE network only — a view frozen before any station node
    # is added, so stations never snap to each other (which would island them off the
    # routable core and get them dropped by the strongly-connected filter).
    bike_graph = graph.copy()

    # Assign each station a synthetic node id below the OSM id range so it can't clash.
    base_id = -1
    station_nodes: list[tuple[int, str, float, float]] = []
    lats = np.array([lat for _n, lat, _lon in stations])
    lons = np.array([lon for _n, _lat, lon in stations])
    elevs = dem.get_elevations(lons=lons, lats=lats)
    for (name, lat, lon), elev in zip(stations, elevs, strict=True):
        node_id = base_id
        base_id -= 1
        elevation = float(elev) if not np.isnan(elev) else 0.0
        graph.add_node(node_id, x=lon, y=lat, elevation=elevation, is_station=True, station_name=name)
        station_nodes.append((node_id, name, lat, lon))
        bike_node, dist = _nearest_bike_node(graph=bike_graph, lat=lat, lon=lon)
        if dist <= RailConfig.STATION_TRANSFER_RADIUS_M:
            # Bidirectional bike↔station link. The boarding wait is applied at ride
            # time when a leg ENTERS a station (build_track), so no time is stored here.
            graph.add_edge(bike_node, node_id, length=dist, mode=Mode.TRANSFER)
            graph.add_edge(node_id, bike_node, length=dist, mode=Mode.TRANSFER)

    _connect_stations_along_lines(graph=graph, station_nodes=station_nodes, lines=lines)
    return len(station_nodes)


def _rail_network(lines: list[list[tuple[float, float]]]) -> nx.Graph:
    """Undirected graph of the whole rail network: vertices joined where segments meet.

    OSM splits a physical line into thousands of tiny ways; we stitch them by keying
    each vertex on its rounded (lat, lon) so shared endpoints become one node. Edge
    weight is the real great-circle segment length in metres.
    """
    net: nx.Graph = nx.Graph()
    for line in lines:
        for (lat_a, lon_a), (lat_b, lon_b) in zip(line[:-1], line[1:], strict=True):
            ka = (round(lat_a, 6), round(lon_a, 6))
            kb = (round(lat_b, 6), round(lon_b, 6))
            if ka != kb:
                net.add_edge(ka, kb, length=haversine_distance_m(lat_a=lat_a, lon_a=lon_a, lat_b=lat_b, lon_b=lon_b))
    return net


def _connect_stations_along_lines(
    graph: nx.MultiDiGraph,
    station_nodes: list[tuple[int, str, float, float]],
    lines: list[list[tuple[float, float]]],
) -> None:
    """Add rail edges between stations ADJACENT along the connected rail network.

    Builds the whole rail network (segments stitched at shared vertices), snaps each
    station to its nearest rail vertex, then joins two stations iff the shortest rail
    path between them passes through NO other station vertex — i.e. they are consecutive
    stops (handles both directions and branches at junctions). Robust to OSM splitting a
    physical line into thousands of tiny ways.
    """
    net = _rail_network(lines=lines)
    if net.number_of_nodes() == 0:
        return
    net_vertices = list(net.nodes)
    net_lats = np.array([v[0] for v in net_vertices])
    net_lons = np.array([v[1] for v in net_vertices])

    # Snap each station to its nearest rail vertex (only if close enough to the tracks).
    vertex_to_station: dict[tuple[float, float], int] = {}
    snapped: list[tuple[int, tuple[float, float]]] = []
    for node_id, _name, lat, lon in station_nodes:
        d2 = (net_lats - lat) ** 2 + ((net_lons - lon) * np.cos(np.radians(lat))) ** 2
        vertex = net_vertices[int(d2.argmin())]
        dist = haversine_distance_m(lat_a=lat, lon_a=lon, lat_b=vertex[0], lon_b=vertex[1])
        if dist <= RailConfig.STATION_TRANSFER_RADIUS_M:
            vertex_to_station[vertex] = node_id
            snapped.append((node_id, vertex))

    # Two stations are adjacent iff the shortest rail path between them has no OTHER
    # station vertex in its interior. Connect ALL such neighbours (deduped by node pair).
    added: set[tuple[int, int]] = set()
    for node_id, vertex in snapped:
        if vertex not in net:
            continue
        dists, paths = nx.single_source_dijkstra(net, vertex, weight="length")
        for other_vertex, other in vertex_to_station.items():
            if other == node_id or other_vertex not in paths:
                continue
            pair = (min(node_id, other), max(node_id, other))
            if pair in added or dists[other_vertex] <= 0:
                continue
            if any(interior in vertex_to_station for interior in paths[other_vertex][1:-1]):
                continue  # a closer station sits between them → not consecutive
            added.add(pair)
            graph.add_edge(node_id, other, length=dists[other_vertex], mode=Mode.RAIL)
            graph.add_edge(other, node_id, length=dists[other_vertex], mode=Mode.RAIL)


def _tag_bike_edges(graph: nx.MultiDiGraph) -> None:
    """Mark every not-yet-tagged edge as a bike edge (single source of the mode attr)."""
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        data.setdefault("mode", Mode.BIKE)


def build_region_graph(
    *,
    pbf_path: Path,
    dem: DEMService,
    tolerance_m: float,
    bbox: tuple[float, float, float, float] | None = None,
) -> nx.MultiDiGraph:
    """Build ONE region's consolidated bike+rail graph with baked elevation.

    Args:
        pbf_path: Geofabrik .osm.pbf extract for the region.
        dem: Loaded DEM sampler (node/station elevations are baked here).
        tolerance_m: Intersection consolidation radius (metres); 0 disables.
        bbox: Optional (west, south, east, north) clip to speed up region tests.
    """
    osm = _open_osm(pbf_path=pbf_path, bbox=bbox)
    graph = _cycling_graph(osm=osm)
    raw_count = graph.number_of_nodes()
    logger.info("%s: %d raw cycling nodes", pbf_path.name, raw_count)
    drop_excluded_surface_edges(graph=graph)
    graph = contract_interstitial_nodes(graph=graph)
    graph = consolidate_graph(graph=graph, tolerance_m=tolerance_m)
    graph = ox.truncate.largest_component(graph, strongly=True)
    check_simplify_shrunk(nodes_before=raw_count, nodes_after=graph.number_of_nodes())
    _tag_bike_edges(graph=graph)
    enrich_elevations(graph=graph, dem=dem)
    n_stations = _add_railway(graph=graph, osm=osm, dem=dem)
    bake_edge_geometry_elevations(graph=graph, dem=dem)  # 3D vertices → inference needs no DEM
    logger.info(
        "%s: %d nodes, %d edges, %d stations",
        pbf_path.name,
        graph.number_of_nodes(),
        graph.number_of_edges(),
        n_stations,
    )
    return graph


def build_country_graph(
    *,
    pbf_paths: list[Path],
    dem: DEMService,
    tolerance_m: float,
    bbox: tuple[float, float, float, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build + merge several region graphs into node/edge tables for graph_store.

    Regions are built independently (bounded memory) then unioned by node id. ``bbox``
    clips every region (region tests); None builds full extents. Large resumable runs
    use scripts/build_dach_graph.py, which shares merge_region_tables.
    """
    regions = [
        graph_to_tables(build_region_graph(pbf_path=p, dem=dem, tolerance_m=tolerance_m, bbox=bbox)) for p in pbf_paths
    ]
    return merge_region_tables(regions=regions)


# Each region's synthetic station ids (-1, -2, …) are shifted into a private block
# so regions never collide, regardless of build order — deterministic + resume-safe.
STATION_ID_BLOCK = 100_000_000


def merge_region_tables(regions: list[tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Union per-region (nodes, edges) tables into one deduplicated pair.

    Region ``i`` has its negative (station) ids shifted by ``(i+1)*STATION_ID_BLOCK``
    so stations from different regions never clash; positive OSM ids are globally
    unique already and dedup keeps the first copy of shared boundary nodes/edges.
    """
    assert regions, "need at least one region to merge"
    shifted_nodes: list[pd.DataFrame] = []
    shifted_edges: list[pd.DataFrame] = []
    for index, (nodes_df, edges_df) in enumerate(regions):
        nodes_df, edges_df = _shift_station_ids(nodes_df=nodes_df, edges_df=edges_df, offset=index * STATION_ID_BLOCK)
        shifted_nodes.append(nodes_df)
        shifted_edges.append(edges_df)
    nodes = pd.concat(shifted_nodes, ignore_index=True).drop_duplicates(subset="osmid", keep="first")
    edges = pd.concat(shifted_edges, ignore_index=True).drop_duplicates(
        subset=["from_node", "to_node", "key"], keep="first"
    )
    return nodes, edges


def _shift_station_ids(
    nodes_df: pd.DataFrame, edges_df: pd.DataFrame, offset: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shift negative (station) node ids down by ``offset`` so regions don't collide."""
    if offset == 0:
        return nodes_df, edges_df
    nodes_df = nodes_df.copy()
    edges_df = edges_df.copy()
    mask = nodes_df["osmid"] < 0
    nodes_df.loc[mask, "osmid"] = nodes_df.loc[mask, "osmid"] - offset
    edges_df.loc[edges_df["from_node"] < 0, "from_node"] = edges_df.loc[edges_df["from_node"] < 0, "from_node"] - offset
    edges_df.loc[edges_df["to_node"] < 0, "to_node"] = edges_df.loc[edges_df["to_node"] < 0, "to_node"] - offset
    return nodes_df, edges_df
