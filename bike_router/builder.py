"""Offline builder: a whole-region bike+rail graph from Geofabrik .osm.pbf extracts.

Reads the cycling network + railway with pyrosm, normalizes to an OSMnx-shaped
graph, simplifies/consolidates, bakes DEM elevation, then adds station-access +
rail hops. Returns node/edge tables; the routing sliders decide if A* uses rail.
"""

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyrosm import OSM
from shapely import from_wkt, to_wkt
from shapely.geometry import LineString

from bike_router.constants import (
    GraphConfig,
    Mode,
    NodeType,
    RailConfig,
)
from bike_router.elevation import DEMService
from bike_router.geo import haversine_distance_m, haversine_vec
from bike_router.graph_ops import (
    bake_edge_geometry_elevations,
    consolidate_graph,
    contract_interstitial_nodes,
    drop_disallowed_edges,
    enrich_elevations,
    normalize_pyrosm_graph,
)
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
        tags_as_columns=["name"],  # promote the OSM name tag to a column
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


def _station_entrances(
    node_ids: "np.ndarray", node_lats: "np.ndarray", node_lons: "np.ndarray", lat: float, lon: float
) -> list[tuple[int, float]]:
    """Up to MAX_ENTRANCES bike nodes inside the station radius, as (node id, distance m).

    These nearest N bike nodes within STATION_RADIUS_M (fewer if fewer exist) are declared
    the station's ENTRANCES: reaching one and crossing its station edge puts the rider at
    the station. Empty if the station has no bike node in range.
    """
    dists = haversine_vec(lat_a=lat, lon_a=lon, lat_b=node_lats, lon_b=node_lons)
    within = np.flatnonzero(dists <= RailConfig.STATION_RADIUS_M)
    nearest = within[np.argsort(dists[within])[: RailConfig.STATION_MAX_ENTRANCES]]
    return [(int(node_ids[i]), float(dists[i])) for i in nearest]


def _add_railway(graph: nx.MultiDiGraph, osm: OSM, dem: DEMService) -> int:
    """Add rail-station nodes, station-access edges, and rail edges. Returns #stations.

    Each station is a SEPARATE rail node (never a bike node), DEM-elevated and linked to
    its nearest bike-node ENTRANCES by bidirectional station edges; consecutive stations
    on a line get a bidirectional rail edge.
    """
    stations = _station_points(osm=osm)
    if not stations:
        logger.warning("No railway stations in extract — rail edges skipped")
        return 0
    lines = _rail_lines(osm=osm)

    # Snapshot the BIKE node coordinates BEFORE adding any station node, so a station's
    # entrances are only cycling nodes — never another station (which would island them off
    # the routable core and get them dropped by the strongly-connected filter).
    bike_ids = np.fromiter(graph.nodes, dtype=np.int64)
    bike_lats = np.array([graph.nodes[int(n)]["y"] for n in bike_ids])
    bike_lons = np.array([graph.nodes[int(n)]["x"] for n in bike_ids])

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
        graph.add_node(node_id, x=lon, y=lat, elevation=elevation, node_type=NodeType.RAIL, station_name=name)
        station_nodes.append((node_id, name, lat, lon))
        # Declare the nearest N bike nodes as entrances and link each both ways with a station
        # edge. Its cost = straight-line length + half the boarding charge (cost.py), so board
        # + alight sum to a full boarding; nothing time-related is stored on the edge.
        for bike_node, dist in _station_entrances(
            node_ids=bike_ids, node_lats=bike_lats, node_lons=bike_lons, lat=lat, lon=lon
        ):
            graph.add_edge(bike_node, node_id, length=dist, mode=Mode.STATION)
            graph.add_edge(node_id, bike_node, length=dist, mode=Mode.STATION)

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
        if dist <= RailConfig.STATION_RADIUS_M:
            vertex_to_station[vertex] = node_id
            snapped.append((node_id, vertex))

    # Two stations are adjacent iff the shortest rail path between them has no OTHER
    # station vertex in its interior. Connect ALL such neighbours (deduped by node pair).
    station_coord = {node_id: (lat, lon) for node_id, _name, lat, lon in station_nodes}
    added: set[tuple[int, int]] = set()
    for node_id, vertex in snapped:
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
            # Keep the REAL along-track polyline (not just its length): rail vertices are
            # (lat, lon); anchor both ends at the station nodes so the drawn line runs
            # station→track→station. Both directed edges share it (densify orients per node).
            here_lat, here_lon = station_coord[node_id]
            there_lat, there_lon = station_coord[other]
            track_coords = [(lon, lat) for lat, lon in paths[other_vertex]]
            geometry = LineString([(here_lon, here_lat), *track_coords, (there_lon, there_lat)])
            graph.add_edge(node_id, other, length=dists[other_vertex], mode=Mode.RAIL, geometry=geometry)
            graph.add_edge(other, node_id, length=dists[other_vertex], mode=Mode.RAIL, geometry=geometry)


def _tag_bike_defaults(graph: nx.MultiDiGraph) -> None:
    """Mark every current edge bike and every current node a bike node (single source).

    Called before _add_railway, when the graph holds ONLY the cycling network — so every
    node is a bike node and every edge a bike edge. _add_railway then adds rail nodes/edges.
    """
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        data["mode"] = Mode.BIKE
    for _node, data in graph.nodes(data=True):
        data["node_type"] = NodeType.BIKE


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
    name = pbf_path.name
    logger.info("%s: [1/9] parsed %d raw cycling nodes", name, raw_count)
    drop_disallowed_edges(graph=graph)
    logger.info("%s: [2/9] dropped disallowed edges → %d edges", name, graph.number_of_edges())
    graph = contract_interstitial_nodes(graph=graph)
    logger.info("%s: [3/9] contracted degree-2 nodes → %d nodes", name, graph.number_of_nodes())
    graph = consolidate_graph(graph=graph, tolerance_m=tolerance_m)
    logger.info("%s: [4/9] consolidated intersections → %d nodes", name, graph.number_of_nodes())
    graph = ox.truncate.largest_component(graph, strongly=True)
    logger.info("%s: [5/9] largest strongly-connected component → %d nodes", name, graph.number_of_nodes())
    check_simplify_shrunk(nodes_before=raw_count, nodes_after=graph.number_of_nodes())
    _tag_bike_defaults(graph=graph)
    enrich_elevations(graph=graph, dem=dem)
    logger.info("%s: [6/9] baked node elevations", name)
    n_stations = _add_railway(graph=graph, osm=osm, dem=dem)
    logger.info("%s: [7/9] wired %d railway stations", name, n_stations)
    bake_edge_geometry_elevations(graph=graph, dem=dem)  # 3D vertices → inference needs no DEM
    logger.info("%s: [8/9] baked edge geometry elevations", name)
    # Re-run the strongly-connected filter AFTER railway wiring: guarantees ZERO dangling nodes.
    graph = ox.truncate.largest_component(graph, strongly=True)
    logger.info(
        "%s: [9/9] done — %d nodes, %d edges, %d stations",
        name,
        graph.number_of_nodes(),
        graph.number_of_edges(),
        n_stations,
    )
    return graph


def remap_contiguous(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renumber a region's node ids to contiguous ``0..N-1`` (Phase 2).

    osmnx consolidation leaves gapped 0-based bike ids and our station code adds negative
    ids, so a region spans e.g. ``[-3 .. 103561]`` with holes. Map every id to its rank in
    sorted order and rewrite ``osmid`` + both edge endpoints, so afterwards
    ``n_nodes == max_id + 1`` (no gaps, no negatives). Station-ness lives in ``node_type``.
    """
    ordered = sorted(nodes_df["osmid"].tolist())
    remap = {old: new for new, old in enumerate(ordered)}
    nodes_df = nodes_df.copy()
    edges_df = edges_df.copy()
    nodes_df["osmid"] = nodes_df["osmid"].map(remap)
    edges_df["from_node"] = edges_df["from_node"].map(remap)
    edges_df["to_node"] = edges_df["to_node"].map(remap)
    return nodes_df, edges_df


def reindex_region(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, offset: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shift a region's already-contiguous ``0..N-1`` ids by ``offset`` (Phase 3).

    Inputs come from remap_contiguous, so ids are dense ``0..N-1``; adding a running total
    (``ΣN`` of earlier regions) yields a globally contiguous, collision-free id space.
    """
    if offset == 0:
        return nodes_df, edges_df
    nodes_df = nodes_df.copy()
    edges_df = edges_df.copy()
    nodes_df["osmid"] = nodes_df["osmid"] + offset
    edges_df["from_node"] = edges_df["from_node"] + offset
    edges_df["to_node"] = edges_df["to_node"] + offset
    return nodes_df, edges_df


def dedup_by_geometry(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse border duplicates that appear in two regions' reference-complete extracts.

    Nodes are the same physical node if their (lat, lon) coincide to COORD_PRECISION; the
    lower-(lat, lon) copy is kept and the others' ids are repointed onto it. Edges are the
    same only if BOTH endpoints AND the 3D geometry (rounded WKT, or endpoints+mode for a
    null-geometry rail/station hop) coincide — so genuinely parallel distinct roads survive.
    """
    prec = GraphConfig.COORD_PRECISION
    nodes_df = nodes_df.copy()
    # Canonical node per rounded (lat, lon): the row with the lowest (lat, lon), then id.
    nodes_df = nodes_df.sort_values(["lat", "lon", "osmid"], kind="stable")
    key_lat = nodes_df["lat"].round(prec)
    key_lon = nodes_df["lon"].round(prec)
    nodes_df["_key"] = list(zip(key_lat, key_lon, strict=True))
    canonical = nodes_df.groupby("_key")["osmid"].transform("first")
    repoint = dict(zip(nodes_df["osmid"], canonical, strict=True))  # every id → its kept id
    kept_nodes = nodes_df.drop_duplicates(subset="_key", keep="first").drop(columns="_key")

    edges_df = edges_df.copy()
    edges_df["from_node"] = edges_df["from_node"].map(repoint)
    edges_df["to_node"] = edges_df["to_node"].map(repoint)

    # Edge identity = endpoints + geometry (rounded) + mode; parallel distinct roads differ
    # in geometry and are both kept. Null geometry (rail/station hop) dedups on endpoints+mode.
    def _geom_key(wkt: object) -> str:
        if not isinstance(wkt, str):
            return ""
        return str(to_wkt(from_wkt(wkt), rounding_precision=prec))

    edges_df["_ekey"] = list(
        zip(
            edges_df["from_node"],
            edges_df["to_node"],
            edges_df["mode"],
            [_geom_key(w) for w in edges_df["geometry_wkt"]],
            strict=True,
        )
    )
    kept_edges = edges_df.drop_duplicates(subset="_ekey", keep="first").drop(columns="_ekey")
    return kept_nodes, kept_edges
