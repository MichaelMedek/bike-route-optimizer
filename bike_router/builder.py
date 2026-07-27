"""Offline builder: a whole-region bike+rail graph from Geofabrik .osm.pbf extracts.

Reads the cycling network + railway with pyrosm, normalizes to an OSMnx-shaped
graph, simplifies/consolidates, bakes DEM elevation, then adds station-access +
rail hops. Returns node/edge tables; the routing sliders decide if A* uses rail.
"""

import logging
from pathlib import Path
from typing import cast

import geopandas as gpd
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


def _get_network(
    osm: OSM, custom_filter: dict[str, list[str]] | None, network_type: str = "all"
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | None:
    """One entry point to pyrosm's routable-network builder for BOTH bike and rail layers.

    Returns pyrosm's ``(nodes_gdf, edges_gdf)`` (or None if the layer is absent). Both networks go
    through the same LineString-only extractor — no layer gets special geometry handling.
    """
    res = osm.get_network(network_type=network_type, custom_filter=custom_filter, nodes=True)
    return cast("tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | None", res)


def _cycling_graph(osm: OSM) -> nx.MultiDiGraph:
    """Read the cycling network from a pbf as a normalized OSMnx-shaped graph."""
    res = _get_network(osm=osm, custom_filter=None, network_type="cycling")
    assert res is not None, "pbf has no cycling network"
    nodes, edges = res
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


def _rail_network(osm: OSM) -> nx.Graph:
    """Undirected graph of the whole rail network, built by pyrosm's own network builder.

    Uses the SAME ``_get_network`` extractor as the bike graph (LineString edges only — mistagged
    area/polygon rail is excluded for free). Vertices are keyed on rounded (lat, lon) so the
    station-snapping below can match by coordinate; edge weight is the real segment length.
    """
    res = _get_network(osm=osm, custom_filter={"railway": list(RailConfig.RAIL_TAGS)})
    net: nx.Graph = nx.Graph()
    if res is None:
        return net
    _nodes, edges = res
    for geom in edges.geometry:
        coords = list(geom.coords)  # (lon, lat) segment vertices; get_network → LineString only
        for (lon_a, lat_a), (lon_b, lat_b) in zip(coords[:-1], coords[1:], strict=True):
            ka = (round(lat_a, GraphConfig.COORD_PRECISION), round(lon_a, GraphConfig.COORD_PRECISION))
            kb = (round(lat_b, GraphConfig.COORD_PRECISION), round(lon_b, GraphConfig.COORD_PRECISION))
            if ka != kb:
                net.add_edge(ka, kb, length=haversine_distance_m(lat_a=lat_a, lon_a=lon_a, lat_b=lat_b, lon_b=lon_b))
    return net


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
    net = _rail_network(osm=osm)

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

    _connect_stations_along_lines(graph=graph, station_nodes=station_nodes, net=net)
    return len(station_nodes)


def _connect_stations_along_lines(
    graph: nx.MultiDiGraph,
    station_nodes: list[tuple[int, str, float, float]],
    net: nx.Graph,
) -> None:
    """Add rail edges between stations ADJACENT along the connected rail network.

    Snaps each station to its nearest rail vertex, then joins two stations iff the shortest
    rail path between them passes through NO other station vertex — i.e. they are consecutive
    stops (handles both directions and branches at junctions). ``net`` is the stitched rail
    graph from ``_rail_network``.
    """
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
    logger.info(f"{name}: [1/9] parsed {raw_count} raw cycling nodes")
    drop_disallowed_edges(graph=graph)
    logger.info(f"{name}: [2/9] dropped disallowed edges → {graph.number_of_edges()} edges")
    graph = contract_interstitial_nodes(graph=graph)
    logger.info(f"{name}: [3/9] contracted degree-2 nodes → {graph.number_of_nodes()} nodes")
    graph = consolidate_graph(graph=graph, tolerance_m=tolerance_m)
    logger.info(f"{name}: [4/9] consolidated intersections → {graph.number_of_nodes()} nodes")
    graph = ox.truncate.largest_component(graph, strongly=True)
    logger.info(f"{name}: [5/9] largest strongly-connected component → {graph.number_of_nodes()} nodes")
    check_simplify_shrunk(nodes_before=raw_count, nodes_after=graph.number_of_nodes())
    _tag_bike_defaults(graph=graph)
    enrich_elevations(graph=graph, dem=dem)
    logger.info(f"{name}: [6/9] baked node elevations")
    n_stations = _add_railway(graph=graph, osm=osm, dem=dem)
    logger.info(f"{name}: [7/9] wired {n_stations} railway stations")
    bake_edge_geometry_elevations(graph=graph, dem=dem)  # 3D vertices → inference needs no DEM
    logger.info(f"{name}: [8/9] baked edge geometry elevations")
    # Re-run the strongly-connected filter AFTER railway wiring: guarantees ZERO dangling nodes.
    graph = ox.truncate.largest_component(graph, strongly=True)
    logger.info(
        f"{name}: [9/9] done — {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, {n_stations} stations"
    )
    return graph


def remap_contiguous(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renumber a region's gapped/negative node ids to contiguous ``0..N-1`` (Phase 2).

    Maps every id to its rank in sorted order and rewrites ``osmid`` + both edge endpoints, so
    afterwards ``n_nodes == max_id + 1``. Station-ness lives in ``node_type``, not the id sign.
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

    Nodes coinciding in (lat, lon) to COORD_PRECISION collapse to the lower-(lat, lon) copy
    (others repointed onto it); edges collapse only if endpoints AND geometry (rounded WKT, or
    endpoints+mode for null-geometry hops) coincide — so genuinely parallel roads both survive.
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
