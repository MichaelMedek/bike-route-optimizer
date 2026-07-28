"""Source-agnostic graph transforms shared by the offline builder and inference.

These operate on an OSMnx-shaped ``nx.MultiDiGraph`` (node attrs ``x``/``y``, edge
attrs ``length``/``surface``/``highway``/``geometry``) regardless of whether the graph
came from a pyrosm ``.osm.pbf`` read or a reconstructed corridor subset — so the same
code path enforces surface filtering, intersection consolidation, and elevation baking
everywhere (no duplicated logic). Degree-2 contraction happens upstream in pyrosm's
``to_graph(simplify=True)``.
"""

import logging

import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import LineString

from bike_router.cost import road_included, surface_included
from bike_router.elevation import DEMService

logger = logging.getLogger(__name__)


# Node/edge attributes pyrosm attaches that COLLIDE with the (osmid) node index or
# the (u, v, key) edge index when osmnx converts a graph to/from GeoDataFrames.
# Stripping them makes pyrosm graphs safe for ox.projection / ox.simplification.
# NOTE: ``geometry`` is deliberately KEPT — the real edge polyline is what makes the
# 3D path and DEM-draped elevation follow the true road instead of a straight line.
_PYROSM_NODE_JUNK = ("osmid", "geometry", "tags", "version", "visible", "changeset", "timestamp")
_PYROSM_EDGE_JUNK = ("osmid", "u", "v", "key", "tags", "version", "timestamp", "osm_type")


def normalize_pyrosm_graph(graph: nx.MultiDiGraph) -> None:
    """Strip pyrosm's index-colliding node/edge attributes in place.

    pyrosm's ``to_graph(osmnx_compatible=True)`` keeps an ``osmid`` node attribute
    (duplicating the node id) and raw ``u``/``v``/``osmid`` edge attributes; osmnx's
    graph↔gdf round-trip then raises "cannot insert osmid, already exists". We keep
    the routing-relevant attrs (x/y on nodes; length/surface/highway/geometry on edges).
    """
    for _node, data in graph.nodes(data=True):
        for junk in _PYROSM_NODE_JUNK:
            data.pop(junk, None)
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        for junk in _PYROSM_EDGE_JUNK:
            data.pop(junk, None)


def drop_disallowed_edges(graph: nx.MultiDiGraph) -> None:
    """Remove edges whose surface OR highway tag names a category outside its allowlist.

    ALLOWLIST (symmetric): only SurfaceConfig.SURFACE_TIER surfaces and RoadConfig.ROAD_TIER
    highway classes (+ untagged) enter the graph; any other named surface (sand/dirt/…) or
    highway (motorway/raceway/…) is dropped up front so no route uses it. Orphaned nodes are
    removed; a later largest_component call restores connectivity.
    """
    doomed = [
        (node_a, node_b, key)
        for node_a, node_b, key, data in graph.edges(keys=True, data=True)
        if not (surface_included(surface=data.get("surface")) and road_included(highway=data.get("highway")))
    ]
    graph.remove_edges_from(doomed)
    graph.remove_nodes_from([node for node in list(graph.nodes) if graph.degree(node) == 0])


def consolidate_graph(graph: nx.MultiDiGraph, tolerance_m: float) -> nx.MultiDiGraph:
    """Merge intersection clusters within ``tolerance_m`` metres (shrinks the graph).

    Projects to an auto-selected UTM zone (consolidation needs metric units),
    merges nodes whose ``tolerance_m``-radius buffers overlap, rebuilds the topology
    with reconnected edges + updated lengths, then unprojects back to EPSG:4326.
    """
    assert tolerance_m > 0, "consolidation tolerance must be positive"
    projected = ox.projection.project_graph(graph)
    consolidated = ox.simplification.consolidate_intersections(
        projected, tolerance=tolerance_m, rebuild_graph=True, dead_ends=False, reconnect_edges=True
    )
    unprojected: nx.MultiDiGraph = ox.projection.project_graph(consolidated, to_latlong=True)
    logger.info(f"Consolidated ({tolerance_m:.0f}m): {graph.number_of_nodes()} → {unprojected.number_of_nodes()} nodes")
    return unprojected


def snap_endpoints(
    graph: nx.MultiDiGraph,
    start_latlon: tuple[float, float],
    dest_latlon: tuple[float, float],
) -> tuple[int, int]:
    """Snap (lat, lon) start/dest to their nearest graph node ids.

    OSMnx expects X=longitude, Y=latitude.
    """
    start_lat, start_lon = start_latlon
    dest_lat, dest_lon = dest_latlon
    source = int(ox.distance.nearest_nodes(graph, X=start_lon, Y=start_lat))
    target = int(ox.distance.nearest_nodes(graph, X=dest_lon, Y=dest_lat))
    assert source in graph and target in graph, "snapped nodes must exist in the graph"
    return source, target


def _fill_nan_with_mean(values: "np.ndarray") -> tuple["np.ndarray", int]:
    """Neutral-fill NaN DEM samples with the finite mean (0.0 if all NaN); returns (filled, nan_count)."""
    nan_mask = np.isnan(values)
    nan_count = int(nan_mask.sum())
    if nan_count:
        fill_value = float(np.nanmean(values)) if not np.all(nan_mask) else 0.0
        values = np.where(nan_mask, fill_value, values)
    return values, nan_count


def enrich_elevations(graph: nx.MultiDiGraph, dem: DEMService) -> None:
    """Attach an ``elevation`` attribute to every node via one bulk DEM sample.

    OSMnx stores node coords as x=lon, y=lat. Out-of-coverage/nodata cells come
    back NaN and are neutral-filled with the graph mean so they don't poison the
    elevation penalty / A*. Runs at BUILD time only (baked into the artifact).
    """
    nodes = list(graph.nodes)
    assert nodes, "graph must have nodes to enrich"
    lons = np.array([graph.nodes[node]["x"] for node in nodes], dtype=np.float64)
    lats = np.array([graph.nodes[node]["y"] for node in nodes], dtype=np.float64)

    elevations = dem.get_elevations(lons=lons, lats=lats)
    assert len(elevations) == len(nodes), "one elevation per node expected"

    elevations, nan_count = _fill_nan_with_mean(values=elevations)
    if nan_count:
        logger.warning(f"{nan_count}/{len(nodes)} nodes had no DEM coverage (nodata) → neutral-filled")

    assert not np.any(np.isnan(elevations)), "all node elevations must be finite after fill"
    for node, elevation in zip(nodes, elevations, strict=True):
        graph.nodes[node]["elevation"] = float(elevation)


def bake_edge_geometry_elevations(graph: nx.MultiDiGraph, dem: DEMService) -> None:
    """Replace each edge's 2D polyline with a 3D one (lon, lat, elev), in place.

    Samples the DEM at EVERY geometry vertex so the baked artifact fully describes the
    route's terrain — inference then reads these elevations directly and never touches
    the DEM. Bike and rail edges both carry a real polyline; only the short station
    access-links (no geometry) are left as straight hops. Runs at BUILD time only.
    All vertices across the graph are sampled in one bulk call.
    """
    edges = [(u, v, k, d) for u, v, k, d in graph.edges(keys=True, data=True) if d.get("geometry") is not None]
    if not edges:
        return
    counts = [len(d["geometry"].coords) for _u, _v, _k, d in edges]
    flat_lon = np.array([c[0] for _u, _v, _k, d in edges for c in d["geometry"].coords], dtype=np.float64)
    flat_lat = np.array([c[1] for _u, _v, _k, d in edges for c in d["geometry"].coords], dtype=np.float64)

    elevs, _nan_count = _fill_nan_with_mean(values=dem.get_elevations(lons=flat_lon, lats=flat_lat))

    offset = 0
    for (_u, _v, _k, data), n in zip(edges, counts, strict=True):
        coords = list(data["geometry"].coords)
        data["geometry"] = LineString([(lon, lat, float(elevs[offset + i])) for i, (lon, lat) in enumerate(coords)])
        offset += n
    assert offset == len(flat_lon), "every vertex must be consumed exactly once"
