"""Graph construction, endpoint snapping, and elevation enrichment.

Builds the routable bike network in three measured, route-preserving steps:
    1. ox.graph_from_polygon(..., simplify=False, retain_all=True) — download +
       build once, skipping OSMnx's internal largest-component deep-copy (~130s).
    2. _contract_interstitial_nodes — geometry-free degree-2 contraction in place
       (~5s vs ox.simplify_graph's ~34s of unused shapely geometry).
    3. ox.truncate.largest_component(strongly=True) — the routable directed core.
The result is cached to disk (pickle, keyed by corridor bounds), so re-tuning the
routing parameters on the same start/end skips the whole build — the OSRM/
GraphHopper "preprocess once, reuse" pattern. Elevation is attached via the reused
DEMService (vectorized), NOT ox.elevation.add_node_elevations_raster (which
mis-samples the arcsecond DEM).
"""

import hashlib
import logging
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import Polygon

from bike_router.constants import OutputConfig, SurfaceConfig
from bike_router.cost import surface_tier
from bike_router.elevation import DEMService
from bike_router.progress import ProgressFn, null_progress

logger = logging.getLogger(__name__)

# Cache Overpass responses under the project's gitignored cache/ dir so repeated
# runs of the same corridor skip the network round-trip (and avoid HTTP 429).
OutputConfig.CACHE_DIR.mkdir(parents=True, exist_ok=True)
ox.settings.use_cache = True
ox.settings.cache_folder = str(OutputConfig.CACHE_DIR)
# OSMnx drops `surface`/`tracktype` by default — we MUST retain them; the whole
# surface-quality routing depends on it (≈80% of ways carry a surface tag).
_EXTRA_WAY_TAGS = ["surface", "tracktype"]
ox.settings.useful_tags_way = list(dict.fromkeys(list(ox.settings.useful_tags_way) + _EXTRA_WAY_TAGS))

_RAW_COUNT_ATTR = "raw_node_count"  # stashed on the cached graph for Sanity 1


def _graph_cache_path(polygon: Polygon) -> Path:
    """Pickle-cache path keyed by the corridor's rounded bounds (6 dp ≈ 0.1 m)."""
    bounds = tuple(round(value, 6) for value in polygon.bounds)
    digest = hashlib.sha1(repr(bounds).encode()).hexdigest()[:16]
    return OutputConfig.CACHE_DIR / f"bikegraph_{digest}.pkl"


def _drop_excluded_surface_edges(graph: nx.MultiDiGraph) -> None:
    """Remove edges whose surface tier is EXCLUDED_TIER (soft natural ground).

    Bike-legal but genuinely bad to ride (mud/sand/grass/…); dropping them up front
    guarantees no route uses them. Orphaned nodes are removed; the later
    largest_component call restores strong connectivity.
    """
    doomed = [
        (node_a, node_b, key)
        for node_a, node_b, key, data in graph.edges(keys=True, data=True)
        if surface_tier(surface=data.get("surface")) >= SurfaceConfig.EXCLUDED_TIER
    ]
    graph.remove_edges_from(doomed)
    graph.remove_nodes_from([node for node in list(graph.nodes) if graph.degree(node) == 0])


def _contract_interstitial_nodes(graph: nx.MultiDiGraph, progress: ProgressFn = null_progress) -> nx.MultiDiGraph:
    """Contract degree-2 pass-through nodes in place (geometry-free simplification).

    Removes non-intersection/dead-end nodes, summing run length — shortest paths are
    unchanged. ``progress`` is called (nodes seen, total) as the worklist drains.
    """
    total = graph.number_of_nodes()
    worklist = list(graph.nodes)
    seen: set[int] = set()
    while worklist:
        node = worklist.pop()
        seen.add(node)
        progress(min(len(seen), total), total)
        if node not in graph:
            continue
        preds, succs = set(graph.predecessors(node)), set(graph.successors(node))
        neighbours = (preds | succs) - {node}
        is_passthrough = (
            len(neighbours) == 2 and preds == succs and graph.in_degree(node) == 2 and graph.out_degree(node) == 2
        )
        if not is_passthrough:
            continue
        node_a, node_b = tuple(neighbours)
        if graph.has_edge(node_a, node_b) or graph.has_edge(node_b, node_a):
            continue  # contracting would collide with an existing edge → keep node
        sample = next(iter(graph.get_edge_data(node_a, node).values()))
        length_fwd = min(d["length"] for d in graph.get_edge_data(node_a, node).values()) + min(
            d["length"] for d in graph.get_edge_data(node, node_b).values()
        )
        length_rev = min(d["length"] for d in graph.get_edge_data(node_b, node).values()) + min(
            d["length"] for d in graph.get_edge_data(node, node_a).values()
        )
        graph.add_edge(node_a, node_b, length=length_fwd, surface=sample.get("surface"), highway=sample.get("highway"))
        graph.add_edge(node_b, node_a, length=length_rev, surface=sample.get("surface"), highway=sample.get("highway"))
        graph.remove_node(node)
        worklist.append(node_a)  # neighbours may now be degree-2 → collapse the chain
        worklist.append(node_b)
    progress(total, total)
    return graph


def build_bike_graph(polygon: Polygon, progress: ProgressFn = null_progress) -> tuple[nx.MultiDiGraph, int]:
    """Return the routable bike network for the corridor as (graph, raw_count).

    On a cache hit, loads the prebuilt graph from disk. On a miss: download once,
    drop tier-2 surfaces, contract degree-2 nodes (driving ``progress``), core it.
    """
    cache_path = _graph_cache_path(polygon=polygon)
    if cache_path.exists():
        with open(cache_path, "rb") as handle:
            graph = pickle.load(handle)
        raw_count = int(graph.graph[_RAW_COUNT_ATTR])
        logger.info("Loaded cached bike graph (%d nodes) from %s", graph.number_of_nodes(), cache_path)
        assert graph.number_of_nodes() > 0, "cached bike graph must not be empty"
        return graph, raw_count

    # retain_all=True skips OSMnx's slow internal largest_component copy; we then
    # drop tier-2 surfaces, contract degree-2 nodes ourselves, and take the core.
    raw = ox.graph_from_polygon(polygon, network_type="bike", simplify=False, retain_all=True)
    raw_count = int(raw.number_of_nodes())
    _drop_excluded_surface_edges(raw)
    graph = _contract_interstitial_nodes(raw, progress=progress)
    graph = ox.truncate.largest_component(graph, strongly=True)
    graph.graph[_RAW_COUNT_ATTR] = raw_count
    logger.info("Built bike graph: %d raw → %d core nodes", raw_count, graph.number_of_nodes())
    assert graph.number_of_nodes() > 0, "bike graph must not be empty"
    with open(cache_path, "wb") as handle:
        pickle.dump(graph, handle)
    return graph, raw_count


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


def enrich_elevations(graph: nx.MultiDiGraph, dem: DEMService) -> None:
    """Attach an ``elevation`` attribute to every node via one bulk DEM sample.

    OSMnx stores node coords as x=lon, y=lat. Out-of-coverage/nodata cells come
    back NaN and are neutral-filled with the corridor mean so they don't poison
    the elevation penalty / A*.
    """
    nodes = list(graph.nodes)
    assert nodes, "graph must have nodes to enrich"
    lons = np.array([graph.nodes[node]["x"] for node in nodes], dtype=np.float64)
    lats = np.array([graph.nodes[node]["y"] for node in nodes], dtype=np.float64)

    elevations = dem.get_elevations(lons=lons, lats=lats)
    assert len(elevations) == len(nodes), "one elevation per node expected"

    nan_mask = np.isnan(elevations)
    nan_count = int(nan_mask.sum())
    if nan_count:
        fill_value = float(np.nanmean(elevations)) if not np.all(nan_mask) else 0.0
        elevations = np.where(nan_mask, fill_value, elevations)
        logger.warning(
            "%d/%d nodes had no DEM coverage (nodata) → filled with corridor mean %.1f m",
            nan_count,
            len(nodes),
            fill_value,
        )

    assert not np.any(np.isnan(elevations)), "all node elevations must be finite after fill"
    for node, elevation in zip(nodes, elevations, strict=True):
        graph.nodes[node]["elevation"] = float(elevation)
