"""Graph construction, endpoint snapping, and elevation enrichment.

Wraps OSMnx 2.x calls (verified against the current docs):
    - ox.graph_from_polygon(polygon, network_type="bike", simplify=True)
    - ox.truncate.largest_component(graph, strongly=True)   # routable directed core
    - ox.distance.nearest_nodes(graph, X=lon, Y=lat)
Elevation is attached via the reused DEMService (vectorized), NOT
ox.elevation.add_node_elevations_raster (which mis-samples the arcsecond DEM).
"""

import logging

import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import Polygon

from bike_router.constants import OutputConfig
from bike_router.elevation import DEMService

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


def build_bike_graph(polygon: Polygon) -> nx.MultiDiGraph:
    """Extract a bike network from the corridor polygon and keep the routable core.

    `strongly=True` keeps the largest STRONGLY-connected component so a directed
    route is guaranteed to exist between any two of its nodes.
    """
    graph = ox.graph_from_polygon(polygon, network_type="bike", simplify=True)
    raw_count = graph.number_of_nodes()
    graph = ox.truncate.largest_component(graph, strongly=True)
    logger.info("Graph: %d nodes raw → %d nodes in strongly-connected core", raw_count, graph.number_of_nodes())
    assert graph.number_of_nodes() > 0, "bike graph must not be empty"
    return graph


def raw_node_count(polygon: Polygon) -> int:
    """Node count of the UN-simplified graph (for the >50% shrink sanity check)."""
    graph = ox.graph_from_polygon(polygon, network_type="bike", simplify=False)
    count = int(graph.number_of_nodes())
    assert count > 0, "raw graph must not be empty"
    return count


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
