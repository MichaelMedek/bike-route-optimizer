"""Source-agnostic graph transforms shared by the offline builder and inference.

Operate on an OSMnx-shaped nx.MultiDiGraph regardless of whether it came from a pyrosm .osm.pbf read or a
reconstructed corridor — so surface filtering, consolidation, and elevation baking share ONE code path.
"""

import logging

import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import LineString

from bike_router.core.constants import Mode, Schema
from bike_router.core.cost import road_included, surface_included
from bike_router.core.geo import haversine_distance_m
from bike_router.preprocessing.elevation import DEMService

logger = logging.getLogger(__name__)


# pyrosm attaches node/edge attrs that COLLIDE with the (osmid) / (u, v, key) index when osmnx
# converts to/from GeoDataFrames; stripping them makes pyrosm graphs safe for ox.projection.
# ``geometry`` is deliberately KEPT — the real polyline drives the 3D path + DEM-draped elevation.
_PYROSM_NODE_JUNK = (Schema.OSMID, Schema.GEOMETRY, "tags", "version", "visible", "changeset", "timestamp")
_PYROSM_EDGE_JUNK = (Schema.OSMID, "u", "v", Schema.KEY, "tags", "version", "timestamp", "osm_type")


def normalize_pyrosm_graph(graph: nx.MultiDiGraph) -> None:
    """Strip pyrosm's index-colliding node/edge attributes in place.

    pyrosm's duplicate ``osmid``/``u``/``v`` attrs break osmnx's graph↔gdf round-trip; we keep
    only the routing-relevant attrs (x/y on nodes; length/surface/highway/geometry on edges).
    """
    for _node, data in graph.nodes(data=True):
        for junk in _PYROSM_NODE_JUNK:
            data.pop(junk, None)
    for _u, _v, _key, data in graph.edges(keys=True, data=True):
        for junk in _PYROSM_EDGE_JUNK:
            data.pop(junk, None)


def drop_disallowed_edges(graph: nx.MultiDiGraph) -> None:
    """Remove edges whose surface OR highway tag names a category outside its allowlist.

    Symmetric allowlist: only SURFACE_TIER surfaces + ROAD_TIER highways (+ untagged) enter, so no
    route uses others. Orphaned nodes are removed; a later largest_component restores connectivity.
    """
    doomed = [
        (node_a, node_b, key)
        for node_a, node_b, key, data in graph.edges(keys=True, data=True)
        if not (surface_included(surface=data.get(Schema.SURFACE)) and road_included(highway=data.get(Schema.HIGHWAY)))
    ]
    graph.remove_edges_from(doomed)
    graph.remove_nodes_from([node for node in list(graph.nodes) if graph.degree(node) == 0])


def consolidate_graph(graph: nx.MultiDiGraph, tolerance_m: float) -> nx.MultiDiGraph:
    """Merge intersection clusters within ``tolerance_m`` metres (shrinks the graph).

    Projects to auto-selected UTM (consolidation needs metric units), merges nodes whose
    buffers overlap with reconnected edges + updated lengths, then unprojects to EPSG:4326.
    """
    assert tolerance_m > 0, "consolidation tolerance must be positive"
    projected = ox.projection.project_graph(graph)
    # dead_ends=True is ESSENTIAL: a dead-end is a valid destination (rail terminus) and
    # may connect to a neighbour region during Phase-3 stitching.
    consolidated = ox.simplification.consolidate_intersections(
        projected, tolerance=tolerance_m, rebuild_graph=True, dead_ends=True, reconnect_edges=True
    )
    unprojected: nx.MultiDiGraph = ox.projection.project_graph(consolidated, to_latlong=True)
    logger.info(f"Consolidated ({tolerance_m:.0f}m): {graph.number_of_nodes()} → {unprojected.number_of_nodes()} nodes")
    return unprojected


def _densify_coords(coords: list[tuple[float, float]], max_spacing_m: float) -> list[tuple[float, float]]:
    """Subdivide a lon/lat polyline so no consecutive pair is within ``max_spacing_m`` (linear inserts).

    Targets 90% of the cap so haversine sub-gaps stay STRICTLY under it despite lon/lat-linear interpolation
    (evenly-spaced fractions aren't exactly evenly-spaced in metres); existing vertices are always kept.
    """
    target = max_spacing_m * 0.9  # safety margin below the hard cap for the linear-vs-great-circle mismatch
    out: list[tuple[float, float]] = [coords[0]]
    for (lon_a, lat_a), (lon_b, lat_b) in zip(coords[:-1], coords[1:], strict=True):
        gap = haversine_distance_m(lat_a=lat_a, lon_a=lon_a, lat_b=lat_b, lon_b=lon_b)
        steps = int(gap // target) + 1  # e.g. 250 m @ 90 m target → 3 sub-segments, 2 inserted points
        for s in range(1, steps):
            frac = s / steps
            out.append((lon_a + (lon_b - lon_a) * frac, lat_a + (lat_b - lat_a) * frac))
        out.append((lon_b, lat_b))
    return out


def densify_edge_geometry(graph: nx.MultiDiGraph, max_spacing_m: float) -> None:
    """Densify every edge's 2D polyline so no vertex gap exceeds ``max_spacing_m``, in place.

    Guarantees the build's strict vertex-spacing invariant (a route line can't shortcut across a block).
    Called per-layer on the BIKE graph only (rail legitimately spans straight between stations). BUILD only.
    """
    for _u, _v, data in graph.edges(data=True):
        geom = data.get(Schema.GEOMETRY)
        if geom is None:
            continue
        coords = [(float(x), float(y)) for x, y in geom.coords]
        data[Schema.GEOMETRY] = LineString(_densify_coords(coords, max_spacing_m=max_spacing_m))


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

    OSMnx stores x=lon, y=lat. Out-of-coverage/nodata cells return NaN and are neutral-filled
    with the graph mean so they don't poison the elevation penalty. BUILD time only (baked in).
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

    Samples the DEM at EVERY vertex (one bulk call) so the artifact fully describes the terrain and
    inference never touches the DEM. Rail/bike carry real polylines; station links stay straight. BUILD only.
    """
    edges = [(u, v, k, d) for u, v, k, d in graph.edges(keys=True, data=True) if d.get(Schema.GEOMETRY) is not None]
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


def _worst_band_vertex(coords3d: list[tuple[float, float, float]], margin_m: float) -> int | None:
    """Index of the interior vertex whose z is FARTHEST outside the [endpoint z] band, or None if all in.

    The band is the two endpoint elevations; a vertex past ``margin_m`` beyond it is a real crest/dip the
    straight endpoint line can't represent — the split point. Endpoints (0, last) are never returned.
    """
    z = np.asarray([c[2] for c in coords3d], dtype=np.float64)
    if len(z) < 3:
        return None
    lo, hi = min(z[0], z[-1]), max(z[0], z[-1])
    over = np.maximum(z - hi, 0.0) + np.maximum(lo - z, 0.0)
    over[0] = over[-1] = 0.0  # never split at an endpoint
    worst = int(np.argmax(over))
    return worst if over[worst] > margin_m else None


def drop_bike_self_loops(graph: nx.MultiDiGraph) -> int:
    """Remove BIKE self-loop edges (from_node == to_node) — routing no-ops from consolidation. Returns count.

    A loop road whose two ends merged into one cluster node collapses to u→u: A* never traverses it (it
    can't lower cost) and its degenerate single-point elevation band can't be validated. Returns #removed.
    """
    loops = [
        (u, v, k) for u, v, k, d in graph.edges(keys=True, data=True) if u == v and d.get(Schema.MODE) == Mode.BIKE
    ]
    graph.remove_edges_from(loops)
    return len(loops)


def split_bike_edges_at_extrema(graph: nx.MultiDiGraph, *, margin_m: float, next_node_id: int) -> int:
    """Split each BIKE edge at its worst out-of-band elevation vertex so every sub-edge's z stays in band.

    A crest/dip mid-edge becomes a NEW node (z baked from that vertex) and the edge is cut there, repeating
    until in band. Returns the next free node id. BUILD only — runs AFTER bake_edge_geometry_elevations.
    """
    queue = [(u, v, k) for u, v, k, d in graph.edges(keys=True, data=True) if d.get(Schema.MODE) == Mode.BIKE]
    while queue:
        u, v, k = queue.pop()
        data = graph.edges[u, v, k]
        coords = [(float(x), float(y), float(zz)) for x, y, zz in data[Schema.GEOMETRY].coords]
        split = _worst_band_vertex(coords, margin_m=margin_m)
        if split is None:
            continue
        mid = next_node_id
        next_node_id += 1
        mx, my, mz = coords[split]
        graph.add_node(mid, x=mx, y=my, elevation=mz, node_type=graph.nodes[u]["node_type"], station_name=None)
        attrs = {kk: vv for kk, vv in data.items() if kk not in (Schema.GEOMETRY, Schema.LENGTH)}
        left, right = coords[: split + 1], coords[split:]
        graph.remove_edge(u, v, k)
        for a, b, seg in ((u, mid, left), (mid, v, right)):
            new_geom = LineString(seg)
            ck = graph.add_edge(a, b, **{**attrs, Schema.GEOMETRY: new_geom, Schema.LENGTH: new_geom.length})
            queue.append((a, b, ck))  # re-check: the sub-edge may still hold a further extremum
    return next_node_id
