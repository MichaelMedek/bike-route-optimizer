"""Offline builder: a whole-region bike+rail graph from Geofabrik .osm.pbf extracts.

Builds the cycling network and the rail track network INDEPENDENTLY via one shared pyrosm
builder (only the filter differs), bakes DEM elevation, then merges them into one graph by
wiring each station (a separate rail node) to its nearest bike nodes with station edges.
Returns node/edge tables; the routing sliders decide if A* uses rail.
"""

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyrosm import OSM
from shapely import from_wkt, to_wkt

from bike_router.constants import (
    GraphConfig,
    Mode,
    NodeType,
    RailConfig,
)
from bike_router.elevation import DEMService
from bike_router.geo import haversine_vec
from bike_router.graph_ops import (
    bake_edge_geometry_elevations,
    consolidate_graph,
    contract_interstitial_nodes,
    drop_disallowed_edges,
    enrich_elevations,
    normalize_pyrosm_graph,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerSpec:
    """All that differs between the bike and rail layers — everything else is the SAME pipeline.

    ``build_layer_graph`` runs one identical sequence for both; only these fields vary. There is NO
    per-mode code path — the surface/highway allowlist is a declared flag, not a branch.
    """

    custom_filter: str | None  # None → pyrosm's "cycling" road preset; bracket string → rail ways
    filter_type: str | None  # "keep" for a bracket filter; None for the preset
    mode: Mode  # tag stamped on every edge
    node_type: NodeType  # tag stamped on every node
    surface_allowlist: bool  # bike: drop non-rideable surfaces/highways; rail: no such tags → skip


BIKE_LAYER = LayerSpec(
    custom_filter=None, filter_type=None, mode=Mode.BIKE, node_type=NodeType.BIKE, surface_allowlist=True
)
RAIL_LAYER = LayerSpec(
    custom_filter='["railway"~"rail"]',
    filter_type="keep",
    mode=Mode.RAIL,
    node_type=NodeType.RAIL,
    surface_allowlist=False,
)


def _network_graph(osm: OSM, *, custom_filter: str | None, filter_type: str | None) -> nx.MultiDiGraph:
    """Fetch a routable graph from a pbf: pyrosm ``get_network(nodes=True)`` → ``to_graph``.

    ``custom_filter`` (a bracket string, e.g. ``'["railway"~"rail"]'``, with ``filter_type="keep"``)
    selects WHICH ways are kept (a plain-dict filter is NOT used — it defaults to exclude + highway-
    only, silently returning the whole road net). ``force_bidirectional=True``: every edge exists in
    both directions (same length, opposite elevation delta) — a bike may ride any road up or down,
    and trains run both ways. ``network_type="cycling"`` only sets the (overridden) directionality.
    """
    res = cast(
        "tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | None",
        osm.get_network(network_type="cycling", custom_filter=custom_filter, filter_type=filter_type, nodes=True),
    )
    assert res is not None, "pbf has no matching network"
    nodes, edges = res
    graph: nx.MultiDiGraph = osm.to_graph(
        nodes, edges, graph_type="networkx", osmnx_compatible=True, retain_all=True, force_bidirectional=True
    )
    normalize_pyrosm_graph(graph=graph)
    return graph


def build_layer_graph(osm: OSM, *, layer: LayerSpec, tolerance_m: float) -> nx.MultiDiGraph:
    """Build ONE fully-preprocessed, single-component layer graph for EITHER mode (bike or rail).

    THE one shared pipeline — no per-mode branches, only ``layer`` config differs: fetch ways →
    (bike only) drop non-rideable surfaces/highways → contract degree-2 nodes → consolidate junctions
    (``tolerance_m``) → keep the largest weakly-connected component → tag mode/node_type. Per-step logs
    (prefixed by the layer's mode) show which sub-step runs during the long fetch/consolidate gap.
    """
    graph = _network_graph(osm=osm, custom_filter=layer.custom_filter, filter_type=layer.filter_type)
    logger.info(f"  {layer.mode}: fetched {graph.number_of_nodes()} raw nodes")
    if layer.surface_allowlist:
        drop_disallowed_edges(graph=graph)  # bike: rail has no surface/highway tags, would drop all
        logger.info(f"  {layer.mode}: dropped disallowed → {graph.number_of_edges()} edges")
    graph = contract_interstitial_nodes(graph=graph)
    logger.info(f"  {layer.mode}: contracted degree-2 → {graph.number_of_nodes()} nodes")
    graph = consolidate_graph(graph=graph, tolerance_m=tolerance_m)
    logger.info(f"  {layer.mode}: consolidated → {graph.number_of_nodes()} nodes")
    assert graph.number_of_nodes() > 0
    largest = max(nx.weakly_connected_components(graph), key=len)
    graph = graph.subgraph(largest).copy()
    _tag_layer(graph=graph, mode=layer.mode, node_type=layer.node_type)
    return graph


def _open_osm(pbf_path: Path) -> OSM:
    """Open a pbf for parsing. Bbox clipping (if any) happens upstream via osmium (see stage_pbf)."""
    return OSM(str(pbf_path))


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


def _tag_layer(graph: nx.MultiDiGraph, *, mode: Mode, node_type: NodeType) -> None:
    """Stamp every edge's ``mode`` and every node's ``node_type`` in place (single source of truth).

    Used for both layers via build_layer_graph. Station nodes/edges are stamped in _merge_bike_rail.
    """
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        data["mode"] = mode
    for _node, data in graph.nodes(data=True):
        data["node_type"] = node_type


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


def _merge_bike_rail(bike_graph: nx.MultiDiGraph, rail_graph: nx.MultiDiGraph, osm: OSM) -> int:
    """Merge the independent bike + rail graphs at stations; returns #stations.

    Composes ``rail_graph`` (ids relabelled disjoint) into ``bike_graph``, then adds each station as
    a SEPARATE RAIL node joined to its nearest track node (RAIL edge) and up-to-N nearest bike nodes
    (STATION edges) — so a bike route reaches a station only across a station edge (graph_model.svg).
    Node elevations (bike, track, station) are baked in ONE enrich_elevations pass AFTER this merge.

    Args:
        bike_graph: The tagged cycling graph; mutated in place into the merged graph.
        rail_graph: The tagged rail track graph (consumed, relabelled).
        osm: Open pbf, queried for station points.
    """
    stations = _station_points(osm=osm)
    if not stations:
        logger.warning("No railway stations in extract — rail skipped")
        return 0

    # Snapshot BIKE node coords BEFORE composing rail, so a station's entrances are only bike nodes.
    bike_ids = np.fromiter(bike_graph.nodes, dtype=np.int64)
    assert bike_ids.size, "bike graph must be non-empty before merging rail"
    bike_lats = np.array([bike_graph.nodes[int(n)]["y"] for n in bike_ids])
    bike_lons = np.array([bike_graph.nodes[int(n)]["x"] for n in bike_ids])

    # Relabel rail ids into a compact range disjoint from the bike ids, then compose into one graph.
    offset = int(bike_ids.max()) + 1
    rail_graph = nx.relabel_nodes(rail_graph, {old: offset + i for i, old in enumerate(rail_graph.nodes)}, copy=True)
    bike_graph.add_nodes_from(rail_graph.nodes(data=True))
    bike_graph.add_edges_from(rail_graph.edges(keys=True, data=True))

    # Assign each station a synthetic node id below the OSM id range so it can't clash. Elevation is
    # baked later by the single enrich_elevations pass (with the track nodes), not here.
    have_track = rail_graph.number_of_nodes() > 0
    for station_id, (name, lat, lon) in enumerate(stations, start=1):
        node_id = -station_id
        bike_graph.add_node(node_id, x=lon, y=lat, node_type=NodeType.RAIL, station_name=name)
        # Snap the station onto the track: a RAIL edge to its nearest track node (library nearest_nodes).
        if have_track:
            track_node, snap_dist = ox.distance.nearest_nodes(rail_graph, X=lon, Y=lat, return_dist=True)
            if snap_dist <= RailConfig.STATION_RADIUS_M:
                bike_graph.add_edge(node_id, int(track_node), length=float(snap_dist), mode=Mode.RAIL)
                bike_graph.add_edge(int(track_node), node_id, length=float(snap_dist), mode=Mode.RAIL)
        # Declare the nearest N bike nodes its entrances; each STATION edge costs straight-line length
        # + half the boarding charge (cost.py), so board + alight sum to a full boarding.
        for bike_node, dist in _station_entrances(
            node_ids=bike_ids, node_lats=bike_lats, node_lons=bike_lons, lat=lat, lon=lon
        ):
            bike_graph.add_edge(bike_node, node_id, length=dist, mode=Mode.STATION)
            bike_graph.add_edge(node_id, bike_node, length=dist, mode=Mode.STATION)
    return len(stations)


def _assert_single_component(*, graph: nx.MultiDiGraph) -> None:
    """Fail fast unless the layer graph is exactly ONE weakly-connected component.

    Rail must be one network (every train reaches every other); bike must be one (no stranded
    islands). build_layer_graph already keeps the largest component, so >1 here means a real bug.
    """
    n = nx.number_weakly_connected_components(graph) if graph.number_of_nodes() else 0
    assert n == 1, f"graph must be exactly 1 component, got {n} — build invariant violated"


def build_region_graph(
    *,
    pbf_path: Path,
    dem: DEMService,
    tolerance_m: float,
) -> nx.MultiDiGraph:
    """Build ONE region's consolidated bike+rail graph with baked elevation.

    Bike and rail graphs are built by the SAME ``build_layer_graph`` (only ``LayerSpec`` differs),
    each asserted to be exactly ONE component, and ONLY THEN merged by adding station link edges.
    ``pbf_path`` is parsed whole — any bbox clipping happens upstream (osmium pre-clip in Phase 2).

    Args:
        pbf_path: Geofabrik .osm.pbf extract for the region (already clipped if a sub-region).
        dem: Loaded DEM sampler (node/station elevations are baked here).
        tolerance_m: Intersection consolidation radius (metres); 0 disables.
    """
    osm = _open_osm(pbf_path=pbf_path)
    name = pbf_path.name
    logger.info(f"{name}: [0/6] osm loaded from {pbf_path}")
    # [1] BIKE and [2] RAIL — the identical build_layer_graph pipeline, only LayerSpec differs.
    graph = build_layer_graph(osm=osm, layer=BIKE_LAYER, tolerance_m=tolerance_m)
    _assert_single_component(graph=graph)
    logger.info(f"{name}: [1/6] bike layer → {graph.number_of_nodes()} nodes (1 component)")
    rail_graph = build_layer_graph(osm=osm, layer=RAIL_LAYER, tolerance_m=tolerance_m)
    _assert_single_component(graph=rail_graph)
    logger.info(f"{name}: [2/6] rail layer → {rail_graph.number_of_nodes()} nodes (1 component)")
    # [3] MERGE: only now are the two single-component graphs joined by station link edges.
    n_stations = _merge_bike_rail(bike_graph=graph, rail_graph=rail_graph, osm=osm)
    logger.info(f"{name}: [3/6] merged {rail_graph.number_of_nodes()} rail nodes + {n_stations} stations")
    enrich_elevations(graph=graph, dem=dem)
    logger.info(f"{name}: [4/6] baked node elevations")
    bake_edge_geometry_elevations(graph=graph, dem=dem)  # 3D vertices → inference needs no DEM
    logger.info(f"{name}: [5/6] baked edge geometry elevations")
    # SCC filter AFTER station wiring: guarantees ZERO dangling nodes in the written artifact.
    graph = ox.truncate.largest_component(graph, strongly=True)
    logger.info(
        f"{name}: [6/6] done — {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, {n_stations} stations"
    )
    return graph


def stage_pbf(*, raw_pbf: Path, bbox: tuple[float, float, float, float] | None, staging_dir: Path) -> Path:
    """Stage the parse-ready pbf into ``staging_dir`` and return its path (the ONE file the build reads).

    ``osmium extract`` (C++, ~5 s, multi-core) pre-clips to the bbox with complete_ways so boundary-
    crossing ways keep all nodes — the build then parses only the corridor instead of re-decoding the
    whole country (the ~50-min-per-half cost). Whole regions (bbox=None) are copied as-is.
    """
    staged = staging_dir / raw_pbf.name  # single source of truth for the staged path
    if bbox is None:
        shutil.copy(raw_pbf, staged)
        return staged
    west, south, east, north = bbox
    subprocess.run(  # noqa: S603 — osmium is a trusted, installed C++ tool, fixed args
        [
            "osmium",
            "extract",
            "-b",
            f"{west},{south},{east},{north}",
            "--strategy",
            "complete_ways",
            str(raw_pbf),
            "-o",
            str(staged),
        ],
        check=True,
    )
    return staged


def build_region_graph_clipped(
    *, raw_pbf: Path, dem: DEMService, tolerance_m: float, bbox: tuple[float, float, float, float] | None
) -> nx.MultiDiGraph:
    """Stage (osmium-clip if bbox, else copy) then build — the full clip+build workflow as one unit.

    The clip runs to a temp dir that is auto-removed after parsing, so no staged pbf lingers. Returns
    the same graph as ``build_region_graph`` on the (clipped) pbf.
    """
    with tempfile.TemporaryDirectory() as tmp:
        staged = stage_pbf(raw_pbf=raw_pbf, bbox=bbox, staging_dir=Path(tmp))
        return build_region_graph(pbf_path=staged, dem=dem, tolerance_m=tolerance_m)


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

    Nodes coinciding in (lat, lon, node_type) to COORD_PRECISION collapse to the lower-(lat, lon)
    copy (others repointed onto it) — node_type is IN the key so a bike node and a station/rail node
    at the same spot stay SEPARATE (a station is its own node reachable only via a station edge).
    Edges collapse only if endpoints AND geometry (rounded WKT, or endpoints+mode for null-geometry
    hops) coincide — so genuinely parallel roads both survive.
    """
    prec = GraphConfig.COORD_PRECISION
    nodes_df = nodes_df.copy()
    # Canonical node per rounded (lat, lon, node_type): the row with the lowest (lat, lon), then id.
    # node_type MUST be in the key — else a coincident bike+rail node would merge, leaving a bike
    # edge pointing at a rail node (breaks the type invariant the graph model guarantees).
    nodes_df = nodes_df.sort_values(["lat", "lon", "osmid"], kind="stable")
    key_lat = nodes_df["lat"].round(prec)
    key_lon = nodes_df["lon"].round(prec)
    nodes_df["_key"] = list(zip(key_lat, key_lon, nodes_df["node_type"], strict=True))
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
