"""Offline builder: a whole-region bike+rail graph from Geofabrik .osm.pbf extracts.

Builds cycling + rail networks INDEPENDENTLY via one shared pyrosm builder, bakes DEM elevation, then
merges them by wiring each station to its nearest bike nodes. Returns node/edge tables.
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
import pyproj
from pyrosm import OSM
from shapely import from_wkt, to_wkt
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from bike_router.core.constants import (
    NAME_KEY,
    WGS84_CRS,
    BuildValidationConfig,
    GraphConfig,
    Mode,
    NodeType,
    RailConfig,
    Schema,
)
from bike_router.core.geo import haversine_distance_m, haversine_vec
from bike_router.preprocessing.elevation import DEMService
from bike_router.preprocessing.graph_ops import (
    bake_edge_geometry_elevations,
    consolidate_graph,
    densify_edge_geometry,
    drop_bike_self_loops,
    drop_disallowed_edges,
    enrich_elevations,
    normalize_pyrosm_graph,
    split_bike_edges_at_extrema,
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
    # Routable passenger rail from RailConfig.RAIL_TAGS (single source). Anchored so
    # "abandoned"/"disused" variants don't match; tram + funicular stay excluded (urban, not intercity).
    custom_filter=f'["railway"~"^({"|".join(RailConfig.RAIL_TAGS)})$"]',
    filter_type="keep",
    mode=Mode.RAIL,
    node_type=NodeType.RAIL,
    surface_allowlist=False,
)


def _network_graph(osm: OSM, *, custom_filter: str | None, filter_type: str | None) -> nx.MultiDiGraph:
    """Fetch a routable graph from a pbf: pyrosm ``get_network(nodes=True)`` → ``to_graph``.

    A bracket ``custom_filter`` + ``filter_type="keep"`` selects WHICH ways survive (a plain-dict filter
    silently returns the whole road net). ``force_bidirectional=True``: bikes ride up/down, trains run both.
    """
    logger.info(f"    get_network (filter={custom_filter}) — parsing pbf ways/nodes ...")
    res = cast(
        "tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | None",
        osm.get_network(network_type="cycling", custom_filter=custom_filter, filter_type=filter_type, nodes=True),
    )
    assert res is not None, "pbf has no matching network"
    nodes, edges = res
    logger.info(f"    get_network → {len(nodes)} nodes / {len(edges)} edges; building networkx graph ...")
    graph: nx.MultiDiGraph = osm.to_graph(
        nodes,
        edges,
        graph_type="networkx",
        osmnx_compatible=True,
        retain_all=True,
        force_bidirectional=True,
        network_type="cycling",  # explicit: keep_metadata=False strips the columns to_graph auto-detects from
        simplify=True,  # contract degree-2 chains DURING build — never materialize the raw-node graph
    )
    normalize_pyrosm_graph(graph=graph)
    return graph


def build_layer_graph(osm: OSM, *, layer: LayerSpec, tolerance_m: float) -> nx.MultiDiGraph:
    """Build ONE preprocessed layer graph for EITHER mode (bike or rail) — ALL components kept.

    THE one shared pipeline (no per-mode branch, only ``layer`` config): fetch degree-2-contracted ways →
    (bike only) drop non-rideable → consolidate junctions → tag. ALL components kept; global truncation is Phase 3.
    """
    graph = _network_graph(osm=osm, custom_filter=layer.custom_filter, filter_type=layer.filter_type)
    logger.info(f"  {layer.mode}: fetched {graph.number_of_nodes()} simplified nodes")
    if layer.surface_allowlist:
        drop_disallowed_edges(graph=graph)  # bike: rail has no surface/highway tags, would drop all
        logger.info(f"  {layer.mode}: dropped disallowed → {graph.number_of_edges()} edges")
    graph = consolidate_graph(graph=graph, tolerance_m=tolerance_m)
    logger.info(f"  {layer.mode}: consolidated → {graph.number_of_nodes()} nodes")
    assert graph.number_of_nodes() > 0
    if layer.surface_allowlist:  # bike only: densify so no vertex gap exceeds the strict build invariant
        densify_edge_geometry(graph=graph, max_spacing_m=BuildValidationConfig.MAX_VERTEX_SPACING_M)
    _tag_layer(graph=graph, mode=layer.mode, node_type=layer.node_type)
    return graph


def _open_osm(pbf_path: Path) -> OSM:
    """Open a pbf for parsing, dropping unused metadata to speed the parse. Bbox clip is upstream.

    ``keep_metadata=False`` drops version/timestamp/changeset (unused); routing tags stay. Uses the in-memory
    engine deliberately — the out-of-core engine spills to disk and measured SLOWER on dense slices.
    """
    return OSM(str(pbf_path), keep_metadata=False)


def _station_points(osm: OSM) -> list[tuple[str, float, float]]:
    """Extract (name, lat, lon) for railway stations/halts; empty list if none.

    Polygon/way stations are reduced to their centroid so every stop is one point.
    """
    stations = osm.get_data_by_custom_criteria(
        custom_filter={"railway": list(RailConfig.STATION_TAGS)},
        filter_type="keep",
        tags_as_columns=[NAME_KEY],  # promote the OSM name tag to a column
        keep_nodes=True,
        keep_ways=True,
        keep_relations=False,
    )
    if stations is None or stations.empty:
        return []
    points = stations.geometry.representative_point()
    names = stations[NAME_KEY] if NAME_KEY in stations.columns else [None] * len(stations)
    return [
        (name if isinstance(name, str) else Mode.STATION, float(pt.y), float(pt.x))
        for name, pt in zip(names, points, strict=True)
    ]


def _tag_layer(graph: nx.MultiDiGraph, *, mode: Mode, node_type: NodeType) -> None:
    """Stamp every edge's ``mode`` and every node's ``node_type`` + ``station_name`` (single source of truth).

    ``station_name=None`` on ALL layer nodes keeps the attribute present on every node (real station
    names are stamped later in _merge_bike_rail) so graph_to_tables can read it strictly.
    """
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        data["mode"] = mode
    for _node, data in graph.nodes(data=True):
        data["node_type"] = node_type
        data["station_name"] = None


def _station_entrances(
    node_ids: "np.ndarray", node_lats: "np.ndarray", node_lons: "np.ndarray", lat: float, lon: float
) -> list[tuple[int, float]]:
    """Up to MAX_ENTRANCES bike nodes inside the station radius, as (node id, distance m).

    The nearest N bike nodes within STATION_RADIUS_M (fewer if fewer exist) become the station's
    ENTRANCES: reaching one and crossing its station edge puts the rider at the station. Empty if none.
    """
    dists = haversine_vec(lat_a=lat, lon_a=lon, lat_b=node_lats, lon_b=node_lons)
    within = np.flatnonzero(dists <= RailConfig.STATION_RADIUS_M)
    nearest = within[np.argsort(dists[within])[: RailConfig.STATION_MAX_ENTRANCES]]
    return [(int(node_ids[i]), float(dists[i])) for i in nearest]


def _nearest_tracks(
    *, rail_graph: nx.MultiDiGraph, rail_proj: nx.MultiDiGraph, lats: "np.ndarray", lons: "np.ndarray"
) -> list[tuple[int, float, float]]:
    """Snap MANY (lat, lon) points to their nearest track EDGE via ONE vectorized ``nearest_edges``.

    Per point returns (endpoint_node, node_dist_m, line_dist_m); the query runs on the PROJECTED graph
    (Euclidean nearest_edges mis-picks on lat/lon at ~48°N). ``line_dist_m`` gates on-network membership.
    """
    tr = pyproj.Transformer.from_crs(WGS84_CRS, rail_proj.graph["crs"], always_xy=True)
    px, py = tr.transform(lons, lats)  # vectorized reprojection
    edges = ox.distance.nearest_edges(
        rail_proj, X=np.asarray(px), Y=np.asarray(py)
    )  # ONE R-tree query → array of (u,v,k)
    results: list[tuple[int, float, float]] = []
    for (u, v, key), lat, lon in zip(edges, lats, lons, strict=True):
        node = min(
            (u, v),
            key=lambda t: haversine_distance_m(
                lat_a=lat, lon_a=lon, lat_b=rail_graph.nodes[t]["y"], lon_b=rail_graph.nodes[t]["x"]
            ),
        )
        node_dist = haversine_distance_m(
            lat_a=lat, lon_a=lon, lat_b=rail_graph.nodes[node]["y"], lon_b=rail_graph.nodes[node]["x"]
        )
        geom = rail_graph.get_edge_data(u, v)[key].get(Schema.GEOMETRY)
        if geom is None:
            geom = LineString(
                [
                    (rail_graph.nodes[u]["x"], rail_graph.nodes[u]["y"]),
                    (rail_graph.nodes[v]["x"], rail_graph.nodes[v]["y"]),
                ]
            )
        proj, _pt = nearest_points(geom, Point(lon, lat))
        line_dist = haversine_distance_m(lat_a=lat, lon_a=lon, lat_b=proj.y, lon_b=proj.x)
        results.append((int(node), node_dist, line_dist))
    return results


def _merge_bike_rail(bike_graph: nx.MultiDiGraph, rail_graph: nx.MultiDiGraph, osm: OSM) -> int:
    """Merge the independent bike + rail graphs at stations; returns #stations.

    Composes ``rail_graph`` (ids relabelled disjoint) into ``bike_graph``, then adds each station as a SEPARATE
    RAIL node wired to its nearest track node (RAIL edge) + up-to-N nearest bike nodes (STATION edges). Elevations baked after.

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

    # Project the (relabelled) rail graph ONCE and snap ALL stations to their nearest track edge in ONE
    # vectorized nearest_edges call (a per-station loop rebuilds the R-tree each time — minutes/region).
    rail_proj = ox.projection.project_graph(rail_graph) if rail_graph.number_of_nodes() else None
    if rail_proj is not None:
        st_lats = np.array([lat for _n, lat, _lon in stations])
        st_lons = np.array([lon for _n, _lat, lon in stations])
        snaps: list[tuple[int, float, float] | None] = list(
            _nearest_tracks(rail_graph=rail_graph, rail_proj=rail_proj, lats=st_lats, lons=st_lons)
        )
    else:
        snaps = [None] * len(stations)  # station-only region (no track): wire bike entrances only

    n_wired = 0
    for station_id, ((name, lat, lon), snap) in enumerate(zip(stations, snaps, strict=True), start=1):
        # A station|halt point off the routable-rail network (>STATION_RADIUS_M by TRUE point-to-line
        # distance) is a tram/funicular/park stop we don't route on — DROP it (logged), don't orphan.
        if snap is not None:
            track_node, node_dist, line_dist = snap
            if line_dist > RailConfig.STATION_RADIUS_M:
                logger.info(f"  station dropped (off rail): {name!r} — {line_dist:.0f} m to nearest rail line")
                continue
        node_id = -station_id
        bike_graph.add_node(node_id, x=lon, y=lat, node_type=NodeType.RAIL, station_name=name)
        # Wire the station (kept at its RAW position) onto the rail graph via the nearer endpoint of its
        # nearest track EDGE — gated on point-to-LINE distance, not node distance (consolidation thins nodes).
        if snap is not None:
            bike_graph.add_edge(node_id, int(track_node), length=node_dist, mode=Mode.RAIL)
            bike_graph.add_edge(int(track_node), node_id, length=node_dist, mode=Mode.RAIL)
        # Declare the nearest N bike nodes its entrances; each STATION edge costs straight-line length
        # + half the boarding charge (cost.py), so board + alight sum to a full boarding.
        entrances = _station_entrances(node_ids=bike_ids, node_lats=bike_lats, node_lons=bike_lons, lat=lat, lon=lon)
        # A station with NO bike node within radius is rail-reachable but has no bike entrance — WARN,
        # not fail: rural halts sit on the rail with the nearest mapped road 200–500 m away (OSM sparsity).
        # It stays train-only; a neighbouring region may still supply an entrance after Phase-3 stitching.
        if not entrances:
            logger.warning(
                f"station {name!r} at ({lat:.5f}, {lon:.5f}) has no bike node within "
                f"{RailConfig.STATION_RADIUS_M:.0f} m — kept as train-only (no bike entrance)"
            )
        for bike_node, dist in entrances:
            bike_graph.add_edge(bike_node, node_id, length=dist, mode=Mode.STATION)
            bike_graph.add_edge(node_id, bike_node, length=dist, mode=Mode.STATION)
        n_wired += 1
    return n_wired


def build_region_graph(
    *,
    pbf_path: Path,
    dem: DEMService,
    tolerance_m: float,
) -> nx.MultiDiGraph:
    """Build ONE region's consolidated bike+rail graph with baked elevation.

    Bike and rail are built by the SAME ``build_layer_graph`` (only ``LayerSpec`` differs), each keeping
    ALL components (global truncation is Phase 3), then merged at stations. ``pbf_path`` is parsed whole.

    Args:
        pbf_path: Geofabrik .osm.pbf extract for the region (already clipped if a sub-region).
        dem: Loaded DEM sampler (node/station elevations are baked here).
        tolerance_m: Intersection consolidation radius (metres); 0 disables.
    """
    osm = _open_osm(pbf_path=pbf_path)
    name = pbf_path.name
    logger.info(f"{name}: [1/6] osm loaded from {pbf_path}")
    # [1] BIKE and [2] RAIL — the identical build_layer_graph pipeline, only LayerSpec differs. Neither
    # is truncated to one component here: a region is a CLIP whose fringe connects via neighbours; the
    # single global largest_component runs in Phase 3 after seams are stitched.
    graph = build_layer_graph(osm=osm, layer=BIKE_LAYER, tolerance_m=tolerance_m)
    logger.info(f"{name}: [2/6] bike layer → {graph.number_of_nodes()} nodes")
    rail_graph = build_layer_graph(osm=osm, layer=RAIL_LAYER, tolerance_m=tolerance_m)
    logger.info(f"{name}: [3/6] rail layer → {rail_graph.number_of_nodes()} nodes")
    # [3] MERGE: join the two graphs by station link edges.
    n_stations = _merge_bike_rail(bike_graph=graph, rail_graph=rail_graph, osm=osm)
    logger.info(f"{name}: [4/6] merged {rail_graph.number_of_nodes()} rail nodes + {n_stations} stations")
    enrich_elevations(graph=graph, dem=dem)
    logger.info(f"{name}: [5/6] baked node elevations")
    bake_edge_geometry_elevations(graph=graph, dem=dem)  # 3D vertices → inference needs no DEM
    logger.info(f"{name}: [6/6] baked edge geometry elevations")
    # Drop bike self-loops (consolidation artifacts, routing no-ops), then split remaining bike edges at
    # real elevation extrema so every sub-edge's z stays within its endpoint band — strict build invariant.
    n_loops = drop_bike_self_loops(graph=graph)
    if n_loops:
        logger.info(f"{name}: dropped {n_loops} bike self-loops")
    next_id = max(graph.nodes) + 1
    split_bike_edges_at_extrema(graph=graph, margin_m=BuildValidationConfig.ELEV_BAND_MARGIN_M, next_node_id=next_id)
    return graph


def stage_pbf(*, raw_pbf: Path, bbox: tuple[float, float, float, float] | None, staging_dir: Path) -> Path:
    """Stage the parse-ready pbf into ``staging_dir`` and return its path (the ONE file the build reads).

    ``osmium extract`` (C++, ~5 s) pre-clips to the bbox with complete_ways so boundary-crossing ways keep
    all nodes — the build parses only the corridor, not the whole country. Whole regions (bbox=None) copied as-is.
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

    Nodes coinciding in (lat, lon, node_type) to COORD_PRECISION collapse to the lower copy — node_type is IN
    the key so coincident bike and rail nodes stay SEPARATE. Edges collapse only if endpoints AND geometry coincide.
    """
    prec = GraphConfig.COORD_PRECISION
    nodes_df = nodes_df.copy()
    # Canonical node per rounded (lat, lon, node_type): the row with the lowest (lat, lon), then id.
    # node_type MUST be in the key — else a coincident bike+rail node would merge, leaving a bike
    # edge pointing at a rail node (breaks the type invariant the graph model guarantees).
    nodes_df = nodes_df.sort_values([Schema.LAT, Schema.LON, Schema.OSMID], kind="stable")
    key_lat = nodes_df["lat"].round(prec)
    key_lon = nodes_df["lon"].round(prec)
    nodes_df["_key"] = list(zip(key_lat, key_lon, nodes_df["node_type"], strict=True))
    canonical = nodes_df.groupby("_key")["osmid"].transform("first")
    repoint = dict(zip(nodes_df["osmid"], canonical, strict=True))  # every id → its kept id
    kept_nodes = nodes_df.drop_duplicates(subset="_key", keep="first").drop(columns="_key")

    edges_df = edges_df.copy()
    edges_df["from_node"] = edges_df["from_node"].map(repoint)
    edges_df["to_node"] = edges_df["to_node"].map(repoint)
    # Repointing can collapse a short bike edge's two endpoints onto ONE canonical node → a u→u self-loop
    # (routing no-op, degenerate elevation band). Drop bike self-loops; rail/station loops are harmless/kept.
    self_loop = (edges_df["from_node"] == edges_df["to_node"]) & (edges_df["mode"] == Mode.BIKE)
    edges_df = edges_df[~self_loop].reset_index(drop=True)

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
