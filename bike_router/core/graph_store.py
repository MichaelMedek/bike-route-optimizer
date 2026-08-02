"""Prebuilt DACH graph: tiled GeoParquet serialization + windowed corridor load.

The builder tiles the whole bike+rail graph on a coarse lat/lon grid; inference downloads it once
from HF then reads only the corridor's tiles (self-documenting nodes/edges schema; travel time derived).
"""

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import list_repo_files, snapshot_download
from shapely import covers, from_wkt, points
from shapely.geometry import Polygon, box

from bike_router.core.constants import GraphConfig, Mode, NodeType, RailConfig, RoutingParams, Schema
from bike_router.core.cost import edge_cost_array
from bike_router.core.errors import OutOfCoverageError
from bike_router.core.geo import haversine_vec, nearest_index
from bike_router.core.progress import ProgressFn
from bike_router.core.route_path import RouteEdge, RouteNode, RoutePath

logger = logging.getLogger(__name__)

_DOWNLOAD_POLL_S = 0.5  # how often the main thread samples on-disk file count for progress

NODE_COLS = [Schema.OSMID, Schema.LAT, Schema.LON, Schema.ELEVATION_M, Schema.NODE_TYPE, Schema.STATION_NAME]
EDGE_COLS = [
    Schema.FROM_NODE,
    Schema.TO_NODE,
    Schema.KEY,
    Schema.LENGTH_M,
    Schema.HEIGHT_DIFF_M,
    Schema.SURFACE,
    Schema.HIGHWAY,
    Schema.MODE,
    Schema.GEOMETRY_WKT,
]
# Minimal columns the CSR routing pass needs (no geometry/station_name — those are re-read per chosen
# edge in load_path_edges). ``key`` stays so the unreliable-elevation sister file joins per parallel edge.
_ROUTE_NODE_COLS = [Schema.OSMID, Schema.LAT, Schema.LON, Schema.ELEVATION_M, Schema.NODE_TYPE]
_ROUTE_EDGE_COLS = [
    Schema.FROM_NODE,
    Schema.TO_NODE,
    Schema.KEY,
    Schema.LENGTH_M,
    Schema.SURFACE,
    Schema.HIGHWAY,
    Schema.MODE,
]
# The sister tile's columns: the (from, to, key) join triple + the baked deviation it contributes.
_UNRELIABLE_COLS = [Schema.FROM_NODE, Schema.TO_NODE, Schema.KEY, Schema.ELEVATION_DEVIATION_M]


def tile_index(lat: float, lon: float, tile_deg: float) -> tuple[int, int]:
    """(row, col) tile index for a coordinate on the coarse lat/lon grid."""
    return math.floor(lat / tile_deg), math.floor(lon / tile_deg)


def tile_name(row: int, col: int) -> str:
    """Filename stem for a tile (negative-safe, e.g. tile_96_16)."""
    return f"tile_{row}_{col}"


def _covering_tiles(bounds: tuple[float, float, float, float], tile_deg: float, margin: int) -> list[tuple[int, int]]:
    """All (row, col) tiles overlapping a bbox, grown by ``margin`` tiles each side.

    The margin catches edges that cross a tile boundary into a neighbour tile.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    row_lo, col_lo = tile_index(lat=min_lat, lon=min_lon, tile_deg=tile_deg)
    row_hi, col_hi = tile_index(lat=max_lat, lon=max_lon, tile_deg=tile_deg)
    return [
        (row, col)
        for row in range(row_lo - margin, row_hi + margin + 1)
        for col in range(col_lo - margin, col_hi + margin + 1)
    ]


def _intersecting_tiles(*, corridor: Polygon, tile_deg: float) -> list[tuple[int, int]]:
    """(row, col) tiles whose grid cell actually overlaps the corridor polygon.

    Far fewer than the corridor bbox for a diagonal route — only the cells the tube
    truly crosses, not the whole bounding rectangle.
    """
    min_lon, min_lat, max_lon, max_lat = corridor.bounds
    row_lo, col_lo = tile_index(lat=min_lat, lon=min_lon, tile_deg=tile_deg)
    row_hi, col_hi = tile_index(lat=max_lat, lon=max_lon, tile_deg=tile_deg)
    return [
        (row, col)
        for row in range(row_lo, row_hi + 1)
        for col in range(col_lo, col_hi + 1)
        if corridor.intersects(box(col * tile_deg, row * tile_deg, (col + 1) * tile_deg, (row + 1) * tile_deg))
    ]


def _warn_if_missing_unreliable_elevation(*, graph_dir: Path) -> None:
    """Warn if the unreliable-elevation sister folder is absent — the deviation penalty is then inert.

    Written post-build by scripts/flag_unreliable_elevation.py; a graph without it routes fine but never
    deprioritizes edges whose baked terrain is unreliable (see cost.edge_deviation_array).
    """
    sister_dir = graph_dir / GraphConfig.UNRELIABLE_ELEVATION_SUBDIR
    if not any(sister_dir.glob(f"tile_*{GraphConfig.TILE_SUFFIX}")):
        logger.warning(
            f"No {GraphConfig.UNRELIABLE_ELEVATION_SUBDIR}/ tiles under {graph_dir} — the unreliable-elevation "
            f"penalty is INERT. Run scripts/flag_unreliable_elevation.py after the build and re-upload the graph."
        )


def download_graph_from_hf(target_dir: Path, progress: ProgressFn) -> Path:
    """Download the prebuilt DACH graph artifact from Hugging Face if missing.

    snapshot_download (HF's xet-coordinated fetch) runs in a worker thread while the main
    thread polls the on-disk file count for progress. Idempotent via meta.json.
    """
    meta_path = target_dir / GraphConfig.META_FILENAME
    if meta_path.exists():
        logger.debug(f"DACH graph already present at {target_dir}")
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading DACH graph from HF {GraphConfig.HF_REPO_ID} …")
        repo_files = list_repo_files(repo_id=GraphConfig.HF_REPO_ID, repo_type="dataset")
        total = len(repo_files)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                snapshot_download,
                repo_id=GraphConfig.HF_REPO_ID,
                repo_type="dataset",
                local_dir=str(target_dir),
                max_workers=GraphConfig.HF_MAX_WORKERS,
            )
            while not future.done():
                done = sum(1 for name in repo_files if (target_dir / name).exists())  # real files, not .cache meta
                progress(done, total)  # main thread: safe for st.progress
                time.sleep(_DOWNLOAD_POLL_S)
            future.result()  # re-raise any download failure
        progress(total, total)
        assert meta_path.exists(), "download did not produce meta.json"
    _warn_if_missing_unreliable_elevation(graph_dir=target_dir)
    return target_dir


def load_meta(graph_dir: Path) -> dict[str, Any]:
    """Read the artifact's meta.json (country bbox, tile grid, tolerance, counts)."""
    meta: dict[str, Any] = json.loads((graph_dir / GraphConfig.META_FILENAME).read_text())
    return meta


def read_tiles(
    directory: Path,
    columns: list[str],
    tiles: list[tuple[int, int]] | None,
    filters: list[tuple[str, str, object]] | None,
) -> pd.DataFrame:
    """Concatenate per-tile Parquet files in ``directory`` into one DataFrame.

    ``tiles`` selects specific (row, col) tiles (missing skipped); ``tiles=None`` reads EVERY
    ``tile_*.parquet``. ``filters`` = optional pyarrow pushdown so a tile yields only matching rows.
    """
    if tiles is None:
        paths = sorted(directory.glob(f"tile_*{GraphConfig.TILE_SUFFIX}"))
    else:
        paths = [
            p
            for row, col in tiles
            if (p := directory / f"{tile_name(row=row, col=col)}{GraphConfig.TILE_SUFFIX}").exists()
        ]
    frames = [pd.read_parquet(path, filters=filters) for path in paths]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def _merge_unreliable_elevation(
    *, edges_df: pd.DataFrame, graph_dir: Path, tiles: list[tuple[int, int]]
) -> pd.DataFrame:
    """Left-join the baked elevation-deviation sister tiles onto ``edges_df`` by (from, to, key).

    The migration writes ``edge_unreliable_elevation/tile_*.parquet`` only where offenders exist; a
    missing folder/tile or unjoined edge reads as 0.0 → no penalty (see cost.edge_deviation_array).
    """
    sister = read_tiles(
        directory=graph_dir / GraphConfig.UNRELIABLE_ELEVATION_SUBDIR,
        columns=_UNRELIABLE_COLS,
        tiles=tiles,
        filters=None,
    )
    if sister.empty:
        edges_df[Schema.ELEVATION_DEVIATION_M] = 0.0
        return edges_df
    merged = edges_df.merge(sister, on=[Schema.FROM_NODE, Schema.TO_NODE, Schema.KEY], how="left")
    merged[Schema.ELEVATION_DEVIATION_M] = merged[Schema.ELEVATION_DEVIATION_M].fillna(0.0)
    return merged


def _load_layer(
    *,
    corridor: Polygon,
    graph_dir: Path,
    node_type: str,
    edge_modes: list[str],
    node_columns: list[str],
    edge_columns: list[str],
    extra_from_ids: frozenset[int],
) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    """Read one mode-layer's nodes/edges for a corridor, keeping only nodes inside it.

    Tile/type/mode pushdown + column projection; nodes masked to those the corridor COVERS, then a
    ``from_node in (inside_ids ∪ extra_from_ids)`` edge pushdown avoids materializing 100k-edge tiles.
    """
    tile_deg = load_meta(graph_dir=graph_dir)["tile_deg"]
    tiles = _intersecting_tiles(corridor=corridor, tile_deg=tile_deg)
    nodes_df = read_tiles(
        directory=graph_dir / GraphConfig.NODES_SUBDIR,
        columns=node_columns,
        tiles=tiles,
        filters=[(Schema.NODE_TYPE, "==", node_type)],
    )
    inside_mask = covers(corridor, points(nodes_df["lon"].to_numpy(dtype=float), nodes_df["lat"].to_numpy(dtype=float)))
    nodes_df = nodes_df[inside_mask].reset_index(drop=True)
    inside_ids = set(nodes_df["osmid"].astype(int))
    edges_df = read_tiles(
        directory=graph_dir / GraphConfig.EDGES_SUBDIR,
        columns=edge_columns,
        tiles=tiles,
        filters=[(Schema.MODE, "in", edge_modes), (Schema.FROM_NODE, "in", list(inside_ids | extra_from_ids))],
    )
    edges_df = _merge_unreliable_elevation(edges_df=edges_df, graph_dir=graph_dir, tiles=tiles)
    return nodes_df, edges_df, inside_ids


def load_route_tables(
    *,
    bike_corridor: Polygon,
    rail_corridor: Polygon,
    graph_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combined (nodes_df, edges_df) for the two-corridor routing window — the SINGLE combine.

    Tight bike tube + wide sparse rail tube, recombined with bike ring, rail↔rail, and station
    bridges; the minimal _ROUTE_*_COLS stay memory-lean. Not component-pruned (a water-gap is valid).
    """
    bike_nodes, bike_edges, bike_ids = _load_layer(
        corridor=bike_corridor,
        graph_dir=graph_dir,
        node_type=NodeType.BIKE,
        edge_modes=[Mode.BIKE],
        node_columns=_ROUTE_NODE_COLS,
        edge_columns=_ROUTE_EDGE_COLS,
        extra_from_ids=frozenset(),
    )
    # Rail layer also carries STATION edges; a bike→rail station link has from_node in the BIKE
    # layer, so admit bike_ids to the from_node pushdown or those links would be filtered out.
    rail_nodes, rail_edges, rail_ids = _load_layer(
        corridor=rail_corridor,
        graph_dir=graph_dir,
        node_type=NodeType.RAIL,
        edge_modes=[Mode.RAIL, Mode.STATION],
        node_columns=_ROUTE_NODE_COLS,
        edge_columns=_ROUTE_EDGE_COLS,
        extra_from_ids=frozenset(bike_ids),
    )
    assert not bike_nodes.empty, "bike corridor is outside the prebuilt graph coverage (no node tiles)"

    from_in = rail_edges["from_node"].astype(int)
    to_in = rail_edges["to_node"].astype(int)
    is_station = rail_edges["mode"] == Mode.STATION
    # rail↔rail edges: both endpoints in the rail tube; station edges: bike end in bike tube AND
    # rail end in rail tube (the two ids sets are disjoint, so membership picks the right endpoint).
    keep_rail = (~is_station) & from_in.isin(rail_ids) & to_in.isin(rail_ids)
    keep_station = is_station & (
        (from_in.isin(bike_ids) & to_in.isin(rail_ids)) | (from_in.isin(rail_ids) & to_in.isin(bike_ids))
    )
    rail_edges = rail_edges[keep_rail | keep_station]

    bf, bt = bike_edges["from_node"].astype(int), bike_edges["to_node"].astype(int)
    bike_edges = bike_edges[bf.isin(bike_ids) & bt.isin(bike_ids)]

    nodes_df = pd.concat([bike_nodes, rail_nodes], ignore_index=True)
    edges_df = pd.concat([bike_edges, rail_edges], ignore_index=True)
    return nodes_df, edges_df


def load_path_edges(*, path_nodes: list[tuple[int, float, float]], params: RoutingParams, graph_dir: Path) -> RoutePath:
    """Re-read ONLY the chosen path's edges (with geometry) into an ordered RoutePath.

    Re-reading the tiny path's tiles WITH geometry costs a few MB vs the ~GB of a full networkx graph;
    per hop the CHEAPEST parallel candidate is kept via edge_cost with the SAME params the router used.

    Args:
        path_nodes: the route's (osmid, lat, lon) in order — coords give the tiles to read.
        params: the rider's preferences (so the re-read picks the same parallel edge as routing).
        graph_dir: prebuilt-graph dir.
    """
    assert len(path_nodes) >= 2, "path must have >= 2 nodes"
    path_osmids = [int(osmid) for osmid, _lat, _lon in path_nodes]
    tile_deg = load_meta(graph_dir=graph_dir)["tile_deg"]
    tiles = sorted(
        {
            tile
            for _osmid, lat, lon in path_nodes
            for tile in _covering_tiles(bounds=(lon, lat, lon, lat), tile_deg=tile_deg, margin=1)
        }
    )
    # Read the path's nodes, index by osmid, and reindex to the route order (pandas — no py loop).
    nodes_df = (
        read_tiles(
            directory=graph_dir / GraphConfig.NODES_SUBDIR,
            columns=NODE_COLS,
            tiles=tiles,
            filters=[(Schema.OSMID, "in", list(set(path_osmids)))],
        )
        .set_index(Schema.OSMID)
        .reindex(path_osmids)
    )
    nodes = [
        RouteNode(
            osmid=int(osmid),
            lat=float(row.lat),
            lon=float(row.lon),
            elevation_m=float(row.elevation_m),
            node_type=str(row.node_type),
            station_name=str_or_none(value=row.station_name),
        )
        for osmid, row in zip(path_osmids, nodes_df.itertuples(index=False), strict=True)
    ]
    edges_df = read_tiles(
        directory=graph_dir / GraphConfig.EDGES_SUBDIR,
        columns=EDGE_COLS,
        tiles=tiles,
        filters=[(Schema.FROM_NODE, "in", list(set(path_osmids)))],
    )
    edges_df = _merge_unreliable_elevation(edges_df=edges_df, graph_dir=graph_dir, tiles=tiles)
    return RoutePath(nodes=nodes, edges=_select_path_edges(nodes=nodes, edges_df=edges_df, params=params))


def _select_path_edges(*, nodes: list[RouteNode], edges_df: pd.DataFrame, params: RoutingParams) -> list[RouteEdge]:
    """Cheapest parallel edge per consecutive hop, oriented a→b, as an ordered RouteEdge list.

    Both directed orientations are read; the cheapest matching row under ``params`` is kept — the SAME
    min-collapse the CSR router applied — via edge_cost_array, costing the whole table then picking per hop.
    """
    elev = {node.osmid: node.elevation_m for node in nodes}
    known = edges_df["from_node"].isin(elev) & edges_df["to_node"].isin(elev)
    df = edges_df[known].reset_index(drop=True)
    costs = edge_cost_array(edges_df=df, elev_by_osmid=elev, params=params)
    # Cheapest (from, to) → row index across all parallel candidates (vectorized cost, then min per pair).
    best: dict[tuple[int, int], tuple[float, int]] = {}
    for i, (u, v, c) in enumerate(zip(df["from_node"], df["to_node"], costs, strict=True)):
        pair = (int(u), int(v))
        if pair not in best or c < best[pair][0]:
            best[pair] = (float(c), i)

    edges: list[RouteEdge] = []
    for node_a, node_b in zip(nodes[:-1], nodes[1:], strict=True):
        chosen = best.get((node_a.osmid, node_b.osmid))
        assert chosen is not None, f"no edge found for path hop {node_a.osmid}->{node_b.osmid}"
        row = df.iloc[chosen[1]]
        geometry, geometry_z = _oriented_geometry(wkt=row["geometry_wkt"], node_a=node_a)
        edges.append(
            RouteEdge(
                from_node=node_a.osmid,
                to_node=node_b.osmid,
                mode=str(row["mode"]),
                length_m=float(row["length_m"]),
                surface=str_or_none(value=row["surface"]),
                highway=str_or_none(value=row["highway"]),
                geometry=geometry,
                geometry_z=geometry_z,
            )
        )
    return edges


def _oriented_geometry(
    *, wkt: object, node_a: RouteNode
) -> tuple[list[tuple[float, float]] | None, list[float] | None]:
    """WKT LINESTRING → (2D ``[(lon, lat), ...]``, baked z per vertex) oriented to start at node_a.

    Returns (None, None) when absent. The z list (real baked elevation) lets the display warn when
    the linear node-to-node interpolation deviates far from the true terrain on a long edge.
    """
    if not isinstance(wkt, str):
        return None, None
    raw = list(from_wkt(wkt).coords)
    coords = [(float(c[0]), float(c[1])) for c in raw]
    zs = [float(c[2]) if len(c) >= 3 else float("nan") for c in raw]
    first, last = coords[0], coords[-1]
    if abs(first[0] - node_a.lon) + abs(first[1] - node_a.lat) > abs(last[0] - node_a.lon) + abs(last[1] - node_a.lat):
        coords.reverse()
        zs.reverse()
    return coords, zs


def snap_to_node(lat: float, lon: float, graph_dir: Path) -> tuple[float, float, float]:
    """Nearest graph node to (lat, lon) as ``(lat, lon, elevation_m)``.

    Routing is node-to-node, so this resolves a raw geocoded point to its actual start/end node
    and returns that node's baked elevation (no DEM), so the map marker hovers at true terrain height.
    """
    tile_deg = load_meta(graph_dir=graph_dir)["tile_deg"]
    tiles = _covering_tiles(bounds=(lon, lat, lon, lat), tile_deg=tile_deg, margin=1)
    nodes_df = read_tiles(directory=graph_dir / GraphConfig.NODES_SUBDIR, columns=NODE_COLS, tiles=tiles, filters=None)
    if nodes_df.empty:  # user-facing: a place outside the prebuilt graph's coverage
        raise OutOfCoverageError(f"No routable graph near ({lat:.4f}, {lon:.4f}) — outside the covered region.")
    lats = nodes_df["lat"].to_numpy()
    lons = nodes_df["lon"].to_numpy()
    row = nodes_df.iloc[nearest_index(lat=lat, lon=lon, lats=lats, lons=lons)]  # shared nearest-point snap
    return float(row["lat"]), float(row["lon"]), float(row["elevation_m"])


def top_stations(
    graph_dir: Path,
) -> list[tuple[float, float, float, str]]:
    """Prominent local-high rail stations across the coverage area — trip-inspiration "top" stops.

    A station is a top iff it has full Dominanz within TOP_STATION_DOMINANCE_KM AND clears
    TOP_STATION_PROMINENCE_M of Schartenhöhe. Returns (lat, lon, elevation_m, name), highest first.
    """
    nodes_df = read_tiles(directory=graph_dir / GraphConfig.NODES_SUBDIR, columns=NODE_COLS, tiles=None, filters=None)
    stations = nodes_df[(nodes_df["node_type"] == NodeType.RAIL) & nodes_df["station_name"].notna()].reset_index(
        drop=True
    )
    assert not stations.empty, "no station found"
    lats = stations["lat"].to_numpy(dtype=float)
    lons = stations["lon"].to_numpy(dtype=float)
    elevs = stations["elevation_m"].to_numpy(dtype=float)
    tops: list[tuple[float, float, float, str]] = []
    for i in range(len(stations)):
        dists_km = haversine_vec(lat_a=lats[i], lon_a=lons[i], lat_b=lats, lon_b=lons) / 1000.0
        near = elevs[dists_km <= RailConfig.TOP_STATION_DOMINANCE_KM]
        dominant = elevs[i] >= near.max()  # Dominanz: highest station within the radius
        prominent = elevs[i] - near.min() >= RailConfig.TOP_STATION_PROMINENCE_M  # Schartenhöhe: local relief
        if dominant and prominent:
            tops.append((float(lats[i]), float(lons[i]), float(elevs[i]), str(stations["station_name"].iloc[i])))
    return sorted(tops, key=lambda s: s[2], reverse=True)


def str_or_none(value: object) -> str | None:
    """A str value, else None — the ONE 'non-str/NaN → None' coercion for tag/name columns."""
    return value if isinstance(value, str) else None
