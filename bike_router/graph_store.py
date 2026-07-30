"""Prebuilt DACH graph: tiled GeoParquet serialization + windowed corridor load.

The builder tiles the whole DACH bike+rail graph on a coarse lat/lon grid;
inference downloads it once from HF then reads only the corridor's tiles. On-disk
schema uses self-documenting names (nodes/edges below); in-memory keeps OSMnx x/y.

Schema — nodes: osmid, lat, lon, elevation_m, node_type (bike|rail), station_name
(null for bike nodes). edges: from_node, to_node, key, length_m, height_diff_m, surface,
highway, mode, geometry_wkt (WKT LINESTRING; null for straight rail/station hops). Travel
time is NOT stored — derived from length_m + rail constants at route time.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import snapshot_download
from huggingface_hub.utils import tqdm as hf_tqdm
from shapely import covers, from_wkt, points
from shapely.geometry import Polygon, box

from bike_router.constants import GraphConfig, Mode, NodeType, RoutingParams
from bike_router.cost import edge_cost_array
from bike_router.errors import OutOfCoverageError
from bike_router.geo import haversine_vec
from bike_router.progress import ProgressFn, null_progress
from bike_router.route_path import RouteEdge, RouteNode, RoutePath

logger = logging.getLogger(__name__)

_NODE_COLS = ["osmid", "lat", "lon", "elevation_m", "node_type", "station_name"]
_EDGE_COLS = [
    "from_node",
    "to_node",
    "key",
    "length_m",
    "height_diff_m",
    "surface",
    "highway",
    "mode",
    "geometry_wkt",
]


def tile_index(lat: float, lon: float, tile_deg: float = GraphConfig.TILE_DEG) -> tuple[int, int]:
    """(row, col) tile index for a coordinate on the coarse lat/lon grid."""
    return math.floor(lat / tile_deg), math.floor(lon / tile_deg)


def _tile_name(row: int, col: int) -> str:
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


def download_graph_from_hf(target_dir: Path = GraphConfig.GRAPH_DIR, progress: ProgressFn = null_progress) -> Path:
    """Download the prebuilt DACH graph artifact from Hugging Face if missing.

    Uses snapshot_download (concurrent, ``HF_MAX_WORKERS`` files at once) so the ~630-file
    artifact pulls far faster than one-at-a-time. ``progress`` reports genuine
    (files_done, files_total) via the main-thread "Fetching N files" bar — the ONE place the
    app/CLI show a bar. Idempotent: skips entirely once meta.json is already present locally.
    """
    meta_path = target_dir / GraphConfig.META_FILENAME
    if meta_path.exists():
        logger.debug(f"DACH graph already present at {target_dir}")
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading DACH graph from HF {GraphConfig.HF_REPO_ID} …")

    class _FileCountTqdm(hf_tqdm):  # type: ignore[misc]  # hf_tqdm is untyped (Any)
        """Forward ONLY the main-thread 'Fetching N files' bar to ``progress``.

        snapshot_download also creates two byte bars (unit='B') updated from worker threads;
        those are skipped so st.progress is only ever touched on the main thread (off-thread
        st.progress calls silently no-op).
        """

        def update(self, n: float | None = 1) -> bool | None:
            result: bool | None = super().update(n)
            if self.unit != "B" and self.total:
                progress(int(self.n), int(self.total))
            return result

    snapshot_download(
        repo_id=GraphConfig.HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(target_dir),
        max_workers=GraphConfig.HF_MAX_WORKERS,
        tqdm_class=_FileCountTqdm,
    )
    assert meta_path.exists(), "download did not produce meta.json"
    return target_dir


def load_meta(graph_dir: Path = GraphConfig.GRAPH_DIR) -> dict[str, Any]:
    """Read the artifact's meta.json (country bbox, tile grid, tolerance, counts)."""
    meta: dict[str, Any] = json.loads((graph_dir / GraphConfig.META_FILENAME).read_text())
    return meta


def _read_tiles(
    directory: Path,
    columns: list[str],
    tiles: list[tuple[int, int]] | None = None,
    filters: list[tuple[str, str, object]] | None = None,
) -> pd.DataFrame:
    """Concatenate per-tile Parquet files in ``directory`` into one DataFrame.

    ``tiles`` = the specific (row, col) tiles to read (missing skipped) for a corridor window;
    ``tiles=None`` reads EVERY ``tile_*.parquet`` (a whole region/artifact). ``filters`` = optional
    pyarrow predicate pushdown (e.g. by mode/node_type), so a tile yields only matching rows.
    """
    if tiles is None:
        paths = sorted(directory.glob("tile_*.parquet"))
    else:
        paths = [p for row, col in tiles if (p := directory / f"{_tile_name(row=row, col=col)}.parquet").exists()]
    frames = [pd.read_parquet(path, filters=filters) for path in paths]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def _load_layer(
    *,
    corridor: Polygon,
    graph_dir: Path,
    node_type: str,
    edge_modes: list[str],
    node_columns: list[str],
    edge_columns: list[str],
    extra_from_ids: frozenset[int] = frozenset(),
) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    """Read one mode-layer's nodes/edges for a corridor, keeping only nodes inside it.

    Reads only tiles the corridor crosses and only rows of the requested node_type / edge
    modes (parquet pushdown) with only ``node_columns``/``edge_columns`` (column projection).
    Nodes are masked to those the corridor COVERS FIRST, then edges are read with a
    ``from_node in (inside_ids ∪ extra_from_ids)`` pushdown so only corridor edges ever
    materialize (a tile can hold 100k+ edges — masking after a whole-tile read would spike
    memory). ``extra_from_ids`` admits station edges whose from_node is in the OTHER layer
    (bike→rail station links, read in the rail layer). Returns (nodes, edges, inside_ids).
    """
    tile_deg = load_meta(graph_dir=graph_dir)["tile_deg"]
    tiles = _intersecting_tiles(corridor=corridor, tile_deg=tile_deg)
    nodes_df = _read_tiles(
        directory=graph_dir / GraphConfig.NODES_SUBDIR,
        columns=node_columns,
        tiles=tiles,
        filters=[("node_type", "==", node_type)],
    )
    inside_mask = covers(corridor, points(nodes_df["lon"].to_numpy(dtype=float), nodes_df["lat"].to_numpy(dtype=float)))
    nodes_df = nodes_df[inside_mask].reset_index(drop=True)
    inside_ids = set(nodes_df["osmid"].astype(int))
    edges_df = _read_tiles(
        directory=graph_dir / GraphConfig.EDGES_SUBDIR,
        columns=edge_columns,
        tiles=tiles,
        filters=[("mode", "in", edge_modes), ("from_node", "in", list(inside_ids | extra_from_ids))],
    )
    return nodes_df, edges_df, inside_ids


# Minimal columns the CSR router needs: node coords+elev+type, edge endpoints+length+tags+mode.
# geometry_wkt (73% of the edge table) and key/height_diff are read ONLY for the final path.
_ROUTE_NODE_COLS = ["osmid", "lat", "lon", "elevation_m", "node_type"]
_ROUTE_EDGE_COLS = ["from_node", "to_node", "length_m", "surface", "highway", "mode"]


def load_route_tables(
    *,
    bike_corridor: Polygon,
    rail_corridor: Polygon,
    graph_dir: Path = GraphConfig.GRAPH_DIR,
    node_columns: list[str] = _ROUTE_NODE_COLS,
    edge_columns: list[str] = _ROUTE_EDGE_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combined (nodes_df, edges_df) for the two-corridor routing window — the SINGLE combine.

    Tight bike tube + wide rail tube (rail is sparse), recombined with the bike ring, the
    rail↔rail edges, and the station bridges. Default columns are the minimal routing set
    (no geometry) so the corridor load stays memory-lean; callers wanting a full networkx
    graph pass the full schemas. Not component-pruned: a valid multi-component corridor
    (water gap) must not be rejected — the router raises NoRouteError if truly disconnected.
    """
    bike_nodes, bike_edges, bike_ids = _load_layer(
        corridor=bike_corridor,
        graph_dir=graph_dir,
        node_type=NodeType.BIKE,
        edge_modes=[Mode.BIKE],
        node_columns=node_columns,
        edge_columns=edge_columns,
    )
    # Rail layer also carries STATION edges; a bike→rail station link has from_node in the BIKE
    # layer, so admit bike_ids to the from_node pushdown or those links would be filtered out.
    rail_nodes, rail_edges, rail_ids = _load_layer(
        corridor=rail_corridor,
        graph_dir=graph_dir,
        node_type=NodeType.RAIL,
        edge_modes=[Mode.RAIL, Mode.STATION],
        node_columns=node_columns,
        edge_columns=edge_columns,
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


def load_path_edges(
    *, path_nodes: list[tuple[int, float, float]], params: RoutingParams, graph_dir: Path = GraphConfig.GRAPH_DIR
) -> RoutePath:
    """Re-read ONLY the chosen path's edges (with geometry) into an ordered RoutePath.

    The route is an ultra-small subset (hundreds of edges), so re-reading their tiles WITH
    geometry_wkt costs a few MB — vs the ~GB a full-corridor networkx graph would. For each
    consecutive hop the CHEAPEST parallel candidate is kept (recomputed via edge_cost with the
    SAME params the CSR router used — the single cost source), oriented from_node→to_node.

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
        _read_tiles(
            directory=graph_dir / GraphConfig.NODES_SUBDIR,
            columns=_NODE_COLS,
            tiles=tiles,
            filters=[("osmid", "in", list(set(path_osmids)))],
        )
        .set_index("osmid")
        .reindex(path_osmids)
    )
    nodes = [
        RouteNode(
            osmid=int(osmid),
            lat=float(row.lat),
            lon=float(row.lon),
            elevation_m=float(row.elevation_m),
            node_type=str(row.node_type),
            station_name=_str_or_none(value=row.station_name),
        )
        for osmid, row in zip(path_osmids, nodes_df.itertuples(index=False), strict=True)
    ]
    edges_df = _read_tiles(
        directory=graph_dir / GraphConfig.EDGES_SUBDIR,
        columns=_EDGE_COLS,
        tiles=tiles,
        filters=[("from_node", "in", list(set(path_osmids)))],
    )
    return RoutePath(nodes=nodes, edges=_select_path_edges(nodes=nodes, edges_df=edges_df, params=params))


def _select_path_edges(*, nodes: list[RouteNode], edges_df: pd.DataFrame, params: RoutingParams) -> list[RouteEdge]:
    """Cheapest parallel edge per consecutive hop, oriented a→b, as an ordered RouteEdge list.

    Both orientations of each hop are read (edges are directed); the cheapest matching row under
    ``params`` is kept — the SAME min-collapse the CSR router applied — via edge_cost_array (the
    ONE vectorized cost). Costs the whole candidate table at once, then picks per hop.
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
        edges.append(
            RouteEdge(
                from_node=node_a.osmid,
                to_node=node_b.osmid,
                mode=str(row["mode"]),
                length_m=float(row["length_m"]),
                surface=_str_or_none(value=row["surface"]),
                highway=_str_or_none(value=row["highway"]),
                geometry=_oriented_geometry(wkt=row["geometry_wkt"], node_a=node_a),
            )
        )
    return edges


def _oriented_geometry(*, wkt: object, node_a: RouteNode) -> list[tuple[float, float]] | None:
    """WKT LINESTRING → 2D ``[(lon, lat), ...]`` oriented to start at node_a, or None if absent."""
    if not isinstance(wkt, str):
        return None
    coords = [(float(c[0]), float(c[1])) for c in from_wkt(wkt).coords]  # drop any z
    first, last = coords[0], coords[-1]
    if abs(first[0] - node_a.lon) + abs(first[1] - node_a.lat) > abs(last[0] - node_a.lon) + abs(last[1] - node_a.lat):
        coords.reverse()
    return coords


def snap_to_node(lat: float, lon: float, graph_dir: Path = GraphConfig.GRAPH_DIR) -> tuple[float, float, float]:
    """Nearest graph node to (lat, lon) as ``(lat, lon, elevation_m)``.

    Routing is node-to-node, so this resolves a raw geocoded point to the node it
    will actually start/end at — and returns its baked elevation (no DEM needed), so
    the map marker can hover at the true terrain height.
    """
    tile_deg = load_meta(graph_dir=graph_dir)["tile_deg"]
    tiles = _covering_tiles(bounds=(lon, lat, lon, lat), tile_deg=tile_deg, margin=1)
    nodes_df = _read_tiles(directory=graph_dir / GraphConfig.NODES_SUBDIR, tiles=tiles, columns=_NODE_COLS)
    if nodes_df.empty:  # user-facing: a place outside the prebuilt graph's coverage
        raise OutOfCoverageError(f"No routable graph near ({lat:.4f}, {lon:.4f}) — outside the covered region.")
    lats = nodes_df["lat"].to_numpy()
    lons = nodes_df["lon"].to_numpy()
    dists = haversine_vec(lat_a=lat, lon_a=lon, lat_b=lats, lon_b=lons)  # shared vectorized great-circle
    row = nodes_df.iloc[int(dists.argmin())]
    return float(row["lat"]), float(row["lon"]), float(row["elevation_m"])


def _str_or_none(value: object) -> str | None:
    """A str value, else None — the ONE 'non-str/NaN → None' coercion for tag/name columns."""
    return value if isinstance(value, str) else None
