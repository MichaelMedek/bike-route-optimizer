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

import networkx as nx
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from shapely import from_wkt, to_wkt
from shapely.geometry import LineString, Polygon

from bike_router.constants import GraphConfig, Mode, NodeType
from bike_router.errors import OutOfCoverageError
from bike_router.progress import ProgressFn, null_progress

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


def _tile_index(lat: float, lon: float, tile_deg: float = GraphConfig.TILE_DEG) -> tuple[int, int]:
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
    row_lo, col_lo = _tile_index(lat=min_lat, lon=min_lon, tile_deg=tile_deg)
    row_hi, col_hi = _tile_index(lat=max_lat, lon=max_lon, tile_deg=tile_deg)
    return [
        (row, col)
        for row in range(row_lo - margin, row_hi + margin + 1)
        for col in range(col_lo - margin, col_hi + margin + 1)
    ]


def compute_bbox(nodes_df: pd.DataFrame) -> tuple[float, float, float, float]:
    """The (west, south, east, north) bounds of a node table — one source for both build scripts."""
    return (
        float(nodes_df["lon"].min()),
        float(nodes_df["lat"].min()),
        float(nodes_df["lon"].max()),
        float(nodes_df["lat"].max()),
    )


def write_graph_parquet(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, meta: dict[str, Any], out_dir: Path) -> None:
    """Write node/edge tables as lat/lon-tiled Parquet + meta.json under ``out_dir``.

    Nodes are tiled by their own coordinate; edges by their ``from_node``'s tile, so a
    corridor read (covering tiles + 1 margin) reliably pulls both endpoints.
    """
    assert list(nodes_df.columns) == _NODE_COLS, f"nodes schema drift: {list(nodes_df.columns)}"
    assert list(edges_df.columns) == _EDGE_COLS, f"edges schema drift: {list(edges_df.columns)}"
    tile_deg = meta["tile_deg"]

    nodes_dir = out_dir / GraphConfig.NODES_SUBDIR
    edges_dir = out_dir / GraphConfig.EDGES_SUBDIR
    nodes_dir.mkdir(parents=True, exist_ok=True)
    edges_dir.mkdir(parents=True, exist_ok=True)

    node_tiles = [
        _tile_index(lat=lat, lon=lon, tile_deg=tile_deg)
        for lat, lon in zip(nodes_df["lat"], nodes_df["lon"], strict=True)
    ]
    nodes_df = nodes_df.assign(_tile=node_tiles)
    coord = {
        osmid: (lat, lon) for osmid, lat, lon in zip(nodes_df["osmid"], nodes_df["lat"], nodes_df["lon"], strict=True)
    }
    edge_tiles = [
        _tile_index(lat=coord[node][0], lon=coord[node][1], tile_deg=tile_deg) for node in edges_df["from_node"]
    ]
    edges_df = edges_df.assign(_tile=edge_tiles)

    for (row, col), group in nodes_df.groupby("_tile"):
        group[_NODE_COLS].to_parquet(nodes_dir / f"{_tile_name(row=row, col=col)}.parquet", index=False)
    for (row, col), group in edges_df.groupby("_tile"):
        group[_EDGE_COLS].to_parquet(edges_dir / f"{_tile_name(row=row, col=col)}.parquet", index=False)

    (out_dir / GraphConfig.META_FILENAME).write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %d nodes / %d edges across tiles to %s", len(nodes_df), len(edges_df), out_dir)


def download_graph_from_hf(target_dir: Path = GraphConfig.GRAPH_DIR, progress: ProgressFn = null_progress) -> Path:
    """Download the prebuilt DACH graph artifact from Hugging Face if missing.

    Fetches the dataset's files one-by-one (nodes/, edges/, meta.json) so ``progress``
    reports genuine (files_done, files_total) — the ONE place the app/CLI show a bar.
    Idempotent: skips entirely once meta.json is already present locally.
    """
    meta_path = target_dir / GraphConfig.META_FILENAME
    if meta_path.exists():
        logger.debug("DACH graph already present at %s", target_dir)
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading DACH graph from HF %s …", GraphConfig.HF_REPO_ID)
    files = list_repo_files(repo_id=GraphConfig.HF_REPO_ID, repo_type="dataset")
    progress(0, len(files))
    for done, filename in enumerate(files, start=1):
        hf_hub_download(
            repo_id=GraphConfig.HF_REPO_ID, repo_type="dataset", filename=filename, local_dir=str(target_dir)
        )
        progress(done, len(files))
    assert meta_path.exists(), "download did not produce meta.json"
    return target_dir


def load_meta(graph_dir: Path = GraphConfig.GRAPH_DIR) -> dict[str, Any]:
    """Read the artifact's meta.json (country bbox, tile grid, tolerance, counts)."""
    meta: dict[str, Any] = json.loads((graph_dir / GraphConfig.META_FILENAME).read_text())
    return meta


def _read_tiles(directory: Path, tiles: list[tuple[int, int]], columns: list[str]) -> pd.DataFrame:
    """Concatenate the existing per-tile Parquet files among ``tiles`` (missing = skip)."""
    frames = [
        pd.read_parquet(directory / f"{_tile_name(row=row, col=col)}.parquet")
        for row, col in tiles
        if (directory / f"{_tile_name(row=row, col=col)}.parquet").exists()
    ]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def graph_from_tables(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.MultiDiGraph:
    """Reconstruct an OSMnx-shaped MultiDiGraph from node/edge tables.

    Uses bulk add_nodes_from / add_edges_from (networkx C internals). Edges whose
    endpoints are both present become graph edges; edges referencing a node outside
    the loaded window are dropped (they dangle off the tile set).
    """
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_nodes_from(
        (
            int(osmid),
            {
                "x": float(lon),
                "y": float(lat),
                "elevation": float(elev),
                "node_type": NodeType(ntype),
                "station_name": name if isinstance(name, str) else None,
            },
        )
        for osmid, lat, lon, elev, ntype, name in zip(
            nodes_df["osmid"],
            nodes_df["lat"],
            nodes_df["lon"],
            nodes_df["elevation_m"],
            nodes_df["node_type"],
            nodes_df["station_name"],
            strict=True,
        )
    )
    present = set(graph.nodes)
    surfaces = edges_df["surface"].map(lambda v: v if isinstance(v, str) else None)
    highways = edges_df["highway"].map(lambda v: v if isinstance(v, str) else None)
    geoms = edges_df["geometry_wkt"].map(lambda w: from_wkt(w) if isinstance(w, str) else None)
    graph.add_edges_from(
        (
            int(u),
            int(v),
            int(k),
            {"length": float(ln), "height_diff": float(hd), "surface": s, "highway": h, "mode": m, "geometry": g},
        )
        for u, v, k, ln, hd, s, h, m, g in zip(
            edges_df["from_node"],
            edges_df["to_node"],
            edges_df["key"],
            edges_df["length_m"],
            edges_df["height_diff_m"],
            surfaces,
            highways,
            edges_df["mode"],
            geoms,
            strict=True,
        )
        if int(u) in present and int(v) in present
    )
    _assert_height_diffs_consistent(graph)
    _assert_node_edge_types_consistent(graph)
    return graph


def _assert_node_edge_types_consistent(graph: nx.MultiDiGraph) -> None:
    """Hard-fail if any edge's endpoints don't match its mode's required node types.

    The core structural guarantee: a BIKE edge joins two bike nodes, a RAIL edge two rail
    nodes, and a STATION edge exactly one of each. So a bike route can NEVER pass through a
    station node — reaching a station always crosses a station edge (which carries boarding).
    Only rules edges the corridor window fully contains; dangling edges were already dropped.
    """
    for u, v, data in graph.edges(data=True):
        tu, tv = graph.nodes[u]["node_type"], graph.nodes[v]["node_type"]
        mode = data["mode"]
        if mode == Mode.BIKE:
            ok = tu == NodeType.BIKE and tv == NodeType.BIKE
        elif mode == Mode.RAIL:
            ok = tu == NodeType.RAIL and tv == NodeType.RAIL
        elif mode == Mode.STATION:
            ok = {tu, tv} == {NodeType.BIKE, NodeType.RAIL}
        else:
            raise AssertionError(f"unknown edge mode {mode!r} on {u}->{v}")
        assert ok, f"{mode} edge {u}->{v} has inconsistent node types {tu}/{tv} — artifact corrupt"


def _assert_height_diffs_consistent(graph: nx.MultiDiGraph) -> None:
    """Hard-fail if any edge's stored height_diff disagrees with its node elevations.

    height_diff_m is a convenience column derivable from the two nodes' elevation_m;
    a mismatch beyond HEIGHT_DIFF_TOLERANCE_M means the artifact is corrupt or stale.
    """
    tol = GraphConfig.HEIGHT_DIFF_TOLERANCE_M
    for u, v, data in graph.edges(data=True):
        expected = graph.nodes[v]["elevation"] - graph.nodes[u]["elevation"]
        assert abs(data["height_diff"] - expected) <= tol, (
            f"height_diff mismatch on edge {u}->{v}: stored {data['height_diff']:.2f} m "
            f"vs nodes {expected:.2f} m (tol {tol} m) — artifact corrupt or stale"
        )


def load_corridor_graph(corridor: Polygon, graph_dir: Path = GraphConfig.GRAPH_DIR) -> nx.MultiDiGraph:
    """Reconstruct the corridor's bike+rail graph from the tiled artifact.

    Reads only the tiles the corridor bbox spans (+1 margin), rebuilds a MultiDiGraph,
    and returns its largest strongly-connected component. Rail edges are always
    included — the sliders alone decide whether A* uses them.
    """
    meta = load_meta(graph_dir=graph_dir)
    tile_deg = meta["tile_deg"]
    tiles = _covering_tiles(bounds=corridor.bounds, tile_deg=tile_deg, margin=1)
    nodes_df = _read_tiles(directory=graph_dir / GraphConfig.NODES_SUBDIR, tiles=tiles, columns=_NODE_COLS)
    edges_df = _read_tiles(directory=graph_dir / GraphConfig.EDGES_SUBDIR, tiles=tiles, columns=_EDGE_COLS)
    assert not nodes_df.empty, "corridor is outside the prebuilt graph coverage (no node tiles)"

    graph = graph_from_tables(nodes_df=nodes_df, edges_df=edges_df)
    assert graph.number_of_edges() > 0, "corridor graph has no edges"
    import osmnx as ox  # local import: only needed here, keeps module import light

    graph = ox.truncate.largest_component(graph, strongly=True)
    assert graph.number_of_nodes() > 0, "corridor graph empty after taking largest component"
    return graph


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
    d2 = (lats - lat) ** 2 + ((lons - lon) * math.cos(math.radians(lat))) ** 2  # planar; fine for nearest
    row = nodes_df.iloc[int(d2.argmin())]
    return float(row["lat"]), float(row["lon"]), float(row["elevation_m"])


def graph_to_tables(graph: nx.MultiDiGraph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten a built MultiDiGraph into node/edge tables matching the on-disk schema.

    ``node_type`` and ``mode`` are internal invariants the builder sets on every node/edge
    (fail loud if not). Bike and rail edges both keep their real polyline; only the short
    station access-links have no geometry (straight).
    """
    nodes = [
        {
            "osmid": int(n),
            "lat": float(d["y"]),  # OSMnx stores y = latitude
            "lon": float(d["x"]),  # x = longitude
            "elevation_m": float(d["elevation"]),
            "node_type": str(d["node_type"]),  # internal invariant: builder types every node
            "station_name": d.get("station_name") if isinstance(d.get("station_name"), str) else None,
        }
        for n, d in graph.nodes(data=True)
    ]
    edges = [
        {
            "from_node": int(u),
            "to_node": int(v),
            "key": int(k),
            "length_m": float(d["length"]),
            "height_diff_m": float(graph.nodes[v]["elevation"] - graph.nodes[u]["elevation"]),
            "surface": _scalar(d.get("surface")),  # unknown → explicit None (external OSM)
            "highway": _scalar(d.get("highway")),  # ditto (genuinely optional)
            "mode": d["mode"],  # internal invariant: builder tags every edge (fail loud if not)
            "geometry_wkt": _geometry_wkt(d.get("geometry")),
        }
        for u, v, k, d in graph.edges(keys=True, data=True)
    ]
    return pd.DataFrame(nodes, columns=_NODE_COLS), pd.DataFrame(edges, columns=_EDGE_COLS)


def _geometry_wkt(geom: object) -> str | None:
    """Real edge polyline as WKT, or None for straight (rail/station) hops."""
    if isinstance(geom, LineString) and len(geom.coords) >= 2:
        return str(to_wkt(geom, rounding_precision=GraphConfig.COORD_PRECISION))
    return None


def _scalar(value: object) -> object:
    """Collapse a list-valued OSM tag to its first element; unknown/empty → None.

    Consolidation merges parallel ways into list-valued surface/highway; routing
    only needs one representative (surface_tier/road_tier handle scalars). An
    absent or empty tag is stored as an explicit None, never a blank string.
    """
    if isinstance(value, list | tuple):
        return value[0] if value else None
    return value if isinstance(value, str) else None
