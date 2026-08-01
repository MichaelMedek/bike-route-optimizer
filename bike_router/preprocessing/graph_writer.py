"""Build-time GeoParquet writer + MultiDiGraph ↔ tables round-trip (dataset preprocessing).

Runs ONLY offline (region build, Phase-3 combine, Phase-4 validation) and pulls in networkx; inference
never imports this, keeping the runtime import graph networkx-light. Tile helpers/schemas live in graph_store.
"""

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from shapely import from_wkt, to_wkt
from shapely.geometry import LineString

from bike_router.core.constants import WGS84_CRS, GraphConfig, Mode, NodeType, Schema
from bike_router.core.graph_store import EDGE_COLS, NODE_COLS, read_tiles, str_or_none, tile_index, tile_name

logger = logging.getLogger(__name__)


def compute_bbox(nodes_df: pd.DataFrame) -> tuple[float, float, float, float]:
    """The (west, south, east, north) bounds of a node table — one source for both build scripts."""
    return (
        float(nodes_df["lon"].min()),
        float(nodes_df["lat"].min()),
        float(nodes_df["lon"].max()),
        float(nodes_df["lat"].max()),
    )


def write_graph_parquet(
    nodes_df: pd.DataFrame, edges_df: pd.DataFrame, meta: dict[str, Any], out_dir: Path, compression: str
) -> None:
    """Write node/edge tables as lat/lon-tiled Parquet + meta.json under ``out_dir``.

    Nodes are tiled by their own coord; edges by their ``from_node``'s tile, so a corridor
    read (covering tiles + 1 margin) pulls both endpoints.

    Args:
        nodes_df: Node table matching the on-disk schema.
        edges_df: Edge table matching the on-disk schema.
        meta: Metadata dict (must carry ``tile_deg``); written to meta.json.
        out_dir: Output artifact directory.
        compression: Parquet codec — "snappy" (fast, intermediates) or "zstd" (~35% smaller,
            final HF artifact); readers auto-detect the codec.
    """
    assert list(nodes_df.columns) == NODE_COLS, f"nodes schema drift: {list(nodes_df.columns)}"
    assert list(edges_df.columns) == EDGE_COLS, f"edges schema drift: {list(edges_df.columns)}"
    tile_deg = meta["tile_deg"]

    nodes_dir = out_dir / GraphConfig.NODES_SUBDIR
    edges_dir = out_dir / GraphConfig.EDGES_SUBDIR
    nodes_dir.mkdir(parents=True, exist_ok=True)
    edges_dir.mkdir(parents=True, exist_ok=True)

    node_tiles = [
        tile_index(lat=lat, lon=lon, tile_deg=tile_deg)
        for lat, lon in zip(nodes_df["lat"], nodes_df["lon"], strict=True)
    ]
    nodes_df = nodes_df.assign(_tile=node_tiles)
    coord = {
        osmid: (lat, lon) for osmid, lat, lon in zip(nodes_df["osmid"], nodes_df["lat"], nodes_df["lon"], strict=True)
    }
    edge_tiles = [
        tile_index(lat=coord[node][0], lon=coord[node][1], tile_deg=tile_deg) for node in edges_df["from_node"]
    ]
    edges_df = edges_df.assign(_tile=edge_tiles)

    for (row, col), group in nodes_df.groupby("_tile"):
        group[NODE_COLS].to_parquet(
            nodes_dir / f"{tile_name(row=row, col=col)}{GraphConfig.TILE_SUFFIX}", index=False, compression=compression
        )
    for (row, col), group in edges_df.groupby("_tile"):
        group[EDGE_COLS].to_parquet(
            edges_dir / f"{tile_name(row=row, col=col)}{GraphConfig.TILE_SUFFIX}", index=False, compression=compression
        )

    (out_dir / GraphConfig.META_FILENAME).write_text(json.dumps(meta, indent=2))
    logger.info(f"Wrote {len(nodes_df)} nodes / {len(edges_df)} edges across tiles to {out_dir}")


def read_region_tables(region_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one region artifact's full node + edge tables back (all its tiles).

    Used by the Phase-3 combine step to re-load each per-region artifact written by
    write_graph_parquet. Returns (nodes_df, edges_df) with the standard schemas.
    """
    nodes_df = read_tiles(region_dir / GraphConfig.NODES_SUBDIR, columns=NODE_COLS, tiles=None, filters=None)
    edges_df = read_tiles(region_dir / GraphConfig.EDGES_SUBDIR, columns=EDGE_COLS, tiles=None, filters=None)
    return nodes_df, edges_df


def read_full_graph(graph_dir: Path) -> nx.MultiDiGraph:
    """Reconstruct the WHOLE graph from every tile of an artifact (Phase-4 validation).

    Unlike the corridor load (which reads only a corridor's covering tiles), this loads
    all tiles so a connectivity probe tests the entire merged network exactly as saved.
    """
    nodes_df, edges_df = read_region_tables(region_dir=graph_dir)
    return graph_from_tables(nodes_df=nodes_df, edges_df=edges_df)


def graph_from_tables(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.MultiDiGraph:
    """Reconstruct an OSMnx-shaped MultiDiGraph from node/edge tables.

    Bulk add_nodes/edges_from (networkx C internals). Edges referencing a node outside
    the loaded window are dropped (they dangle off the tile set).
    """
    graph = nx.MultiDiGraph(crs=WGS84_CRS)
    graph.add_nodes_from(
        (
            int(n.osmid),
            {
                "x": float(n.lon),
                "y": float(n.lat),
                "elevation": float(n.elevation_m),
                Schema.NODE_TYPE: NodeType(n.node_type),
                Schema.STATION_NAME: str_or_none(value=n.station_name),
            },
        )
        for n in nodes_df.itertuples(index=False)
    )
    present = set(graph.nodes)
    graph.add_edges_from(
        (
            int(e.from_node),
            int(e.to_node),
            int(e.key),
            {
                "length": float(e.length_m),
                "height_diff": float(e.height_diff_m),
                Schema.SURFACE: str_or_none(value=e.surface),
                Schema.HIGHWAY: str_or_none(value=e.highway),
                Schema.MODE: e.mode,
                Schema.GEOMETRY: from_wkt(e.geometry_wkt) if isinstance(e.geometry_wkt, str) else None,
            },
        )
        for e in edges_df.itertuples(index=False)
        if int(e.from_node) in present and int(e.to_node) in present
    )
    _assert_height_diffs_consistent(graph)
    _assert_node_edge_types_consistent(graph)
    return graph


def _assert_node_edge_types_consistent(graph: nx.MultiDiGraph) -> None:
    """Hard-fail if any edge's endpoints don't match its mode's required node types.

    The core guarantee: BIKE joins two bike nodes, RAIL two rail, STATION one of each — so a bike route
    NEVER passes through a station node (it always crosses a station edge carrying boarding).
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


def graph_to_tables(graph: nx.MultiDiGraph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten a built MultiDiGraph into node/edge tables matching the on-disk schema.

    ``node_type`` and ``mode`` are internal invariants the builder sets on every node/edge
    (fail loud if not). Bike/rail edges keep real polylines; station access-links stay straight.
    """
    nodes = [
        {
            Schema.OSMID: int(n),
            Schema.LAT: float(d["y"]),  # OSMnx stores y = latitude
            Schema.LON: float(d["x"]),  # x = longitude
            Schema.ELEVATION_M: float(d["elevation"]),
            Schema.NODE_TYPE: str(d["node_type"]),  # internal invariant: builder types every node
            Schema.STATION_NAME: d["station_name"],  # internal invariant: str for stations, None otherwise
        }
        for n, d in graph.nodes(data=True)
    ]
    edges = [
        {
            Schema.FROM_NODE: int(u),
            Schema.TO_NODE: int(v),
            Schema.KEY: int(k),
            Schema.LENGTH_M: float(d["length"]),
            Schema.HEIGHT_DIFF_M: float(graph.nodes[v]["elevation"] - graph.nodes[u]["elevation"]),
            Schema.SURFACE: _scalar(d.get(Schema.SURFACE)),  # unknown → explicit None (external OSM)
            Schema.HIGHWAY: _scalar(d.get(Schema.HIGHWAY)),  # ditto (genuinely optional)
            Schema.MODE: d["mode"],  # internal invariant: builder tags every edge (fail loud if not)
            Schema.GEOMETRY_WKT: _geometry_wkt(d.get(Schema.GEOMETRY)),
        }
        for u, v, k, d in graph.edges(keys=True, data=True)
    ]
    return pd.DataFrame(nodes, columns=NODE_COLS), pd.DataFrame(edges, columns=EDGE_COLS)


def _geometry_wkt(geom: object) -> str | None:
    """Real edge polyline as WKT, or None for straight (rail/station) hops."""
    if isinstance(geom, LineString) and len(geom.coords) >= 2:
        return str(to_wkt(geom, rounding_precision=GraphConfig.COORD_PRECISION))
    return None


def _scalar(value: object) -> object:
    """Collapse a list-valued OSM tag to its first element; unknown/empty → None.

    Consolidation merges parallel ways into list-valued surface/highway; routing needs one
    representative (surface_tier/road_tier handle scalars). Absent/empty → explicit None.
    """
    if isinstance(value, list | tuple):
        return value[0] if value else None
    return str_or_none(value=value)
