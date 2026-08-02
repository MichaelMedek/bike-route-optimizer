"""Post-processing pass: flag bike edges whose baked terrain deviates far from the node-to-node line.

Run ONCE after the standard build pipeline. Iterates the edge tiles ONE AT A TIME (memory-safe — each
tile is independent, released before the next), computes the SAME deviation the app uses
(core.track.edge_elevation_deviation_m), and writes a sister tile under edge_unreliable_elevation/ with
just (from, to, key, elevation_deviation_m) for edges past the warn threshold — ONLY if any exist. The
router joins it at load and charges the excess as extra climb; a missing sister file means no penalty.

Run:  python scripts/flag_unreliable_elevation.py [--graph-dir DIR]
"""

import argparse
from pathlib import Path

import pandas as pd

from bike_router.core.constants import GradeConfig, GraphConfig, Mode, NodeType, Schema
from bike_router.core.graph_store import NODE_COLS, _oriented_geometry, read_tiles
from bike_router.core.route_path import RouteEdge, RouteNode
from bike_router.core.track import edge_elevation_deviation_m

_WARN = GradeConfig.ELEVATION_DEVIATION_WARN_M
# Columns the deviation needs: the (from, to, key) join triple + mode/length/geometry for the formula.
_EDGE_COLS = [Schema.FROM_NODE, Schema.TO_NODE, Schema.KEY, Schema.LENGTH_M, Schema.MODE, Schema.GEOMETRY_WKT]


def _row_deviation(*, row: pd.Series, elev_by_osmid: dict[int, float]) -> float:
    """Production deviation for one bike-edge row — reuses the core formula via a minimal RouteEdge.

    Endpoints are looked up strictly (main() loads ALL nodes, so a miss is real corruption); the caller
    has already filtered to rows with a baked polyline, so the geometry is present.
    """
    from_node, to_node = int(row[Schema.FROM_NODE]), int(row[Schema.TO_NODE])
    node_a = RouteNode(
        osmid=from_node,
        lat=0.0,
        lon=0.0,
        elevation_m=elev_by_osmid[from_node],
        node_type=NodeType.BIKE,
        station_name=None,
    )
    node_b = RouteNode(
        osmid=to_node, lat=0.0, lon=0.0, elevation_m=elev_by_osmid[to_node], node_type=NodeType.BIKE, station_name=None
    )
    # Deviation is orientation-invariant, so a zero-coord node_a only steers which polyline end we start
    # from; the real endpoint elevations (what the formula reads) are passed correctly.
    coords, zs = _oriented_geometry(wkt=row[Schema.GEOMETRY_WKT], node_a=node_a)
    edge = RouteEdge(
        from_node=from_node,
        to_node=to_node,
        mode=str(row[Schema.MODE]),
        length_m=float(row[Schema.LENGTH_M]),
        surface=None,
        highway=None,
        geometry=coords,
        geometry_z=zs,
    )
    return edge_elevation_deviation_m(node_a=node_a, node_b=node_b, edge=edge)


def _flag_tile(*, edge_path: Path, out_dir: Path, elev_by_osmid: dict[int, float]) -> int:
    """Scan one edge tile; write its sister tile with the offenders' deviation, only if any exist.

    Loads/releases this tile alone (memory-safe). Returns the offender count for the run summary.
    """
    edges = pd.read_parquet(edge_path, columns=_EDGE_COLS)
    bike = edges[(edges[Schema.MODE] == Mode.BIKE) & edges[Schema.GEOMETRY_WKT].apply(lambda w: isinstance(w, str))]
    if bike.empty:
        return 0
    devs = bike.apply(lambda row: _row_deviation(row=row, elev_by_osmid=elev_by_osmid), axis=1)
    over = devs > _WARN
    if not over.any():
        return 0
    sister = bike.loc[over, [Schema.FROM_NODE, Schema.TO_NODE, Schema.KEY]].copy()
    sister[Schema.ELEVATION_DEVIATION_M] = devs[over].to_numpy(dtype=float)
    out_dir.mkdir(parents=True, exist_ok=True)
    sister.to_parquet(out_dir / edge_path.name, compression="zstd", index=False)
    return int(over.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Flag bike edges with unreliable baked elevation, per tile.")
    parser.add_argument("--graph-dir", type=Path, default=GraphConfig.GRAPH_DIR, help="Graph artifact dir to scan.")
    args = parser.parse_args()
    graph_dir = args.graph_dir
    out_dir = graph_dir / GraphConfig.UNRELIABLE_ELEVATION_SUBDIR
    # Node elevations are small (osmid→elev) and edges reference neighbour-tile nodes, so load them once
    # globally; the big per-tile edge tables are what we stream one at a time.
    nodes = read_tiles(directory=graph_dir / GraphConfig.NODES_SUBDIR, columns=NODE_COLS, tiles=None, filters=None)
    elev_by_osmid = {int(o): float(e) for o, e in zip(nodes[Schema.OSMID], nodes[Schema.ELEVATION_M], strict=True)}
    del nodes

    edge_paths = sorted((graph_dir / GraphConfig.EDGES_SUBDIR).glob(f"tile_*{GraphConfig.TILE_SUFFIX}"))
    total_offenders = 0
    for i, edge_path in enumerate(edge_paths, start=1):
        n = _flag_tile(edge_path=edge_path, out_dir=out_dir, elev_by_osmid=elev_by_osmid)
        total_offenders += n
        print(f"[{i}/{len(edge_paths)}] {edge_path.name}: {n} unreliable edges")
    print(f"\nDone. {total_offenders} unreliable bike edges across {len(edge_paths)} tiles → {out_dir}")


if __name__ == "__main__":
    main()
