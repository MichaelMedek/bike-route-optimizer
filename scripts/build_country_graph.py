"""Build the prebuilt bike+rail graph from Geofabrik .osm.pbf extracts.

Runs offline (may take hours for full DACH). Reads each region pbf, builds a
consolidated bike+rail graph with baked node elevation, then writes lat/lon-tiled
GeoParquet + meta.json under GraphConfig.GRAPH_DIR (or --out).

Usage:
    # Schwarzwald test region
    python scripts/build_country_graph.py data/pbf/karlsruhe-regbez.osm.pbf

    # Full DACH (download the three country extracts from Geofabrik first)
    python scripts/build_country_graph.py \
        data/pbf/germany.osm.pbf data/pbf/austria.osm.pbf data/pbf/switzerland.osm.pbf

The DEM is auto-downloaded from Hugging Face on first run (must cover the pbf area).
"""

import argparse
import logging
import time
from pathlib import Path

from tqdm import tqdm

from bike_router.builder import build_region_graph, merge_region_tables
from bike_router.constants import DEMConfig, GraphConfig
from bike_router.elevation import DEMService
from bike_router.graph_store import compute_bbox, graph_to_tables, write_graph_parquet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the prebuilt bike+rail graph from .osm.pbf extracts.")
    parser.add_argument("pbf", nargs="+", type=Path, help="One or more Geofabrik .osm.pbf extracts.")
    parser.add_argument("--dem", type=Path, default=DEMConfig.EURODEM_PATH, help="Local DEM GeoTIFF.")
    parser.add_argument("--out", type=Path, default=GraphConfig.GRAPH_DIR, help="Output artifact dir.")
    parser.add_argument(
        "--tolerance", type=float, default=GraphConfig.CONSOLIDATION_TOLERANCE_M, help="Consolidation radius (m)."
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("W", "S", "E", "N"),
        default=None,
        help="Optional lon/lat clip (west south east north) to build a small test region fast.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    for pbf in args.pbf:
        if not pbf.exists():
            raise FileNotFoundError(f"pbf not found: {pbf}")

    dem_path = args.dem  # local build-time DEM (no download)
    dem = DEMService(dem_path=dem_path)

    bbox = tuple(args.bbox) if args.bbox else None
    start = time.time()
    # Per-region build with a visible tqdm bar (one region at a time → bounded memory).
    regions = [
        graph_to_tables(build_region_graph(pbf_path=pbf, dem=dem, tolerance_m=args.tolerance, bbox=bbox))
        for pbf in tqdm(args.pbf, desc="Building regions", unit="region")
    ]
    nodes_df, edges_df = merge_region_tables(regions=regions)
    n_stations = int((nodes_df["osmid"] < 0).sum())
    bbox = compute_bbox(nodes_df=nodes_df)
    meta = {
        "bbox": list(bbox),
        "tile_deg": GraphConfig.TILE_DEG,
        "tolerance_m": args.tolerance,
        "n_nodes": int(len(nodes_df)),
        "n_edges": int(len(edges_df)),
        "n_stations": n_stations,
        "sources": [p.name for p in args.pbf],
    }
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=meta, out_dir=args.out)

    print(
        f"\nBuilt {len(nodes_df):,} nodes / {len(edges_df):,} edges "
        f"({n_stations:,} stations) in {time.time() - start:.0f}s"
    )
    print(f"bbox W,S,E,N = {bbox}")
    print(f"Artifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
