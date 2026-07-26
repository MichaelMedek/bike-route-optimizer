"""Robust, resumable full-DACH bike+rail graph build (designed to run overnight).

Downloads each Geofabrik sub-region .osm.pbf on demand, builds and CHECKPOINTS each
region in isolation (bounded memory), then merges/tiles to the final GeoParquet.
Rerunning skips checkpointed regions, so a crash loses only region N's partial work.

Why sub-regions (Regierungsbezirke etc.) not whole countries: consolidation memory
scales with a region's node count; a regbez peaks ~20 GB, whole-Germany would OOM.
Per-region isolation keeps peak RAM bounded.

Usage:
    # Confirm on a small test region first (one sub-region, ~5 min):
    python scripts/build_dach_graph.py --only karlsruhe-regbez

    # Full DACH overnight (downloads ~5 GB of pbf, runs for hours):
    python scripts/build_dach_graph.py

    # Resume after a crash — just run the same command again.
    # Force a clean rebuild of everything:
    python scripts/build_dach_graph.py --fresh

Regions that fail after retries are logged and skipped so one bad extract cannot
abort the whole run; the final summary lists any skipped regions.
"""

import argparse
import json
import logging
import time
import urllib.request
from pathlib import Path

from bike_router.builder import build_region_graph, merge_region_tables
from bike_router.constants import DEMConfig, GraphConfig
from bike_router.elevation import DEMService
from bike_router.graph_store import (
    compute_bbox,
    graph_to_tables,
    read_region_checkpoint,
    write_graph_parquet,
    write_region_checkpoint,
)

logger = logging.getLogger("build_dach")

_GEOFABRIK = "https://download.geofabrik.de/europe"

# DACH at Geofabrik sub-region granularity. Big Flächenländer are split into their
# Regierungsbezirke (bounded per-region memory); smaller states + AT + CH stay whole.
# key → geofabrik path (without the -latest.osm.pbf suffix).
DACH_REGIONS: dict[str, str] = {
    # Baden-Württemberg
    "freiburg-regbez": "germany/baden-wuerttemberg/freiburg-regbez",
    "karlsruhe-regbez": "germany/baden-wuerttemberg/karlsruhe-regbez",
    "stuttgart-regbez": "germany/baden-wuerttemberg/stuttgart-regbez",
    "tuebingen-regbez": "germany/baden-wuerttemberg/tuebingen-regbez",
    # Bayern
    "mittelfranken": "germany/bayern/mittelfranken",
    "niederbayern": "germany/bayern/niederbayern",
    "oberbayern": "germany/bayern/oberbayern",
    "oberfranken": "germany/bayern/oberfranken",
    "oberpfalz": "germany/bayern/oberpfalz",
    "schwaben": "germany/bayern/schwaben",
    "unterfranken": "germany/bayern/unterfranken",
    # Nordrhein-Westfalen
    "arnsberg-regbez": "germany/nordrhein-westfalen/arnsberg-regbez",
    "detmold-regbez": "germany/nordrhein-westfalen/detmold-regbez",
    "duesseldorf-regbez": "germany/nordrhein-westfalen/duesseldorf-regbez",
    "koeln-regbez": "germany/nordrhein-westfalen/koeln-regbez",
    "muenster-regbez": "germany/nordrhein-westfalen/muenster-regbez",
    # Remaining German states (whole — each smaller than a big Flächenland regbez)
    "berlin": "germany/berlin",
    "brandenburg": "germany/brandenburg",
    "bremen": "germany/bremen",
    "hamburg": "germany/hamburg",
    "hessen": "germany/hessen",
    "mecklenburg-vorpommern": "germany/mecklenburg-vorpommern",
    "niedersachsen": "germany/niedersachsen",
    "rheinland-pfalz": "germany/rheinland-pfalz",
    "saarland": "germany/saarland",
    "sachsen": "germany/sachsen",
    "sachsen-anhalt": "germany/sachsen-anhalt",
    "schleswig-holstein": "germany/schleswig-holstein",
    "thueringen": "germany/thueringen",
    # Austria + Switzerland (sit directly under europe/, same as our base URL).
    "austria": "austria",
    "switzerland": "switzerland",
}


def _download_pbf(region_key: str, geofabrik_path: str, pbf_dir: Path) -> Path:
    """Download a region's .osm.pbf if missing (atomic: temp + rename). Returns path."""
    pbf_dir.mkdir(parents=True, exist_ok=True)
    dest = pbf_dir / f"{region_key}.osm.pbf"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"{_GEOFABRIK}/{geofabrik_path}-latest.osm.pbf"
    tmp = dest.with_suffix(".pbf.tmp")
    logger.info("Downloading %s …", url)
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 — trusted Geofabrik host
    tmp.replace(dest)
    logger.info("  %.0f MB → %s", dest.stat().st_size / 1024 / 1024, dest.name)
    return dest


def _build_one_region(
    *, region_key: str, geofabrik_path: str, pbf_dir: Path, dem: DEMService, tolerance_m: float, retries: int
) -> bool:
    """Download + build + checkpoint one region. Returns True on success.

    Network/parse failures are retried; a region that still fails is reported and
    skipped (returns False) so a single bad extract never aborts the overnight run.
    """
    for attempt in range(1, retries + 1):
        try:
            pbf = _download_pbf(region_key=region_key, geofabrik_path=geofabrik_path, pbf_dir=pbf_dir)
            graph = build_region_graph(pbf_path=pbf, dem=dem, tolerance_m=tolerance_m)
            nodes_df, edges_df = graph_to_tables(graph)
            write_region_checkpoint(
                nodes_df=nodes_df, edges_df=edges_df, ckpt_dir=pbf_dir.parent / "checkpoints", region_key=region_key
            )
            logger.info("✓ %s: %d nodes / %d edges", region_key, len(nodes_df), len(edges_df))
            return True
        except Exception as error:  # noqa: BLE001 — overnight run must survive any single region
            logger.warning("region %s attempt %d/%d failed: %s", region_key, attempt, retries, error)
            time.sleep(5 * attempt)
    logger.error("✗ %s: giving up after %d attempts — SKIPPED", region_key, retries)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Robust, resumable full-DACH bike+rail graph build.")
    parser.add_argument("--out", type=Path, default=GraphConfig.GRAPH_DIR, help="Final artifact dir.")
    parser.add_argument("--work", type=Path, default=GraphConfig.GRAPH_DIR.parent / "dach_build", help="Work dir.")
    parser.add_argument("--dem", type=Path, default=DEMConfig.EURODEM_PATH)
    parser.add_argument("--tolerance", type=float, default=GraphConfig.CONSOLIDATION_TOLERANCE_M)
    parser.add_argument("--only", nargs="+", help="Build only these region keys (test a subset).")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints (clean rebuild).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    logger.setLevel(logging.INFO)

    regions = {k: DACH_REGIONS[k] for k in args.only} if args.only else dict(DACH_REGIONS)
    pbf_dir = args.work / "pbf"
    ckpt_dir = args.work / "checkpoints"
    dem = DEMService(dem_path=args.dem)

    started = time.time()
    built: list[str] = []
    skipped: list[str] = []
    for index, (region_key, geofabrik_path) in enumerate(regions.items(), start=1):
        already = None if args.fresh else read_region_checkpoint(ckpt_dir=ckpt_dir, region_key=region_key)
        if already is not None:
            logger.info("• %s already checkpointed (%d/%d) — skipping build", region_key, index, len(regions))
            built.append(region_key)
            continue
        logger.info("→ %s (%d/%d)", region_key, index, len(regions))
        ok = _build_one_region(
            region_key=region_key,
            geofabrik_path=geofabrik_path,
            pbf_dir=pbf_dir,
            dem=dem,
            tolerance_m=args.tolerance,
            retries=args.retries,
        )
        (built if ok else skipped).append(region_key)

    # Merge every checkpointed region → tile → write final artifact.
    loaded = [(k, read_region_checkpoint(ckpt_dir=ckpt_dir, region_key=k)) for k in built]
    region_tables = [tables for _k, tables in loaded if tables is not None]
    if not region_tables:
        logger.error("No regions built successfully — nothing to write.")
        return 1

    nodes_df, edges_df = merge_region_tables(regions=region_tables)
    bbox = compute_bbox(nodes_df=nodes_df)
    meta = {
        "bbox": list(bbox),
        "tile_deg": GraphConfig.TILE_DEG,
        "tolerance_m": args.tolerance,
        "n_nodes": int(len(nodes_df)),
        "n_edges": int(len(edges_df)),
        "n_stations": int((nodes_df["osmid"] < 0).sum()),
        "regions_built": built,
        "regions_skipped": skipped,
    }
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=meta, out_dir=args.out)

    print(json.dumps(meta, indent=2))
    print(f"\nBuilt {len(built)} regions ({len(skipped)} skipped) in {(time.time() - started) / 60:.1f} min")
    if skipped:
        print(f"SKIPPED regions (rerun to retry): {', '.join(skipped)}")
    print(f"Artifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
