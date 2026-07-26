"""The single preprocessing script: from zero data to the finished tiled graph fixture.

Two phases with their own progress bars — first DOWNLOAD every outstanding region's
Geofabrik .osm.pbf, then BUILD + CHECKPOINT each in isolation (bounded memory) — before
merging/tiling to the final GeoParquet. Rerunning skips checkpointed regions, so a crash
loses only the region it died on. (Publishing to Hugging Face is the separate
upload_graph_to_huggingface.py.)

Why sub-regions (Regierungsbezirke etc.) not whole countries: consolidation memory
scales with a region's node count; a regbez peaks ~20 GB, whole-Germany would OOM.
Per-region isolation keeps peak RAM bounded.

Usage:
    # Confirm on a small clipped region first (~5 min):
    python scripts/build_dach_graph.py --only karlsruhe-regbez --bbox 8.30 48.40 8.80 48.95

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
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

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


def _download_region(*, region_key: str, geofabrik_path: str, pbf_dir: Path, retries: int) -> Path | None:
    """Download one region's pbf with retries. Returns its path, or None if it never succeeds.

    Only genuine network/IO failures are retried; anything else is a real bug and propagates.
    """
    for attempt in range(1, retries + 1):
        try:
            return _download_pbf(region_key=region_key, geofabrik_path=geofabrik_path, pbf_dir=pbf_dir)
        except (urllib.error.URLError, OSError, TimeoutError) as error:  # external: net/disk only
            logger.warning("download %s attempt %d/%d failed: %s", region_key, attempt, retries, error)
            time.sleep(5 * attempt)
    logger.error("✗ %s: download gave up after %d attempts — SKIPPED", region_key, retries)
    return None


def _build_one_region(
    *,
    region_key: str,
    pbf: Path,
    ckpt_dir: Path,
    dem: DEMService,
    tolerance_m: float,
    bbox: tuple[float, float, float, float] | None,
) -> None:
    """Build + checkpoint one already-downloaded region.

    No try/except: a build failure on a valid extract is a real bug (bad data would have
    failed at download), so it propagates and aborts the run rather than silently skipping.
    ``bbox`` clips the region (fast test builds); None builds its full extent.
    """
    graph = build_region_graph(pbf_path=pbf, dem=dem, tolerance_m=tolerance_m, bbox=bbox)
    nodes_df, edges_df = graph_to_tables(graph=graph)
    write_region_checkpoint(nodes_df=nodes_df, edges_df=edges_df, ckpt_dir=ckpt_dir, region_key=region_key)
    logger.info("✓ %s: %d nodes / %d edges", region_key, len(nodes_df), len(edges_df))


def _assert_dem_covers(*, dem: DEMService, area: tuple[float, float, float, float]) -> None:
    """Fail fast unless the DEM's bounds fully contain the area we're about to build.

    ``area`` is (west, south, east, north) in WGS84 degrees. A too-small DEM would bake
    NaN (flat) elevations for out-of-coverage nodes, silently ruining terrain routing —
    so raise BEFORE any pbf download. Loading dem.bounds also surfaces a missing DEM file.

    Args:
        dem: The elevation service (its bounds are the DEM's real coverage).
        area: The (W, S, E, N) region to be processed this run.
    """
    dw, ds, de, dn = dem.bounds
    aw, as_, ae, an = area
    if not (dw <= aw and ds <= as_ and de >= ae and dn >= an):
        raise ValueError(
            f"DEM coverage {dw:.2f},{ds:.2f},{de:.2f},{dn:.2f} does not contain the build area "
            f"{aw:.2f},{as_:.2f},{ae:.2f},{an:.2f}. Re-crop with scripts/crop_dem_to_dach.py."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Robust, resumable full-DACH bike+rail graph build.")
    parser.add_argument("--out", type=Path, default=GraphConfig.GRAPH_DIR, help="Final artifact dir.")
    parser.add_argument("--work", type=Path, default=GraphConfig.GRAPH_DIR.parent / "dach_build", help="Work dir.")
    parser.add_argument("--tolerance", type=float, default=GraphConfig.CONSOLIDATION_TOLERANCE_M)
    parser.add_argument("--only", nargs="+", help="Build only these region keys (test a subset).")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints (clean rebuild).")
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
    logger.setLevel(logging.INFO)

    regions = {k: DACH_REGIONS[k] for k in args.only} if args.only else dict(DACH_REGIONS)
    pbf_dir = args.work / "pbf"
    ckpt_dir = args.work / "checkpoints"
    bbox = tuple(args.bbox) if args.bbox else None
    # Fail fast: load the DEM up front (the build bakes elevation from it) and confirm it
    # covers the area we're about to process — so a missing/too-small DEM raises BEFORE any
    # pbf is downloaded, not hours in. Crop the DEM with crop_dem_to_dach.py.
    dem = DEMService(dem_path=DEMConfig.EURODEM_PATH)
    _assert_dem_covers(dem=dem, area=bbox or GraphConfig.DACH_BBOX_DEG)
    logger.info("DEM ready: coverage W,S,E,N = %.2f, %.2f, %.2f, %.2f", *dem.bounds)

    started = time.time()
    # Regions already checkpointed need neither download nor build; keep them for the merge.
    done = {
        k for k in regions if not args.fresh and read_region_checkpoint(ckpt_dir=ckpt_dir, region_key=k) is not None
    }
    todo = {k: regions[k] for k in regions if k not in done}
    built: list[str] = list(done)  # sorted just before the merge (order-independent)
    skipped: list[str] = []

    # Phase 1 — download every outstanding region's raw pbf up front (one bar).
    pbfs: dict[str, Path] = {}
    for region_key, geofabrik_path in tqdm(todo.items(), desc="Downloading pbfs", unit="region"):
        pbf = _download_region(
            region_key=region_key, geofabrik_path=geofabrik_path, pbf_dir=pbf_dir, retries=args.retries
        )
        if pbf is None:
            skipped.append(region_key)
        else:
            pbfs[region_key] = pbf

    # Phase 2 — build + checkpoint each downloaded region to a graph (second bar). A build
    # failure propagates (real bug); only download failures are skipped (external, in phase 1).
    for region_key, pbf in tqdm(pbfs.items(), desc="Building regions", unit="region"):
        _build_one_region(
            region_key=region_key,
            pbf=pbf,
            ckpt_dir=ckpt_dir,
            dem=dem,
            tolerance_m=args.tolerance,
            bbox=bbox,
        )
        built.append(region_key)

    # Merge every checkpointed region → tile → write final artifact. Sorted for a
    # deterministic station-id block assignment regardless of build/resume order. A region
    # in `built` was just checkpointed, so a missing read is a real bug (fail loud).
    if not built:
        logger.error("No regions built successfully — nothing to write.")
        return 1
    built = sorted(built)
    region_tables = []
    for region_key in built:
        tables = read_region_checkpoint(ckpt_dir=ckpt_dir, region_key=region_key)
        assert tables is not None, f"checkpoint for built region {region_key!r} is missing/corrupt"
        region_tables.append(tables)

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
