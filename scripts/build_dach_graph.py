"""The single preprocessing script: from zero data to the finished tiled graph.

ONE fixed workflow: download every region's Geofabrik .osm.pbf, build each
to a bike+rail graph with baked elevation, merge, and write the lat/lon-tiled GeoParquet to
GraphConfig.GRAPH_DIR. (Publishing to Hugging Face is the separate upload_graph_to_huggingface.py.)

The output dir must be EMPTY (or absent): a single leftover file — e.g. from a `--only`
smoke run — fails fast, so a partial/clipped artifact can never be mistaken for a full one.

Everything except WHICH regions (--only) and an optional test clip (--bbox) is a fixed.

Usage:
    # Confirm on a small clipped region first (~5 min) — writes to the same fixed output dir,
    # so DELETE that dir before the real run:
    python scripts/build_dach_graph.py --only karlsruhe-regbez --bbox 8.30 48.40 8.80 48.95

    # Full DACH (downloads ~5 GB of pbf, runs for hours). Output dir must be empty first.
    python scripts/build_dach_graph.py

Why sub-regions (Regierungsbezirke etc.) not whole countries: consolidation memory
scales with a region's node count; a regbez peaks ~20 GB, whole-Germany would OOM.
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
from bike_router.graph_store import compute_bbox, graph_to_tables, write_graph_parquet

logger = logging.getLogger("build_dach")

_GEOFABRIK = "https://download.geofabrik.de/europe"
# Raw pbf downloads are cached beside the artifact (re-fetchable input, not build state).
_PBF_DIR = GraphConfig.GRAPH_DIR.parent / "dach_build" / "pbf"
_DOWNLOAD_RETRIES = 5  # transient network blips only; a final failure aborts the whole run

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


def _download_pbf(*, region_key: str, geofabrik_path: str) -> Path:
    """Download a region's .osm.pbf if missing (atomic: temp + rename). Returns its path.

    A failed transfer unlinks its partial .tmp and re-raises so no orphan/partial lingers.
    """
    _PBF_DIR.mkdir(parents=True, exist_ok=True)
    dest = _PBF_DIR / f"{region_key}.osm.pbf"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"{_GEOFABRIK}/{geofabrik_path}-latest.osm.pbf"
    tmp = dest.with_suffix(".pbf.tmp")
    logger.info("Downloading %s …", url)
    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 — trusted Geofabrik host
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)  # never leave a partial behind
        raise
    logger.info("  %.0f MB → %s", dest.stat().st_size / 1024 / 1024, dest.name)
    return dest


def _download_region(*, region_key: str, geofabrik_path: str) -> Path:
    """Download one region's pbf, retrying only transient network failures.

    A retry-exhausted download raises (aborting the whole run) — a missing region must
    never yield a silently-partial artifact. Non-network OSErrors are real bugs: they
    propagate immediately, uncaught.
    """
    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        try:
            return _download_pbf(region_key=region_key, geofabrik_path=geofabrik_path)
        except (urllib.error.URLError, TimeoutError) as error:  # transient network only
            logger.warning("download %s attempt %d/%d failed: %s", region_key, attempt, _DOWNLOAD_RETRIES, error)
            if attempt == _DOWNLOAD_RETRIES:
                raise
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")  # loop either returns or raises


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


def _assert_output_empty(*, out_dir: Path) -> None:
    """Fail fast unless the output dir is absent or empty — no partial/stale artifact reuse.

    A single leftover file (e.g. from an earlier `--only` smoke run) aborts the run, so a
    clipped or half-written graph can never be mistaken for a complete one. Delete the dir
    to rebuild.
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(
            f"Output dir {out_dir} is not empty — delete it to rebuild (this build runs fully or not at all)."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the full DACH bike+rail graph (runs fully or not at all).")
    parser.add_argument("--only", nargs="+", help="Build only these region keys (test a subset).")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("W", "S", "E", "N"),
        default=None,
        help="Optional lon/lat clip (west south east north) to build a small test region fast.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    regions = {k: DACH_REGIONS[k] for k in args.only} if args.only else dict(DACH_REGIONS)
    bbox = tuple(args.bbox) if args.bbox else None
    out_dir = GraphConfig.GRAPH_DIR
    tolerance_m = GraphConfig.CONSOLIDATION_TOLERANCE_M

    # Fail fast, before any download: the output dir must be empty, and the DEM must cover
    # the area we're about to build (a too-small DEM would bake flat elevations everywhere).
    _assert_output_empty(out_dir=out_dir)
    dem = DEMService(dem_path=DEMConfig.EURODEM_PATH)
    _assert_dem_covers(dem=dem, area=bbox or GraphConfig.DACH_BBOX_DEG)
    logger.info("DEM ready: coverage W,S,E,N = %.2f, %.2f, %.2f, %.2f", *dem.bounds)

    started = time.time()

    # Phase 1 — download every region's raw pbf up front (one bar). Any failure aborts.
    pbfs = {
        region_key: _download_region(region_key=region_key, geofabrik_path=geofabrik_path)
        for region_key, geofabrik_path in tqdm(regions.items(), desc="Downloading pbfs", unit="region")
    }

    # Phase 2 — build each region to its (nodes, edges) tables (second bar). Any failure
    # aborts (a build error on a valid extract is a real bug, not something to skip).
    region_tables = [
        graph_to_tables(graph=build_region_graph(pbf_path=pbf, dem=dem, tolerance_m=tolerance_m, bbox=bbox))
        for pbf in tqdm(pbfs.values(), desc="Building regions", unit="region")
    ]

    # Merge → tile → write the final artifact. Sorted regions give a deterministic
    # station-id block assignment (merge offsets by list index).
    nodes_df, edges_df = merge_region_tables(regions=region_tables)
    bbox_out = compute_bbox(nodes_df=nodes_df)
    built = sorted(regions)
    meta = {
        "bbox": list(bbox_out),
        "tile_deg": GraphConfig.TILE_DEG,
        "tolerance_m": tolerance_m,
        "n_nodes": int(len(nodes_df)),
        "n_edges": int(len(edges_df)),
        "n_stations": int((nodes_df["osmid"] < 0).sum()),
        "regions_built": built,
    }
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=meta, out_dir=out_dir)

    print(json.dumps(meta, indent=2))
    print(f"\nBuilt {len(built)} regions in {(time.time() - started) / 60:.1f} min")
    print(f"Artifact: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
