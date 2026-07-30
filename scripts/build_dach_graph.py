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
scales with a region's node count; a regbez peaks ~20-32 GB, whole-Germany would OOM.
Austria and Switzerland have no Geofabrik sub-extracts, so they are bbox-split east/west
(one shared pbf, ~0.5° overlapping halves that Phase-3 dedup stitches) — see DACH_REGIONS.
"""

import argparse
import gc
import json
import logging
import multiprocessing
import resource
import socket
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")  # headless — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from shapely import from_wkt, get_coordinates  # noqa: E402

from bike_router.core.constants import DEMConfig, GraphConfig, Mode, Palette
from bike_router.preprocessing.builder import (
    build_region_graph_clipped,
    remap_contiguous,
    stage_pbf,
)
from bike_router.preprocessing.elevation import DEMService
from bike_router.preprocessing.graph_writer import (
    graph_to_tables,
    write_graph_parquet,
)
from bike_router.preprocessing.regions import (
    DACH_REGIONS,
    Bbox,
    Region,
    assert_all_regions_complete,
    base_meta,
    combine_regions,
    region_complete,
    split_geofabrik_path,
)

logger = logging.getLogger("build_dach")

# One log format for parent AND spawned workers (spawn children don't inherit the parent's config).
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


def _configure_logging() -> None:
    """Set up INFO logging — called in main() AND as each worker's initializer (spawn needs both).

    osmnx's own console logging is left OFF: it emits a line per internal call (hundreds of "Created
    nodes GeoDataFrame from graph"), drowning our own [n/6] stage breadcrumbs. Our logs are enough.
    """
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_LOG_DATEFMT)


_GEOFABRIK = "https://download.geofabrik.de/europe"
# Raw pbf downloads + per-region artifacts are cached under the build dir (re-fetchable input
# / resumable state), separate from the final GRAPH_DIR artifact.
_BUILD_DIR = GraphConfig.GRAPH_DIR.parent / "dach_build"
_PBF_DIR = _BUILD_DIR / "pbf"
_REGIONS_DIR = _BUILD_DIR / "dach_graph_per_region"
_DOWNLOAD_RETRIES = 10  # transient network blips only; a final failure aborts the whole run
# Per-socket-read timeout (s): a stalled transfer (0 bytes flowing) raises after this, turning
# a silent hang into a retryable failure. Geofabrik redirects can pick a dead path — retry escapes it.
_SOCKET_TIMEOUT_S = 60.0
# Per-region staged-pbf ceiling: a region whose osmium clip exceeds this parses into too big a graph
# for one worker's RAM (niedersachsen at 478 MB is the proven-OK high-water mark). Preflight fails
# BEFORE the build loop so an oversized split is caught in seconds, not mid-build.
_MAX_STAGED_PBF_MB = 500.0


def _pbf_dest(*, geofabrik_path: str) -> Path:
    """Local cache path for a region's pbf (keyed by Geofabrik leaf name)."""
    return _PBF_DIR / f"{split_geofabrik_path(geofabrik_path=geofabrik_path)}.osm.pbf"


def _pbf_on_disk(*, geofabrik_path: str) -> bool:
    """True if the region's pbf is already fully downloaded."""
    dest = _pbf_dest(geofabrik_path=geofabrik_path)
    return dest.exists() and dest.stat().st_size > 0


def _download_pbf(*, geofabrik_path: str) -> Path:
    """Download a region's .osm.pbf if missing (atomic: temp + rename). Returns its path.

    Cached by the Geofabrik leaf name, so bbox-split halves sharing one pbf download it once.
    A failed transfer unlinks its partial .tmp and re-raises so no orphan/partial lingers.
    """
    _PBF_DIR.mkdir(parents=True, exist_ok=True)
    dest = _pbf_dest(geofabrik_path=geofabrik_path)
    if _pbf_on_disk(geofabrik_path=geofabrik_path):
        return dest
    url = f"{_GEOFABRIK}/{geofabrik_path}-latest.osm.pbf"
    tmp = dest.with_suffix(".pbf.tmp")
    logger.info(f"Downloading {url} …")
    try:
        # urlretrieve honours the global default socket timeout; a stalled read then raises
        # socket.timeout (a TimeoutError subclass) instead of hanging forever.
        socket.setdefaulttimeout(_SOCKET_TIMEOUT_S)
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 — trusted Geofabrik host
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)  # never leave a partial behind
        raise
    finally:
        socket.setdefaulttimeout(None)  # restore (don't leak the timeout to other sockets)
    logger.info(f"  {dest.stat().st_size / 1024 / 1024:.0f} MB → {dest.name}")
    return dest


def _download_region(*, geofabrik_path: str) -> Path:
    """Download one region's pbf, retrying only transient network failures.

    A retry-exhausted download raises (aborting the whole run) — a missing region must
    never yield a silently-partial artifact. Non-network OSErrors are real bugs: they
    propagate immediately, uncaught.
    """
    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        try:
            return _download_pbf(geofabrik_path=geofabrik_path)
        except (urllib.error.URLError, TimeoutError) as error:  # transient network only
            logger.warning(f"download {geofabrik_path} attempt {attempt}/{_DOWNLOAD_RETRIES} failed: {error}")
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


def _peak_rss_gb() -> float:
    """This process's peak resident memory in GB (ru_maxrss is BYTES on macOS)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def _assert_staged_sizes_ok(*, regions: list[Region], pbfs: dict[str, Path], cli_bbox: Bbox | None) -> None:
    """Preflight: osmium-clip EVERY region to a temp file and fail loud if any exceeds the per-region
    MB ceiling — BEFORE the build loop, so an oversized split is caught in ~seconds (native C++ clip).
    ALL regions are checked (not just to-build ones): the ceiling is a structural invariant of the
    split config. niedersachsen (478 MB) is the proven-OK high-water mark.
    """
    oversized: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for region in tqdm(regions, desc="2/4 Preflight: staged sizes", unit="region"):
            staged = stage_pbf(raw_pbf=pbfs[region.geofabrik_path], bbox=cli_bbox or region.bbox, staging_dir=Path(tmp))
            size_mb = staged.stat().st_size / 1024 / 1024
            if size_mb > _MAX_STAGED_PBF_MB:
                oversized.append(f"{region.key} ({size_mb:.0f} MB)")
            staged.unlink()  # free the temp clip before staging the next (peak temp ≈ one clip)
    if oversized:
        raise ValueError(
            f"Staged pbf exceeds {_MAX_STAGED_PBF_MB:.0f} MB ceiling for: {', '.join(oversized)}. "
            "Split the region into smaller bbox pieces in DACH_REGIONS."
        )


def _process_region(
    *,
    region_key: str,
    pbf: Path,
    bbox: tuple[float, float, float, float] | None,
    tolerance_m: float,
) -> float:
    """Phase 2 worker: build ONE region, remap ids contiguous, write its artifact; return peak RSS (GB).

    Runs in a FRESH child (max_tasks_per_child=1) so the OS reclaims ALL its memory on exit — RSS
    never accumulates across regions. The stage(osmium-clip)+build workflow lives in builder; only
    picklable args cross the process boundary.
    """
    dem = DEMService(dem_path=DEMConfig.EURODEM_PATH)
    graph = build_region_graph_clipped(raw_pbf=pbf, dem=dem, tolerance_m=tolerance_m, bbox=bbox)
    nodes_df, edges_df = graph_to_tables(graph=graph)
    nodes_df, edges_df = remap_contiguous(nodes_df=nodes_df, edges_df=edges_df)
    region_dir = _REGIONS_DIR / region_key
    meta = {
        **base_meta(nodes_df=nodes_df, edges_df=edges_df, tolerance_m=tolerance_m),
        "confirmed_complete": True,  # written LAST via write_graph_parquet → the atomic "done" flag
    }
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=meta, out_dir=region_dir)
    return _peak_rss_gb()  # this child's peak; the process then exits and the OS reclaims everything


def _plot_overview(*, edges_df: pd.DataFrame, out_path: Path) -> None:
    """Save a minimalist matplotlib overview of the final graph: bike edges thin blue, rail thick purple.

    FULLY VECTORIZED — no Python loop over the 10M+ edges: ``from_wkt`` parses the whole WKT column in
    one C call, ``get_coordinates`` extracts every vertex at once (this is what geopandas does), and
    ``np.split`` on the geometry-index change points slices per-edge segments. One LineCollection per
    mode draws them. Station links (short bike↔rail hops) are omitted.
    """
    fig, ax = plt.subplots(figsize=(16, 18))
    # Draw bike first (thin blue), rail on top (thicker purple) so rail reads clearly over the mesh.
    for mode, color, width in ((Mode.BIKE, Palette.START, 0.15), (Mode.RAIL, Palette.RAIL, 0.9)):
        wkts = edges_df.loc[edges_df["mode"] == mode, "geometry_wkt"].dropna()
        geoms = from_wkt(np.asarray(wkts, dtype=object))  # vectorized: whole column in one call
        coords, index = get_coordinates(geoms, return_index=True)  # all vertices at once, (N,2) + source id
        segments = np.split(coords, np.flatnonzero(np.diff(index)) + 1)  # slice per-edge, pure numpy
        # rasterized=True: flatten the millions of segments to a pixel layer in the file (small size, fast).
        ax.add_collection(LineCollection(segments, colors=color, linewidths=width, rasterized=True))
        logger.info(f"  plotted {mode}: {len(segments)} edges ({len(coords)} vertices)")
    ax.autoscale_view()  # LineCollection does not autoscale the axes; do it explicitly
    ax.set_aspect(1.4)  # rough lat/lon aspect at ~50°N
    ax.set_title("DACH graph overview — bike (thin blue) · rail (thick purple)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    logger.info(f"Overview plot written to {out_path}")


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

    _configure_logging()

    regions = [r for r in DACH_REGIONS if r.key in set(args.only)] if args.only else list(DACH_REGIONS)
    cli_bbox = tuple(args.bbox) if args.bbox else None  # global test clip; overrides each region's own bbox
    out_dir = GraphConfig.GRAPH_DIR
    tolerance_m = GraphConfig.CONSOLIDATION_TOLERANCE_M

    # Fail fast, before any download: the output dir must be empty, and the DEM must cover
    # the area we're about to build (a too-small DEM would bake flat elevations everywhere).
    _assert_output_empty(out_dir=out_dir)
    dem = DEMService(dem_path=DEMConfig.EURODEM_PATH)
    _assert_dem_covers(dem=dem, area=cli_bbox or GraphConfig.DACH_BBOX_DEG)
    w, s, e, n = dem.bounds
    logger.info(f"DEM ready: coverage W,S,E,N = {w:.2f}, {s:.2f}, {e:.2f}, {n:.2f}")

    started = time.time()

    # Phase 1 — DOWNLOAD every distinct pbf up front (skip-if-present). Split halves share one pbf,
    # so dedup by geofabrik_path. Any failure aborts.
    pbfs: dict[str, Path] = {}
    for gp in tqdm(sorted({r.geofabrik_path for r in regions}), desc="1/4 Downloading pbfs", unit="pbf"):
        if _pbf_on_disk(geofabrik_path=gp):
            logger.info(f"{gp}: skipped, already on disk")
        pbfs[gp] = _download_region(geofabrik_path=gp)

    # Phase 2 preflight — osmium-clip every to-build region to a temp file and HARD-FAIL if any
    # exceeds the MB ceiling, BEFORE the slow build loop (an oversized split is caught in seconds).
    _assert_staged_sizes_ok(regions=regions, pbfs=pbfs, cli_bbox=cli_bbox)

    # Phase 2 — build each region in a FRESH child (max_tasks_per_child=1) so the OS reclaims all its
    # memory on exit; peak RSS never accumulates (gc can't return RSS to the OS). spawn = macOS default,
    # fork-safe. Skip confirmed_complete regions; a killed worker (OOM) fails loud.
    ctx = multiprocessing.get_context("spawn")
    max_child_rss = 0.0  # largest single-region child peak — the real memory watermark of the build
    with ProcessPoolExecutor(
        max_workers=1, max_tasks_per_child=1, mp_context=ctx, initializer=_configure_logging
    ) as pool:
        for region in tqdm(regions, desc="2/4 Building regions", unit="region"):
            if region_complete(regions_dir=_REGIONS_DIR, region_key=region.key):
                logger.info(f"{region.key}: skipped, already complete")
                continue
            future = pool.submit(
                _process_region,
                region_key=region.key,
                pbf=pbfs[region.geofabrik_path],
                bbox=cli_bbox or region.bbox,
                tolerance_m=tolerance_m,
            )
            # .result() re-raises any worker failure here (BrokenProcessPool if OOM-killed) → fail loud.
            peak = future.result()
            max_child_rss = max(max_child_rss, peak)
            logger.info(f"{region.key}: done, child peak RSS {peak:.1f} GB")

    # Phase 3 — COMBINE per-region artifacts into globally-consistent tiled shards (zstd — the final
    # artifact uploaded to HF, ~35% smaller than snappy; readers auto-detect the codec).
    built = sorted(r.key for r in regions)
    assert_all_regions_complete(regions_dir=_REGIONS_DIR, regions=built)
    nodes_df, edges_df = combine_regions(regions_dir=_REGIONS_DIR, regions=built)
    meta = {
        **base_meta(nodes_df=nodes_df, edges_df=edges_df, tolerance_m=tolerance_m),
        "regions_built": built,
    }
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=meta, out_dir=out_dir, compression="zstd")
    del nodes_df
    gc.collect()

    # Overview plot (Phase-3 prune already enforced connectivity by construction — no separate validation).
    # Saved INTO the artifact dir so the HF upload (upload_folder of GRAPH_DIR) ships it as-is.
    _plot_overview(edges_df=edges_df, out_path=out_dir / "dach_graph_overview.png")

    print(json.dumps(meta, indent=2))
    print(
        f"\nBuilt {len(built)} regions in {(time.time() - started) / 60:.1f} min | "
        f"max per-region child peak RSS {max_child_rss:.1f} GB"
    )
    print(f"Artifact: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
