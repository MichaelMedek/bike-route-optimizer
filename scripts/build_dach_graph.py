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
import gc
import json
import logging
import random
import resource
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm

from bike_router.builder import build_region_graph, dedup_by_geometry, reindex_region, remap_contiguous
from bike_router.constants import DEMConfig, GraphConfig, NodeType
from bike_router.elevation import DEMService
from bike_router.graph_store import (
    compute_bbox,
    graph_to_tables,
    read_full_graph,
    read_region_tables,
    tile_index,
    write_graph_parquet,
)

logger = logging.getLogger("build_dach")

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
_VALIDATION_PROBES = 10  # Phase 4: random cross-region node pairs checked for connectivity
# Sanity ceiling on the merged node count (DACH is ~3–5M); a larger total means a logic error
# upstream, so Phase 3 fails fast rather than writing a suspect artifact.
_MAX_TOTAL_NODES = 100_000_000_000
_VALIDATION_PROBES = 10  # Phase 4: random cross-region node pairs checked for connectivity

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


def _peak_rss_gb() -> float:
    """This process's peak resident memory in GB (ru_maxrss is BYTES on macOS)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def _region_complete(region_key: str) -> bool:
    """True if a region's per-region artifact exists and is flagged confirmed_complete."""
    meta_path = _REGIONS_DIR / region_key / GraphConfig.META_FILENAME
    if not meta_path.exists():
        return False
    return bool(json.loads(meta_path.read_text()).get("confirmed_complete", False))


def _process_region(
    *,
    region_key: str,
    pbf: Path,
    dem: DEMService,
    bbox: tuple[float, float, float, float] | None,
    tolerance_m: float,
) -> None:
    """Phase 2: build ONE region, remap ids contiguous, write its app-loadable artifact.

    Writes to _REGIONS_DIR/<region>/ (standard nodes/+edges/+meta.json), stamping
    confirmed_complete=true LAST so a crash never leaves a half-region mistaken for done.
    Frees the ~20 GB graph before returning so the next region reuses the heap.
    """
    graph = build_region_graph(pbf_path=pbf, dem=dem, tolerance_m=tolerance_m, bbox=bbox)
    nodes_df, edges_df = graph_to_tables(graph=graph)
    nodes_df, edges_df = remap_contiguous(nodes_df=nodes_df, edges_df=edges_df)
    region_dir = _REGIONS_DIR / region_key
    meta = {
        "bbox": list(compute_bbox(nodes_df=nodes_df)),
        "tile_deg": GraphConfig.TILE_DEG,
        "tolerance_m": tolerance_m,
        "n_nodes": int(len(nodes_df)),
        "n_edges": int(len(edges_df)),
        "n_stations": int((nodes_df["node_type"] == NodeType.RAIL).sum()),
        "confirmed_complete": True,  # written LAST via write_graph_parquet → the atomic "done" flag
    }
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=meta, out_dir=region_dir)
    del graph, nodes_df, edges_df
    gc.collect()


def _assert_all_regions_complete(*, regions: list[str]) -> None:
    """Fail loud unless EVERY region has a confirmed_complete per-region artifact (Phase 3 gate)."""
    missing = [r for r in regions if not _region_complete(region_key=r)]
    if missing:
        raise ValueError(
            f"Cannot combine: {len(missing)} region(s) not confirmed_complete: {', '.join(missing)}. "
            "Re-run to (re)build them."
        )


def _combine_regions(*, regions: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase 3: offset each region's contiguous ids into a global space, dedup the seam, re-densify.

    A running offset (ΣN of earlier regions) makes ids collision-free; geometry dedup collapses
    border duplicates; a final remap closes the holes (n_nodes == max_id + 1). Returns (nodes, edges).
    """
    node_frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    offset = 0
    for region_key in tqdm(regions, desc="3/4 Combining regions", unit="region"):
        nodes_df, edges_df = read_region_tables(region_dir=_REGIONS_DIR / region_key)
        nodes_df, edges_df = reindex_region(nodes_df=nodes_df, edges_df=edges_df, offset=offset)
        offset += len(nodes_df)
        node_frames.append(nodes_df)
        edge_frames.append(edges_df)
    if offset >= _MAX_TOTAL_NODES:
        raise ValueError(f"Combined node count {offset} exceeds sanity ceiling {_MAX_TOTAL_NODES} — aborting.")
    nodes_df = pd.concat(node_frames, ignore_index=True)
    edges_df = pd.concat(edge_frames, ignore_index=True)
    nodes_df, edges_df = dedup_by_geometry(nodes_df=nodes_df, edges_df=edges_df)
    # Dedup removed border duplicates, leaving id holes → renumber to dense 0..N-1 (n_nodes==max_id+1).
    return remap_contiguous(nodes_df=nodes_df, edges_df=edges_df)


def _validate_connectivity(*, out_dir: Path) -> None:
    """Phase 4: load the saved production graph and assert it is ONE connected network.

    Exhaustively asserts a single strongly-connected component (the routable invariant), then
    spot-checks _VALIDATION_PROBES random cross-tile pairs with has_path. Fails loud on any break.
    """
    graph = read_full_graph(graph_dir=out_dir)
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        raise ValueError("Validation: final graph has <2 nodes — nothing to connect.")
    n_components = nx.number_strongly_connected_components(graph)
    if n_components != 1:
        raise ValueError(
            f"Validation FAILED: merged graph has {n_components} strongly-connected components "
            "(expected 1) — the network is fragmented across regions."
        )
    rng = random.Random(0)  # deterministic probe selection (reproducible pass/fail)
    for _ in tqdm(range(_VALIDATION_PROBES), desc="4/4 Validating connectivity", unit="probe"):
        source, target = _random_cross_tile_pair(graph=graph, nodes=nodes, rng=rng)
        if not nx.has_path(graph, source, target):
            raise ValueError(
                f"Validation FAILED: no path between cross-region nodes {source} and {target} — "
                "the merged network is fragmented."
            )
    logger.info(
        "Validation OK: 1 strongly-connected component; %d cross-region pairs all connected",
        _VALIDATION_PROBES,
    )


def _random_cross_tile_pair(*, graph: nx.MultiDiGraph, nodes: list[int], rng: random.Random) -> tuple[int, int]:
    """Two random node ids whose tiles differ (deliberately cross-border), for a connectivity probe."""
    for _ in range(1000):  # bounded retries; distinct tiles are overwhelmingly common
        source, target = rng.choice(nodes), rng.choice(nodes)
        s_tile = tile_index(lat=graph.nodes[source]["y"], lon=graph.nodes[source]["x"])
        t_tile = tile_index(lat=graph.nodes[target]["y"], lon=graph.nodes[target]["x"])
        if s_tile != t_tile:
            return source, target
    raise ValueError("Validation: could not find a cross-tile node pair (graph too small?).")


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

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )

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

    # Phase 1 — DOWNLOAD every region's raw pbf up front (skip-if-present). Any failure aborts.
    pbfs = {
        region_key: _download_region(region_key=region_key, geofabrik_path=geofabrik_path)
        for region_key, geofabrik_path in tqdm(regions.items(), desc="1/4 Downloading pbfs", unit="region")
    }

    # Phase 2 — PROCESS each region in isolation: build, remap to contiguous ids, write its own
    # app-loadable artifact, then free the ~20 GB graph before the next (peak RSS ≈ one region).
    # Skip regions already flagged confirmed_complete; a build failure aborts (real bug).
    for region_key, pbf in tqdm(pbfs.items(), desc="2/4 Building regions", unit="region"):
        if _region_complete(region_key=region_key):
            logger.info("%s: skipped, already complete", region_key)
            continue
        _process_region(region_key=region_key, pbf=pbf, dem=dem, bbox=bbox, tolerance_m=tolerance_m)
        logger.info("%s: peak RSS %.1f GB", region_key, _peak_rss_gb())

    # Phase 3 — COMBINE per-region artifacts into globally-consistent tiled shards.
    built = sorted(regions)
    _assert_all_regions_complete(regions=built)
    nodes_df, edges_df = _combine_regions(regions=built)
    meta = {
        "bbox": list(compute_bbox(nodes_df=nodes_df)),
        "tile_deg": GraphConfig.TILE_DEG,
        "tolerance_m": tolerance_m,
        "n_nodes": int(len(nodes_df)),
        "n_edges": int(len(edges_df)),
        "n_stations": int((nodes_df["node_type"] == NodeType.RAIL).sum()),
        "regions_built": built,
    }
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=meta, out_dir=out_dir)
    del nodes_df, edges_df
    gc.collect()

    # Phase 4 — VALIDATE: load the saved production graph and confirm cross-region connectivity.
    _validate_connectivity(out_dir=out_dir)

    print(json.dumps(meta, indent=2))
    print(f"\nBuilt {len(built)} regions in {(time.time() - started) / 60:.1f} min | peak RSS {_peak_rss_gb():.1f} GB")
    print(f"Artifact: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
