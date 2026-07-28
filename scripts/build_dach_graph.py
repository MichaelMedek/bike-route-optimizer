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
import random
import resource
import socket
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import networkx as nx
import osmnx as ox
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")  # headless — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from shapely import from_wkt  # noqa: E402

from bike_router.builder import (
    build_region_graph_clipped,
    dedup_by_geometry,
    reindex_region,
    remap_contiguous,
    stage_pbf,
)
from bike_router.constants import DEMConfig, GraphConfig, Mode, NodeType, OutputConfig, Palette
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

# One log format for parent AND spawned workers (spawn children don't inherit the parent's config).
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


def _configure_logging() -> None:
    """Set up INFO logging — called in main() AND as each worker's initializer (spawn needs both).

    Also routes osmnx's own logs through Python logging.
    """
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    ox.settings.log_console = True
    ox.settings.log_level = logging.INFO


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
# Per-region staged-pbf ceiling: a region whose osmium clip exceeds this parses into too big a graph
# for one worker's RAM (niedersachsen at 478 MB is the proven-OK high-water mark). Preflight fails
# BEFORE the build loop so an oversized split is caught in seconds, not mid-build.
_MAX_STAGED_PBF_MB = 500.0

Bbox = tuple[float, float, float, float]  # (west, south, east, north) in WGS84 degrees


def split_geofabrik_path(geofabrik_path: str) -> str:
    """The Geofabrik leaf name (cache filename) — bbox-split halves share it so a pbf downloads once."""
    return geofabrik_path.rsplit("/", maxsplit=1)[-1]


@dataclass(frozen=True)
class Region:
    """One region to build: a unique output key, its Geofabrik pbf, and an optional bbox clip.

    ``bbox`` splits a too-big whole-country pbf into memory-bounded halves that share one downloaded pbf.
    Adjacent halves OVERLAP by ~0.5° (~75 km, > the ~29 km longest rail edge);
    Phase-3 geometry-dedup then collapses the duplicated seam (the same mechanism that already
    merges Geofabrik regions, which overlap 60–90 km). ``pbf_name`` is the shared cache filename.
    """

    key: str
    geofabrik_path: str
    bbox: Bbox | None = None

    @property
    def pbf_name(self) -> str:
        """Cache filename for the raw pbf — the Geofabrik leaf, so split halves reuse one download."""
        return split_geofabrik_path(geofabrik_path=self.geofabrik_path)


# DACH at Geofabrik sub-region granularity. Big Flächenländer are split into their
# Regierungsbezirke (bounded per-region memory); smaller states stay whole.
# Austria and Switzerland have NO Geofabrik sub-extracts, so they are bbox-split east/west here.
_AUSTRIA = "austria"  # extent ~9.53–17.16 E; dense in the east → split into 3 overlapping slices
_SWITZERLAND = "switzerland"  # extent ~5.96–10.49 E; split east/west at 8.22, ±0.5° overlap
DACH_REGIONS: list[Region] = [
    # Baden-Württemberg
    Region("freiburg-regbez", "germany/baden-wuerttemberg/freiburg-regbez"),
    Region("karlsruhe-regbez", "germany/baden-wuerttemberg/karlsruhe-regbez"),
    Region("stuttgart-regbez", "germany/baden-wuerttemberg/stuttgart-regbez"),
    Region("tuebingen-regbez", "germany/baden-wuerttemberg/tuebingen-regbez"),
    # Bayern
    Region("mittelfranken", "germany/bayern/mittelfranken"),
    Region("niederbayern", "germany/bayern/niederbayern"),
    Region("oberbayern", "germany/bayern/oberbayern"),
    Region("oberfranken", "germany/bayern/oberfranken"),
    Region("oberpfalz", "germany/bayern/oberpfalz"),
    Region("schwaben", "germany/bayern/schwaben"),
    Region("unterfranken", "germany/bayern/unterfranken"),
    # Nordrhein-Westfalen
    Region("arnsberg-regbez", "germany/nordrhein-westfalen/arnsberg-regbez"),
    Region("detmold-regbez", "germany/nordrhein-westfalen/detmold-regbez"),
    Region("duesseldorf-regbez", "germany/nordrhein-westfalen/duesseldorf-regbez"),
    Region("koeln-regbez", "germany/nordrhein-westfalen/koeln-regbez"),
    Region("muenster-regbez", "germany/nordrhein-westfalen/muenster-regbez"),
    # Remaining German states (whole — each smaller than a big Flächenland regbez)
    Region("berlin", "germany/berlin"),
    Region("brandenburg", "germany/brandenburg"),
    Region("bremen", "germany/bremen"),
    Region("hamburg", "germany/hamburg"),
    Region("hessen", "germany/hessen"),
    Region("mecklenburg-vorpommern", "germany/mecklenburg-vorpommern"),
    Region("niedersachsen", "germany/niedersachsen"),
    Region("rheinland-pfalz", "germany/rheinland-pfalz"),
    Region("saarland", "germany/saarland"),
    Region("sachsen", "germany/sachsen"),
    Region("sachsen-anhalt", "germany/sachsen-anhalt"),
    Region("schleswig-holstein", "germany/schleswig-holstein"),
    Region("thueringen", "germany/thueringen"),
    # Austria — one pbf, split into THREE overlapping W/Center/E slices (±0.5° overlap).
    Region("austria-west", _AUSTRIA, bbox=(9.4, 46.3, 13.5, 49.1)),
    Region("austria-center", _AUSTRIA, bbox=(13.0, 46.3, 15.8, 49.1)),
    Region("austria-east", _AUSTRIA, bbox=(15.3, 46.3, 17.25, 49.1)),
    # Switzerland — one pbf, split east/west at 8.22° E with ±0.5° (~75 km) overlap.
    Region("switzerland-west", _SWITZERLAND, bbox=(5.85, 45.7, 8.72, 47.9)),
    Region("switzerland-east", _SWITZERLAND, bbox=(7.72, 45.7, 10.55, 47.9)),
]

_MIN_SPLIT_OVERLAP_DEG = 0.5  # adjacent tiles must overlap ≥ this on the split axis (> longest edge)


def _assert_rectangular_tiling(*, pbf: str, a_key: str, a: Bbox, b_key: str, b: Bbox) -> None:
    """Assert two sibling tiles form a clean rectangular grid pair: aligned on one axis, overlapping
    on the other by ≥ the minimum. Split-axis = the offset axis; perpendicular axis MUST match exactly
    (no ragged tiles). Rejects diagonal/gapped/ragged configs. Symmetric — works for lon OR lat splits.
    """
    aw, as_, ae, an = a
    bw, bs, be, bn = b
    lon_aligned = abs(aw - bw) < 1e-9 and abs(ae - be) < 1e-9  # identical lon range → split is by lat
    lat_aligned = abs(as_ - bs) < 1e-9 and abs(an - bn) < 1e-9  # identical lat range → split is by lon
    if lat_aligned:  # longitudinal bands: perpendicular (lat) matches; require lon overlap
        overlap = min(ae, be) - max(aw, bw)
        axis = "lon"
    elif lon_aligned:  # latitudinal bands: perpendicular (lon) matches; require lat overlap
        overlap = min(an, bn) - max(as_, bs)
        axis = "lat"
    else:
        raise AssertionError(
            f"{pbf}: {a_key}/{b_key} are not a rectangular tiling — neither lon nor lat range is shared "
            "(tiles must be aligned bands; ragged/diagonal splits are forbidden)."
        )
    assert overlap >= _MIN_SPLIT_OVERLAP_DEG, (
        f"{pbf}: {a_key}∩{b_key} {axis} overlap {overlap:.2f}° < {_MIN_SPLIT_OVERLAP_DEG}° — seam won't stitch"
    )


def _assert_split_overlaps(regions: list[Region]) -> None:
    """Import-time invariant: slices sharing one pbf must be a valid OVERLAPPING RECTANGULAR TILING.

    Consecutive tiles (sorted along the split axis) must be aligned on the perpendicular axis and
    overlap ≥0.5° on the split axis, else Phase-3 dedup can't stitch the seam. Fails loud on any
    ragged/gapped/diagonal split. Supports lon-band and lat-band splits (auto-detected per pair).
    """
    by_pbf: dict[str, list[tuple[str, Bbox]]] = {}
    for r in regions:
        if r.bbox is not None:
            by_pbf.setdefault(r.geofabrik_path, []).append((r.key, r.bbox))
    for pbf, slices in by_pbf.items():
        # Sort along whichever axis varies (west edge if lon-split, south edge if lat-split).
        lon_varies = len({round(b[0], 6) for _k, b in slices}) > 1
        ordered = sorted(slices, key=lambda kb: kb[1][0] if lon_varies else kb[1][1])
        for (a_key, a), (b_key, b) in zip(ordered[:-1], ordered[1:], strict=True):
            _assert_rectangular_tiling(pbf=pbf, a_key=a_key, a=a, b_key=b_key, b=b)


_assert_split_overlaps(DACH_REGIONS)  # validate the split config at import — a bad split never runs


def _download_pbf(*, geofabrik_path: str) -> Path:
    """Download a region's .osm.pbf if missing (atomic: temp + rename). Returns its path.

    Cached by the Geofabrik leaf name, so bbox-split halves sharing one pbf download it once.
    A failed transfer unlinks its partial .tmp and re-raises so no orphan/partial lingers.
    """
    _PBF_DIR.mkdir(parents=True, exist_ok=True)
    dest = _PBF_DIR / f"{split_geofabrik_path(geofabrik_path=geofabrik_path)}.osm.pbf"
    if dest.exists() and dest.stat().st_size > 0:
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


def _region_complete(region_key: str) -> bool:
    """True if a region's per-region artifact exists and is flagged confirmed_complete."""
    meta_path = _REGIONS_DIR / region_key / GraphConfig.META_FILENAME
    if not meta_path.exists():
        return False
    return bool(json.loads(meta_path.read_text()).get("confirmed_complete", False))


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
            logger.info(f"{region.key}: staged {size_mb:.0f} MB")
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
        "bbox": list(compute_bbox(nodes_df=nodes_df)),
        "tile_deg": GraphConfig.TILE_DEG,
        "tolerance_m": tolerance_m,
        "n_nodes": int(len(nodes_df)),
        "n_edges": int(len(edges_df)),
        "n_stations": int((nodes_df["node_type"] == NodeType.RAIL).sum()),
        "confirmed_complete": True,  # written LAST via write_graph_parquet → the atomic "done" flag
    }
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=meta, out_dir=region_dir)
    return _peak_rss_gb()  # this child's peak; the process then exits and the OS reclaims everything


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
    logger.info(f"Validation OK: 1 strongly-connected component; {_VALIDATION_PROBES} cross-region pairs all connected")


def _random_cross_tile_pair(*, graph: nx.MultiDiGraph, nodes: list[int], rng: random.Random) -> tuple[int, int]:
    """Two random node ids whose tiles differ (deliberately cross-border), for a connectivity probe."""
    for _ in range(1000):  # bounded retries; distinct tiles are overwhelmingly common
        source, target = rng.choice(nodes), rng.choice(nodes)
        s_tile = tile_index(lat=graph.nodes[source]["y"], lon=graph.nodes[source]["x"])
        t_tile = tile_index(lat=graph.nodes[target]["y"], lon=graph.nodes[target]["x"])
        if s_tile != t_tile:
            return source, target
    raise ValueError("Validation: could not find a cross-tile node pair (graph too small?).")


def _plot_overview(*, edges_df: pd.DataFrame, out_path: Path) -> None:
    """Save a minimalist matplotlib overview of the final graph: bike edges thin blue, rail thick purple.

    A quick visual sanity-check of the whole DACH result before deciding to upload. Station links
    (short bike↔rail hops) are omitted — the two networks' shapes are what matters here.
    """
    fig, ax = plt.subplots(figsize=(16, 18))
    # Draw bike first (thin blue), rail on top (thicker purple) so rail reads clearly over the mesh.
    for mode, color, width in ((Mode.BIKE, Palette.START, 0.15), (Mode.RAIL, Palette.RAIL, 0.9)):
        for wkt in edges_df.loc[edges_df["mode"] == mode, "geometry_wkt"]:
            if isinstance(wkt, str):
                coords = from_wkt(wkt).coords
                ax.plot([c[0] for c in coords], [c[1] for c in coords], color=color, linewidth=width)
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
    pbfs = {
        gp: _download_region(geofabrik_path=gp)
        for gp in tqdm(sorted({r.geofabrik_path for r in regions}), desc="1/4 Downloading pbfs", unit="pbf")
    }

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
            if _region_complete(region_key=region.key):
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
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=meta, out_dir=out_dir, compression="zstd")
    _plot_overview(edges_df=edges_df, out_path=OutputConfig.OUTPUT_DIR / "dach_graph_overview.png")
    del nodes_df, edges_df
    gc.collect()

    # Phase 4 — VALIDATE: load the saved production graph and confirm cross-region connectivity.
    _validate_connectivity(out_dir=out_dir)

    print(json.dumps(meta, indent=2))
    print(
        f"\nBuilt {len(built)} regions in {(time.time() - started) / 60:.1f} min | "
        f"max per-region child peak RSS {max_child_rss:.1f} GB"
    )
    print(f"Artifact: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
