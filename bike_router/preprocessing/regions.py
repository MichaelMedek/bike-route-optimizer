"""DACH region-split config + Phase-3/4 combine, prune, and validation (build-time).

Extracted from the build script so the reusable graph-combination logic is unit-testable
in the preprocessing layer: the region-split declaration + its overlap invariant, the
per-region completion gate, the cumulative-offset combine with seam dedup, and the
component prune (rail strict / bike keep-if-big). The script keeps only orchestration
(download, multiprocessing, plotting, main).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

from bike_router.core.constants import GraphConfig, Mode, NodeType
from bike_router.preprocessing.builder import dedup_by_geometry, reindex_region, remap_contiguous
from bike_router.preprocessing.graph_writer import compute_bbox, read_region_tables, undirected_graph_from_edges

logger = logging.getLogger(__name__)

Bbox = tuple[float, float, float, float]  # (west, south, east, north) in WGS84 degrees

# Sanity ceiling on the merged node count (DACH is ~3–5M); a larger total means a logic error
# upstream, so Phase 3 fails fast rather than writing a suspect artifact.
_MAX_TOTAL_NODES = 100_000_000_000
_MIN_SPLIT_OVERLAP_DEG = 0.5  # adjacent tiles must overlap ≥ this on the split axis (> longest edge)


def split_geofabrik_path(geofabrik_path: str) -> str:
    """The Geofabrik leaf name (cache filename) — bbox-split halves share it so a pbf downloads once."""
    return geofabrik_path.rsplit("/", maxsplit=1)[-1]


@dataclass(frozen=True)
class Region:
    """One region to build: a unique output key, its Geofabrik pbf, and an optional bbox clip.

    ``bbox`` splits a too-big country pbf into memory-bounded halves sharing one download; adjacent halves
    OVERLAP ~0.5° (> the ~29 km longest rail edge) so Phase-3 geometry-dedup collapses the duplicated seam.
    """

    key: str
    geofabrik_path: str
    bbox: Bbox | None = None

    @property
    def pbf_name(self) -> str:
        """Cache filename for the raw pbf — the Geofabrik leaf, so split halves reuse one download."""
        return split_geofabrik_path(geofabrik_path=self.geofabrik_path)


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

    Consecutive tiles (sorted along the split axis) must align on the perpendicular axis and overlap ≥0.5°
    on the split axis, else Phase-3 dedup can't stitch the seam. Fails loud on any ragged/gapped/diagonal split.
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

_assert_split_overlaps(regions=DACH_REGIONS)  # validate the split config at import — a bad split never runs


def region_complete(*, regions_dir: Path, region_key: str) -> bool:
    """True if a region's per-region artifact exists and is flagged confirmed_complete."""
    meta_path = regions_dir / region_key / GraphConfig.META_FILENAME
    if not meta_path.exists():
        return False
    return bool(json.loads(meta_path.read_text()).get("confirmed_complete", False))


def assert_all_regions_complete(*, regions_dir: Path, regions: list[str]) -> None:
    """Fail loud unless EVERY region has a confirmed_complete per-region artifact (Phase 3 gate)."""
    missing = [r for r in regions if not region_complete(regions_dir=regions_dir, region_key=r)]
    if missing:
        raise ValueError(
            f"Cannot combine: {len(missing)} region(s) not confirmed_complete: {', '.join(missing)}. "
            "Re-run to (re)build them."
        )


def base_meta(*, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, tolerance_m: float) -> dict[str, object]:
    """The meta.json keys common to a per-region artifact and the combined DACH artifact."""
    return {
        "bbox": list(compute_bbox(nodes_df=nodes_df)),
        "tile_deg": GraphConfig.TILE_DEG,
        "tolerance_m": tolerance_m,
        "n_nodes": int(len(nodes_df)),
        "n_edges": int(len(edges_df)),
        "n_stations": int((nodes_df["node_type"] == NodeType.RAIL).sum()),
    }


def combine_regions(*, regions_dir: Path, regions: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase 3: offset each region's contiguous ids into a global space, dedup the seam, re-densify.

    A running offset (ΣN of earlier regions) makes ids collision-free; geometry dedup collapses border
    duplicates; a prune drops strays; a final remap closes the holes (n_nodes == max_id + 1). Returns (nodes, edges).
    """
    node_frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    offset = 0
    for region_key in regions:
        nodes_df, edges_df = read_region_tables(region_dir=regions_dir / region_key)
        nodes_df, edges_df = reindex_region(nodes_df=nodes_df, edges_df=edges_df, offset=offset)
        offset += len(nodes_df)
        node_frames.append(nodes_df)
        edge_frames.append(edges_df)
    if offset >= _MAX_TOTAL_NODES:
        raise ValueError(f"Combined node count {offset} exceeds sanity ceiling {_MAX_TOTAL_NODES} — aborting.")
    nodes_df = pd.concat(node_frames, ignore_index=True)
    edges_df = pd.concat(edge_frames, ignore_index=True)
    nodes_df, edges_df = dedup_by_geometry(nodes_df=nodes_df, edges_df=edges_df)
    # ONE global component prune, AFTER dedup has stitched the region seams: rail → single component,
    # bike → keep every island ≥ MIN_BIKE_COMPONENT_KM. The sole connectivity gate (no separate pass).
    nodes_df, edges_df = prune_components(nodes_df=nodes_df, edges_df=edges_df)
    # Dedup + prune removed nodes, leaving id holes → renumber to dense 0..N-1 (n_nodes==max_id+1).
    return remap_contiguous(nodes_df=nodes_df, edges_df=edges_df)


def prune_components(*, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop stray components: RAIL strict (largest comp only), BIKE keep-if-big. Endpoint-only, no geometry.

    A region is a CLIP, so bike roads legitimately split into pieces connecting only via neighbours: keep the
    largest RAIL comp but EVERY bike comp ≥ MIN_BIKE_COMPONENT_KM; an edge lives iff both endpoints survive.
    """
    bike_e = edges_df[edges_df["mode"] == Mode.BIKE]
    rail_e = edges_df[edges_df["mode"] == Mode.RAIL]
    if bike_e.empty:
        raise ValueError("prune: no bike edges in the merged graph — corrupt build (bike network missing).")
    if rail_e.empty:
        raise ValueError("prune: no rail edges in the merged graph — corrupt build (rail network missing).")

    # RAIL: largest weakly-connected component only.
    gr = undirected_graph_from_edges(edges_df=rail_e)
    keep_rail = max(nx.connected_components(gr), key=len) if gr.number_of_nodes() else set()

    # BIKE: every weakly-connected component with total length ≥ threshold.
    gb = undirected_graph_from_edges(edges_df=bike_e, weight_col="length_m")
    keep_bike: set[int] = set()
    kept_comps = dropped_comps = 0
    for comp in nx.connected_components(gb):
        sub = gb.subgraph(comp)
        km = sub.size(weight="weight") / 1000.0
        if km >= GraphConfig.MIN_BIKE_COMPONENT_KM:
            keep_bike |= comp
            kept_comps += 1
        else:
            dropped_comps += 1

    keep = keep_bike | keep_rail
    n_before, e_before = len(nodes_df), len(edges_df)
    nodes_df = nodes_df[nodes_df["osmid"].isin(keep)]
    edges_df = edges_df[edges_df["from_node"].isin(keep) & edges_df["to_node"].isin(keep)]
    logger.info(
        f"prune: bike kept {kept_comps} comps (dropped {dropped_comps} <{GraphConfig.MIN_BIKE_COMPONENT_KM:.0f}km), "
        f"rail 1 comp → {len(nodes_df)}/{n_before} nodes, {len(edges_df)}/{e_before} edges"
    )
    return nodes_df, edges_df
