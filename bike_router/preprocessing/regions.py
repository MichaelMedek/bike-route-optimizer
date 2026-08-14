"""DACH region-split config + Phase-3/4 combine, prune, and validation (build-time).

The reusable graph-combination logic, unit-testable in the preprocessing layer: region-split + overlap
invariant, completion gate, cumulative-offset combine with seam dedup, component prune (rail strict / bike keep-if-big).
"""

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely import from_wkt
from shapely.geometry import LineString

from bike_router.core.constants import GeoConfig, GraphConfig, Mode, NodeType, RailConfig, Schema
from bike_router.core.geo import haversine_vec
from bike_router.core.graph_store import EDGE_COLS, NODE_COLS
from bike_router.preprocessing.builder import dedup_by_geometry, reindex_region, remap_contiguous
from bike_router.preprocessing.graph_ops import densify_polyline
from bike_router.preprocessing.graph_writer import compute_bbox, read_region_tables

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


def merge_station_platforms(*, station_xy: "np.ndarray", merge_m: float) -> "np.ndarray":
    """Union platforms within ``merge_m`` of each other into one complex; return each platform's root label.

    Union-find over the KD-tree radius pairs — co-located platforms (e.g. Eyach / Eyach HzL, 33 m) collapse
    to one node while a duplicate name 100 km away stays separate (merge is by geometry, not name).
    """
    n = len(station_xy)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in cKDTree(station_xy).query_pairs(r=merge_m):
        parent[find(a)] = find(b)
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def _trace_to_root(*, node: int, parent: dict[int, int]) -> list[int]:
    """The vertex chain from a BFS root (parent == -1) down to ``node``, in root→node order."""
    chain: list[int] = []
    while node != -1:
        chain.append(node)
        node = parent[node]
    return chain[::-1]


def _trace_via_predecessors(*, node: int, pred: "np.ndarray") -> list[int]:
    """The welded-track chain from ``node`` to its nearest seed, following scipy's predecessor tree.

    ``pred`` is dijkstra(min_only) predecessors: ``pred[v]`` steps toward v's nearest seed (or -9999 at it).
    Returns node→seed order (so the arrival vertex is first), tracing real track vertices the whole way.
    """
    chain = [node]
    while pred[node] >= 0:
        node = int(pred[node])
        chain.append(node)
    return chain


def watershed_station_adjacency(
    *, coords: "np.ndarray", neighbours: list[list[int]], region: "np.ndarray", seal_m: float
) -> tuple[dict[int, set[int]], dict[tuple[int, int], list[int]]]:
    """Per-complex BFS over welded track vertices → symmetric complex adjacency + the track path per pair.

    Multi-source Dijkstra owns each vertex by nearest complex; a flood halts at another region or its throat
    (within ``seal_m``), else walks through. Returns adjacency AND each pair's welded-vertex path (real track).
    """
    n_vertices = len(coords)
    # Vectorized undirected CSR: flatten the adjacency list, keep u<v, weight = euclidean edge length.
    degrees = np.fromiter((len(a) for a in neighbours), dtype=np.int64, count=n_vertices)
    us = np.repeat(np.arange(n_vertices), degrees)
    vs = np.concatenate([np.asarray(a, dtype=np.int64) for a in neighbours]) if n_vertices else np.empty(0, np.int64)
    upper = us < vs
    u2, v2 = us[upper], vs[upper]
    w = np.hypot(coords[u2, 0] - coords[v2, 0], coords[u2, 1] - coords[v2, 1])
    csr = csr_matrix(
        (np.concatenate([w, w]), (np.concatenate([u2, v2]), np.concatenate([v2, u2]))),
        shape=(n_vertices, n_vertices),
    )
    seeds = np.where(region >= 0)[0]
    owner_dist, pred, source = dijkstra(csr, indices=seeds, min_only=True, return_predecessors=True)
    # source is scipy's per-vertex nearest-seed (or -9999 when a vertex is unreachable from every seed);
    # clip the sentinel before indexing so an isolated track component just gets owner -1 (walk-through).
    owner = np.where(source >= 0, region[np.clip(source, 0, n_vertices - 1)], -1)
    seeds_by_complex: dict[int, list[int]] = {}
    for v in seeds:
        seeds_by_complex.setdefault(int(region[v]), []).append(int(v))
    visited = np.full(n_vertices, -1, dtype=np.int64)
    # A step to nb arrives at complex `arrival[nb]` (>=0) or is walk-through (-1): its own region's vertices
    # and throat vertices (owned within seal, off-region) are arrivals; everything else keeps flooding.
    arrival = np.where(region >= 0, region, np.where((owner >= 0) & (owner_dist <= seal_m), owner, -1))
    graph: dict[int, set[int]] = {}
    paths: dict[tuple[int, int], list[int]] = {}

    def flood(rep: int, rep_seeds: list[int]) -> None:
        """One complex's BFS: record the first complex each branch arrives at + the FULL welded path to it.

        The path is stitched from two real-track halves: rep-seed→arrival (this flood's BFS parent tree) and
        arrival→other-seed (scipy's nearest-seed predecessor tree), so it spans platform-to-platform with no gap.
        """
        parent = {s: -1 for s in rep_seeds}
        queue = deque(rep_seeds)
        for s in rep_seeds:
            visited[s] = rep
        while queue:
            cur = queue.popleft()
            for nb in neighbours[cur]:
                if visited[nb] == rep:
                    continue
                visited[nb] = rep
                parent[nb] = cur
                reached = int(arrival[nb])
                if reached < 0 or reached == rep:
                    queue.append(nb)
                    continue
                graph.setdefault(rep, set()).add(reached)
                key = (min(rep, reached), max(rep, reached))
                if key not in paths:  # first BFS arrival wins; stitch both real-track halves at the arrival vertex
                    head = _trace_to_root(node=nb, parent=parent)  # rep-seed → arrival vertex
                    tail = _trace_via_predecessors(node=nb, pred=pred)  # arrival vertex → its nearest (other) seed
                    paths[key] = head + tail[1:]  # drop the duplicated arrival vertex where the halves meet

    for rep, rep_seeds in seeds_by_complex.items():
        flood(rep, rep_seeds)
    for a in list(graph):
        for b in list(graph[a]):
            graph.setdefault(b, set()).add(a)
    return graph, paths


def weld_rail_track(*, geometries: "np.ndarray", weld_m: float) -> tuple["np.ndarray", list[list[int]], "np.ndarray"]:
    """Fuse rail WKT LINESTRINGs into ONE planar graph, welding shared vertices on a ``weld_m`` grid.

    Returns metric ``(V,2)`` coords, an undirected adjacency list, and each vertex's real ``(lon,lat,z)``
    (NULL geometry dropped) so a track path can be re-emitted as a true polyline.
    """
    mpd = GeoConfig.METERS_PER_DEGREE_EQUATOR
    coord_to_vertex: dict[tuple[int, int], int] = {}
    xs: list[float] = []
    ys: list[float] = []
    lonlatz: list[tuple[float, float, float]] = []
    adjacency: dict[int, set[int]] = {}
    for wkt in geometries:
        if not isinstance(wkt, str):
            continue  # corrupt builder connector (no geometry) — not real track
        raw = np.asarray(from_wkt(wkt).coords)
        px = raw[:, 0] * mpd * np.cos(np.radians(raw[:, 1]))
        py = raw[:, 1] * mpd
        previous = -1
        for j, (x, y) in enumerate(zip(px, py, strict=True)):
            key = (round(x / weld_m), round(y / weld_m))
            vertex = coord_to_vertex.get(key)
            if vertex is None:
                vertex = len(xs)
                coord_to_vertex[key] = vertex
                xs.append(float(x))
                ys.append(float(y))
                z = float(raw[j, 2]) if raw.shape[1] >= 3 else float(np.nan)
                lonlatz.append((float(raw[j, 0]), float(raw[j, 1]), z))
            if previous not in (-1, vertex):
                adjacency.setdefault(previous, set()).add(vertex)
                adjacency.setdefault(vertex, set()).add(previous)
            previous = vertex
    coords = np.column_stack([xs, ys]) if xs else np.empty((0, 2))
    neighbours = [sorted(adjacency.get(v, set())) for v in range(len(xs))]
    vertex_lonlatz = np.array(lonlatz, dtype=np.float64) if lonlatz else np.empty((0, 3))
    return coords, neighbours, vertex_lonlatz


def rail_edge(
    *,
    a: int,
    b: int,
    latlon: dict[int, tuple[float, float]],
    elev_by_osmid: dict[int, float],
    polyline: "np.ndarray | None",
) -> dict[str, object]:
    """One directed station→station rail row (a=from, b=to).

    Geometry is the real welded-track ``polyline`` (an ``(n,3)`` lon/lat/z path oriented a→b), ANCHORED at
    both ends to the fixed station coords and densified so no segment exceeds RAIL_MAX_VERTEX_SPACING_M.
    """
    la, lo_a = latlon[a]
    lb, lo_b = latlon[b]
    ea, eb = elev_by_osmid[a], elev_by_osmid[b]
    # Anchor to the FIXED station nodes so the drawn line connects to them (endpoints == edge nodes); the
    # traced track sits between. Densify so a sparse-OSM straight (real track, few vertices) can't span >200 m.
    track = polyline if polyline is not None and len(polyline) >= 1 else np.empty((0, 3))
    anchored = np.vstack([[lo_a, la, ea], track, [lo_b, lb, eb]])
    dense = densify_polyline(anchored, max_spacing_m=RailConfig.RAIL_MAX_VERTEX_SPACING_M)
    length_m = float(
        haversine_vec(lat_a=dense[:-1, 1], lon_a=dense[:-1, 0], lat_b=dense[1:, 1], lon_b=dense[1:, 0]).sum()
    )
    return {
        Schema.FROM_NODE: a,
        Schema.TO_NODE: b,
        Schema.KEY: 0,
        Schema.LENGTH_M: length_m,
        Schema.HEIGHT_DIFF_M: eb - ea,
        Schema.SURFACE: None,
        Schema.HIGHWAY: None,
        Schema.MODE: Mode.RAIL,
        Schema.GEOMETRY_WKT: LineString([(float(x), float(y), float(z)) for x, y, z in dense]).wkt,
    }


def _oriented_track_polyline(
    *, vpath: "list[int] | None", vertex_lonlatz: "np.ndarray", from_lon: float, from_lat: float
) -> "np.ndarray | None":
    """The welded-track lon/lat/z path for a vertex chain, flipped to start nearest ``(from_lon, from_lat)``."""
    if vpath is None or len(vpath) < 2:
        return None
    line = vertex_lonlatz[vpath]  # (n, 3) lon/lat/z along the real track
    d0 = (line[0, 0] - from_lon) ** 2 + (line[0, 1] - from_lat) ** 2
    d1 = (line[-1, 0] - from_lon) ** 2 + (line[-1, 1] - from_lat) ** 2
    return line if d0 <= d1 else line[::-1]


def _build_rail_rows(
    *,
    adjacency: dict[int, set[int]],
    paths: dict[tuple[int, int], list[int]],
    survivor: dict[int, int],
    latlon: dict[int, tuple[float, float]],
    elev_by_osmid: dict[int, float],
    vertex_lonlatz: "np.ndarray",
) -> list[dict[str, object]]:
    """Both-direction rail edge rows for every adjacent complex pair, each tracing the real welded track."""
    seen: set[tuple[int, int]] = set()
    rows: list[dict[str, object]] = []
    for a_root, roots in adjacency.items():
        for b_root in roots:
            sa, sb = survivor[int(a_root)], survivor[int(b_root)]
            if sa == sb or (min(sa, sb), max(sa, sb)) in seen:
                continue
            seen.add((min(sa, sb), max(sa, sb)))
            vpath = paths.get((min(int(a_root), int(b_root)), max(int(a_root), int(b_root))))
            fwd = _oriented_track_polyline(
                vpath=vpath, vertex_lonlatz=vertex_lonlatz, from_lon=latlon[sa][1], from_lat=latlon[sa][0]
            )
            rows.append(rail_edge(a=sa, b=sb, latlon=latlon, elev_by_osmid=elev_by_osmid, polyline=fwd))
            rows.append(
                rail_edge(
                    a=sb, b=sa, latlon=latlon, elev_by_osmid=elev_by_osmid, polyline=None if fwd is None else fwd[::-1]
                )
            )
    return rows


def consolidate_rail(*, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace the corrupt rail layer with a clean station↔station graph (one edge per adjacent pair).

    Drops NULL-geometry connectors + unnamed track nodes, merges platforms within STATION_MERGE_M, and emits
    one straight rail edge per watershed-adjacent station pair; bike + station-access edges pass through.
    """
    is_rail_node = nodes_df[Schema.NODE_TYPE] == NodeType.RAIL
    is_station = is_rail_node & nodes_df[Schema.STATION_NAME].notna()
    is_rail_edge = edges_df[Schema.MODE] == Mode.RAIL
    if not is_station.any() or not is_rail_edge.any():
        return nodes_df, edges_df  # no rail to consolidate (e.g. a bike-only synthetic frame)

    stations = nodes_df[is_station].reset_index(drop=True)
    lat = stations[Schema.LAT].to_numpy(dtype=np.float64)
    lon = stations[Schema.LON].to_numpy(dtype=np.float64)
    station_xy = np.column_stack(
        [lon * GeoConfig.METERS_PER_DEGREE_EQUATOR * np.cos(np.radians(lat)), lat * GeoConfig.METERS_PER_DEGREE_EQUATOR]
    )
    osmids = stations[Schema.OSMID].to_numpy(dtype=np.int64)
    elevs = stations[Schema.ELEVATION_M].to_numpy(dtype=np.float64)
    complex_of = merge_station_platforms(station_xy=station_xy, merge_m=RailConfig.STATION_MERGE_M)

    # Per complex, sorted by (name-length, name, osmid): first row = shorter name; min-osmid = survivor id.
    grp = pd.DataFrame({"root": complex_of, "sid": osmids, "row": np.arange(len(osmids))})
    grp["name"] = stations[Schema.STATION_NAME].astype(str).to_numpy()
    grp["namelen"] = grp["name"].str.len()
    by_osmid = grp.sort_values(["root", "sid"]).groupby("root")
    survivor = by_osmid["sid"].first().astype(np.int64).to_dict()
    survivor_row = by_osmid["row"].first().astype(np.int64).to_dict()
    kept_name = grp.sort_values(["root", "namelen", "name"]).groupby("root")["name"].first().to_dict()
    platform_to_survivor = {int(o): int(survivor[int(r)]) for o, r in zip(osmids, complex_of, strict=True)}

    coords, neighbours, vertex_lonlatz = weld_rail_track(
        geometries=edges_df.loc[is_rail_edge, Schema.GEOMETRY_WKT].to_numpy(), weld_m=RailConfig.TRACK_WELD_M
    )
    logger.info(f"consolidate_rail: welded {len(coords)} track vertices from {len(stations)} platforms; watershed …")
    region = np.full(len(coords), -1, dtype=np.int64)
    if len(coords):
        # Track vertices within merge_m of ANY platform get their nearest platform's complex (vectorized:
        # one nearest-platform query, then keep only those inside the radius).
        dist, nearest = cKDTree(station_xy).query(coords)
        inside = dist <= RailConfig.STATION_MERGE_M
        region[inside] = complex_of[nearest[inside]]
    adjacency, paths = watershed_station_adjacency(
        coords=coords, neighbours=neighbours, region=region, seal_m=RailConfig.STATION_THROAT_SEAL_M
    )

    elev_by_osmid = {int(osmids[i]): float(elevs[i]) for i in range(len(osmids))}
    latlon = {int(osmids[i]): (float(lat[i]), float(lon[i])) for i in range(len(osmids))}
    rail_rows = _build_rail_rows(
        adjacency=adjacency,
        paths=paths,
        survivor=survivor,
        latlon=latlon,
        elev_by_osmid=elev_by_osmid,
        vertex_lonlatz=vertex_lonlatz,
    )

    survivor_rows = sorted(survivor_row.values())
    survivor_nodes = stations.iloc[survivor_rows].copy()
    survivor_nodes[Schema.STATION_NAME] = [kept_name[int(complex_of[r])] for r in survivor_rows]
    kept_nodes = pd.concat([nodes_df[~is_rail_node], survivor_nodes[NODE_COLS]], ignore_index=True)

    # Re-point STATION access edges from any merged-away platform onto its survivor (bike endpoints, absent
    # from the map, keep their id via fillna). Vectorized map + fillna, no per-row Python.
    access = edges_df[edges_df[Schema.MODE] == Mode.STATION].copy()
    remap = pd.Series(platform_to_survivor, dtype=np.int64)
    for col in (Schema.FROM_NODE, Schema.TO_NODE):
        access[col] = access[col].map(remap).fillna(access[col]).astype(np.int64)
    bike_edges = edges_df[edges_df[Schema.MODE] == Mode.BIKE]
    new_rail = pd.DataFrame(rail_rows, columns=EDGE_COLS)
    kept_edges = pd.concat([bike_edges, access[EDGE_COLS], new_rail], ignore_index=True)
    logger.info(f"consolidate_rail: {len(survivor_nodes)} stations, {len(new_rail)} rail edges (both directions)")
    return kept_nodes, kept_edges


def combine_regions(*, regions_dir: Path, regions: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase 3: offset each region's ids into a global space, dedup the seam, consolidate rail, re-densify.

    Running-offset combine + geometry dedup stitch region borders; consolidate_rail rebuilds the clean
    station↔station rail layer; a prune drops strays; a final remap closes id holes. Returns (nodes, edges).
    """
    node_frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    offset = 0
    for i, region_key in enumerate(regions, start=1):
        nodes_df, edges_df = read_region_tables(region_dir=regions_dir / region_key)
        nodes_df, edges_df = reindex_region(nodes_df=nodes_df, edges_df=edges_df, offset=offset)
        offset += len(nodes_df)
        node_frames.append(nodes_df)
        edge_frames.append(edges_df)
        logger.info(f"combine [{i}/{len(regions)}] read {region_key}: +{len(nodes_df)} nodes → {offset} total")
    if offset >= _MAX_TOTAL_NODES:
        raise ValueError(f"Combined node count {offset} exceeds sanity ceiling {_MAX_TOTAL_NODES} — aborting.")
    nodes_df = pd.concat(node_frames, ignore_index=True)
    edges_df = pd.concat(edge_frames, ignore_index=True)
    logger.info(f"combine: concatenated {len(nodes_df)} nodes / {len(edges_df)} edges; deduping seams …")
    nodes_df, edges_df = dedup_by_geometry(nodes_df=nodes_df, edges_df=edges_df)
    logger.info(f"combine: after dedup {len(nodes_df)} nodes / {len(edges_df)} edges; consolidating rail …")
    # Rebuild the rail layer as ONE clean edge per station pair (drops corrupt connectors + track nodes),
    # AFTER seam dedup so cross-region track is welded, BEFORE prune so it runs on the clean topology.
    nodes_df, edges_df = consolidate_rail(nodes_df=nodes_df, edges_df=edges_df)
    logger.info(f"combine: after rail consolidation {len(nodes_df)} nodes / {len(edges_df)} edges; pruning …")
    # ONE global component prune, AFTER dedup has stitched the region seams: rail → single component,
    # bike → keep every island ≥ MIN_BIKE_COMPONENT_KM. The sole connectivity gate (no separate pass).
    nodes_df, edges_df = prune_components(nodes_df=nodes_df, edges_df=edges_df)
    logger.info(f"combine: after prune {len(nodes_df)} nodes / {len(edges_df)} edges; remapping to dense ids …")
    # Dedup + prune removed nodes, leaving id holes → renumber to dense 0..N-1 (n_nodes==max_id+1).
    nodes_df, edges_df = remap_contiguous(nodes_df=nodes_df, edges_df=edges_df)
    logger.info(f"combine: DONE — {len(nodes_df)} nodes / {len(edges_df)} edges (dense ids 0..N-1)")
    return nodes_df, edges_df


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
    gr = nx.Graph()
    gr.add_edges_from(zip(rail_e["from_node"], rail_e["to_node"], strict=True))
    keep_rail = max(nx.connected_components(gr), key=len) if gr.number_of_nodes() else set()

    # BIKE: every weakly-connected component with total length ≥ threshold.
    gb = nx.Graph()
    gb.add_weighted_edges_from(zip(bike_e["from_node"], bike_e["to_node"], bike_e["length_m"], strict=True))
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
