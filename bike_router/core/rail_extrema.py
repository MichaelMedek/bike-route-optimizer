"""Station extrema: local-max ("top") and local-min ("bottom") rail stations for trip planning.

Two passes: (1) direction-only candidacy — dead-end always, else ≥MIN_NEIGHBORS rail branches going the
same way; (2) prominence — each branch walks on-side terrain until it rises/falls ≥PROMINENCE_M.
"""

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from bike_router.core.constants import GeoConfig, GraphConfig, Mode, NodeType, RailConfig, Schema
from bike_router.core.graph_store import NODE_COLS, read_tiles

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StationExtrema:
    """The 🚞 marker payload from ONE graph scan: green local-max + red local-min stations."""

    maxima: list[tuple[float, float, float, str]]
    minima: list[tuple[float, float, float, str]]


def load_stations(*, graph_dir: Path) -> pd.DataFrame:
    """Every named rail station across the coverage area (osmid, lat, lon, elevation_m, name).

    Pushes the ``node_type == rail`` filter into pyarrow so only rail rows are read (not all ~8M nodes).
    """
    nodes_df = read_tiles(
        directory=graph_dir / GraphConfig.NODES_SUBDIR,
        columns=NODE_COLS,
        tiles=None,
        filters=[(Schema.NODE_TYPE, "==", NodeType.RAIL)],
    )
    stations = nodes_df[nodes_df["station_name"].notna()].reset_index(drop=True)
    assert not stations.empty, "no station found"
    return stations


def station_rail_neighbors(*, edges_df: pd.DataFrame, station_ids: set[int]) -> dict[int, set[int]]:
    """Each station → its next-stop stations up/down the line (the station-level graph).

    Vectorized graph-Voronoi: one multi-source ``dijkstra(min_only, unweighted)`` labels every node with
    its nearest station, then station-level edges are the unique label pairs of rail edges crossing cells.
    """
    from_ids = edges_df["from_node"].to_numpy(dtype=np.int64)
    to_ids = edges_df["to_node"].to_numpy(dtype=np.int64)
    # Dense-index every node, build the undirected CSR (both directions), then flood from station rows.
    codes, uniques = pd.factorize(np.concatenate([from_ids, to_ids]))
    n = len(uniques)
    fu, tu = codes[: len(from_ids)], codes[len(from_ids) :]
    rows = np.concatenate([fu, tu])
    cols = np.concatenate([tu, fu])
    graph = csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n, n))
    id_to_row = {int(osmid): i for i, osmid in enumerate(uniques)}
    seed_rows = np.array([id_to_row[s] for s in station_ids if s in id_to_row], dtype=np.int64)
    _dist, _pred, sources = dijkstra(
        graph, directed=False, indices=seed_rows, min_only=True, unweighted=True, return_predecessors=True
    )
    label = uniques[sources]  # each node → its nearest station's osmid (its Voronoi cell)
    # Rail edges whose two endpoints fall in DIFFERENT cells connect those two stations (next stops).
    su, sv = label[fu], label[tu]
    cross = su != sv
    neighbors: dict[int, set[int]] = {s: set() for s in station_ids}
    for a, b in np.unique(np.sort(np.stack([su[cross], sv[cross]], axis=1), axis=1), axis=0):
        neighbors[int(a)].add(int(b))
        neighbors[int(b)].add(int(a))
    return neighbors


def extremum_candidates(
    *, station_graph: dict[int, set[int]], elev_by_id: dict[int, float], want_high: bool
) -> set[int]:
    """PASS 1 — direction-only candidacy (no prominence yet).

    A dead-end (1 branch) is a candidate iff its neighbour is on-side (lower for a top, higher for a
    bottom); a junction (≥2 branches) iff ≥EXTREMUM_MIN_NEIGHBORS immediate neighbours are on-side.
    """
    out: set[int] = set()
    for source, neighbors in station_graph.items():
        if not neighbors:
            continue
        elev_s = elev_by_id[source]
        on_side = sum(1 for n in neighbors if (elev_by_id[n] < elev_s if want_high else elev_by_id[n] > elev_s))
        if len(neighbors) == 1:
            if on_side == 1:
                out.add(source)
        elif on_side >= RailConfig.EXTREMUM_MIN_NEIGHBORS:
            out.add(source)
    return out


def branch_confirms(
    *, station_graph: dict[int, set[int]], elev_by_id: dict[int, float], source: int, first: int, want_high: bool
) -> bool:
    """Whether one branch clears the prominence gate — it rises (top) / falls (bottom) ≥PROMINENCE_M.

    Walks outward from ``first`` following ON-SIDE terrain only (higher for a bottom, lower for a top),
    stopping a path once it crests back toward ``source``; confirms if any path reaches ≥prominence away.
    """
    prom = RailConfig.EXTREMA_STATION_PROMINENCE_M
    elev_s = elev_by_id[source]
    visited = {source}
    queue = deque([first])
    while queue:
        cur = queue.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        diff = elev_s - elev_by_id[cur] if want_high else elev_by_id[cur] - elev_s
        if diff >= prom:
            return True  # climbed/dropped far enough along this branch
        if diff <= 0:
            continue  # terrain turned back off-side (a crest/trough) — this path can't confirm
        queue.extend(station_graph[cur])
    return False


def is_confirmed_extremum(
    *, station_graph: dict[int, set[int]], elev_by_id: dict[int, float], source: int, want_high: bool
) -> bool:
    """PASS 2 — prominence. A dead-end confirms on its one branch, a junction on ≥EXTREMUM_MIN_NEIGHBORS."""
    branches = sorted(station_graph[source])
    if not branches:
        return False
    confirmed = sum(
        1
        for first in branches
        if branch_confirms(
            station_graph=station_graph, elev_by_id=elev_by_id, source=source, first=first, want_high=want_high
        )
    )
    if len(branches) == 1:
        return confirmed == 1
    return confirmed >= RailConfig.EXTREMUM_MIN_NEIGHBORS


def extremum_stations(
    *, stations_df: pd.DataFrame, station_graph: dict[int, set[int]], want_high: bool
) -> pd.DataFrame:
    """The confirmed local-max (want_high) / local-min stations — Pass 1 candidacy then Pass 2 prominence.

    A station is kept iff it is a direction-only candidate AND ≥MIN_NEIGHBORS branches clear the prominence
    gate (a dead-end on its one branch). Sorted high-first (max) / low-first (min).
    """
    elev_by_id = {int(row.osmid): float(row.elevation_m) for row in stations_df.itertuples(index=False)}
    tagged = extremum_candidates(station_graph=station_graph, elev_by_id=elev_by_id, want_high=want_high)
    keep = [
        i
        for i, row in enumerate(stations_df.itertuples(index=False))
        if int(row.osmid) in tagged
        and is_confirmed_extremum(
            station_graph=station_graph, elev_by_id=elev_by_id, source=int(row.osmid), want_high=want_high
        )
    ]
    return stations_df.iloc[keep].sort_values(Schema.ELEVATION_M, ascending=not want_high).reset_index(drop=True)


def station_markers(*, stations_df: pd.DataFrame, want_high: bool) -> list[tuple[float, float, float, str]]:
    """(lat, lon, elevation_m, name) markers, nudged north (tops) / south (bottoms) so a both-station splits.

    A station that is both a top and a bottom gets a marker in each set at the same point; the ±latitude
    offset keeps them separately clickable (top north, bottom south) — see EXTREMUM_MARKER_OFFSET_M.
    """
    d_lat = RailConfig.EXTREMUM_MARKER_OFFSET_M / GeoConfig.METERS_PER_DEGREE_EQUATOR
    offset = d_lat if want_high else -d_lat
    return [
        (float(row.lat) + offset, float(row.lon), float(row.elevation_m), str(row.station_name))
        for row in stations_df.itertuples(index=False)
    ]


def station_extrema(*, graph_dir: Path) -> StationExtrema:
    """The 🚞 payload from ONE scan: green local-max + red local-min station markers."""
    stations_df = load_stations(graph_dir=graph_dir)
    edges_lite = read_tiles(
        directory=graph_dir / GraphConfig.EDGES_SUBDIR,
        columns=[Schema.FROM_NODE, Schema.TO_NODE],
        tiles=None,
        filters=[(Schema.MODE, Schema.FILTER_IN, [Mode.RAIL])],
    )
    station_ids = set(stations_df["osmid"].astype(int))
    station_graph = station_rail_neighbors(edges_df=edges_lite, station_ids=station_ids)
    maxima_df = extremum_stations(stations_df=stations_df, station_graph=station_graph, want_high=True)
    minima_df = extremum_stations(stations_df=stations_df, station_graph=station_graph, want_high=False)
    logger.info(f"Station-extrema scan: {len(maxima_df)} tops, {len(minima_df)} bottoms")
    return StationExtrema(
        maxima=station_markers(stations_df=maxima_df, want_high=True),
        minima=station_markers(stations_df=minima_df, want_high=False),
    )
