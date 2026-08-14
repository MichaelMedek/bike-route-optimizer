"""Station extrema: local-max ("top") and local-min ("bottom") rail stations for trip planning.

Reads the clean station↔station rail graph the build baked in (no track weld/watershed at runtime);
classifies extrema by direction-candidacy + key-col prominence over that graph.
"""

import logging
from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from pathlib import Path
from typing import NamedTuple, TypeVar

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from bike_router.core.constants import GeoConfig, GraphConfig, Mode, NodeType, RailConfig, Schema
from bike_router.core.graph_store import NODE_COLS, read_tiles

logger = logging.getLogger(__name__)

# Extrema helpers are node-id-agnostic — they run on the osmid-keyed OR the name-keyed station graph.
_Node = TypeVar("_Node", bound=Hashable)


class StationGraph(NamedTuple):
    """ONE cached read of the clean rail layer, shared by track-graph / line-degrees / extrema."""

    stations_df: pd.DataFrame  # every named rail station (osmid, lat, lon, elevation_m, name)
    graph: dict[str, set[str]]  # station NAME → directly-connected station names
    elev_by_name: dict[str, float]  # station NAME → elevation (m)


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


@lru_cache(maxsize=4)
def station_track_graph(graph_dir: Path) -> StationGraph:
    """ONE cached read → the stations frame + name-keyed line-degree graph + name→elevation.

    The build baked ONE rail edge per adjacent station pair (platforms merged, parallel tracks collapsed);
    a duplicate name (Hochdorf DE/CH, 145 km apart, distinct nodes) keeps its highest-degree node, no merge.
    """
    stations_df = load_stations(graph_dir=graph_dir)
    edges_df = read_tiles(
        directory=graph_dir / GraphConfig.EDGES_SUBDIR,
        columns=[Schema.FROM_NODE, Schema.TO_NODE],
        tiles=None,
        filters=[(Schema.MODE, Schema.FILTER_IN, [Mode.RAIL])],
    )
    name_of = {int(r.osmid): str(r.station_name) for r in stations_df.itertuples(index=False)}
    by_osmid: dict[int, set[int]] = {osmid: set() for osmid in name_of}
    for u, v in zip(edges_df[Schema.FROM_NODE], edges_df[Schema.TO_NODE], strict=True):
        iu, iv = int(u), int(v)
        if iu in by_osmid and iv in by_osmid and iu != iv:
            by_osmid[iu].add(iv)
            by_osmid[iv].add(iu)
    kept: dict[str, int] = {}  # a duplicate name keeps its highest-degree node (its real, unmerged degree)
    for osmid, neighbours in by_osmid.items():
        name = name_of[osmid]
        if name not in kept or len(neighbours) > len(by_osmid[kept[name]]):
            kept[name] = osmid
    kept_ids = set(kept.values())
    graph: dict[str, set[str]] = {}
    for name, osmid in kept.items():
        neighbour_names = {name_of[nb] for nb in by_osmid[osmid] if nb in kept_ids}
        neighbour_names.discard(name)  # drop any self-edge a same-name neighbour would introduce
        graph[name] = neighbour_names
    elev_of = {int(r.osmid): float(r.elevation_m) for r in stations_df.itertuples(index=False)}
    elev_by_name = {name: elev_of[osmid] for name, osmid in kept.items()}
    return StationGraph(stations_df=stations_df, graph=graph, elev_by_name=elev_by_name)


def station_line_degrees(*, graph_dir: Path) -> dict[str, int]:
    """Per-station rail line-degree: station name → number of distinct next-stations along the track.

    The degree IS the size of the station's neighbour set in ``station_track_graph`` (parallel rails,
    switches, and multi-track throats between the same two stations already collapsed to one edge offline).
    """
    return {name: len(neighbours) for name, neighbours in station_track_graph(graph_dir).graph.items()}


def extremum_candidates(
    *, station_graph: dict[_Node, set[_Node]], elev_by_id: dict[_Node, float], want_high: bool
) -> set[_Node]:
    """PASS 1 — direction-only candidacy (no prominence yet).

    A dead-end (1 branch) is a candidate iff its neighbour is on-side (lower for a top, higher for a
    bottom); a junction (≥2 branches) iff ≥EXTREMUM_MIN_NEIGHBORS immediate neighbours are on-side.
    """
    out: set[_Node] = set()
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
    *,
    station_graph: dict[_Node, set[_Node]],
    elev_by_id: dict[_Node, float],
    source: _Node,
    first: _Node,
    want_high: bool,
) -> bool:
    """Whether one branch clears the prominence gate — it rises (top) / falls (bottom) ≥PROMINENCE_M.

    Walks outward from ``first`` following ON-SIDE terrain only (lower for a top, higher for a bottom),
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
    *, station_graph: dict[_Node, set[_Node]], elev_by_id: dict[_Node, float], source: _Node, want_high: bool
) -> bool:
    """PASS 2 — prominence. A dead-end confirms on its one branch, a junction on ≥EXTREMUM_MIN_NEIGHBORS."""
    branches = list(station_graph[source])
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


def key_col_prominence(
    *, station_graph: dict[_Node, set[_Node]], elev_by_id: dict[_Node, float], source: _Node, want_high: bool
) -> float:
    """Topographic key-col prominence of ``source`` as a max (want_high) / min over the station graph.

    Dijkstra on the path's extreme intervening elevation: the lowest saddle to cross to reach HIGHER
    (top) / LOWER (bottom) ground. inf if none exists (a global summit/valley of its component).
    """
    elev_s = elev_by_id[source]
    best = {source: elev_s}
    counter = 0  # tiebreaker so the heap never compares two _Node values
    pq: list[tuple[float, int, _Node]] = [(-elev_s if want_high else elev_s, counter, source)]
    while pq:
        signed, _tie, u = heappop(pq)
        ridge = -signed if want_high else signed
        if (ridge < best[u]) if want_high else (ridge > best[u]):
            continue
        if u != source and ((elev_by_id[u] > elev_s) if want_high else (elev_by_id[u] < elev_s)):
            return abs(ridge - elev_s)  # reached higher/lower ground; prominence = saddle depth
        for v in station_graph[u]:
            nridge = min(ridge, elev_by_id[v]) if want_high else max(ridge, elev_by_id[v])
            improved = (nridge > best.get(v, -np.inf)) if want_high else (nridge < best.get(v, np.inf))
            if improved:
                best[v] = nridge
                counter += 1
                heappush(pq, (-nridge if want_high else nridge, counter, v))
    return float(np.inf)


def extremum_stations(
    *, stations_df: pd.DataFrame, station_graph: dict[int, set[int]], want_high: bool
) -> pd.DataFrame:
    """The confirmed local-max (want_high) / local-min stations — Pass 1 candidacy then Pass 2 prominence.

    A station is kept iff it is a direction-only candidate AND clears the prominence gate. Sorted
    high-first (max) / low-first (min).
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


def station_extrema(*, graph_dir: Path) -> StationExtrema:
    """The 🚞 payload from ONE scan: green local-max + red local-min station markers.

    Dominance + prominence on the clean graph: a top is a strict local peak with key-col prominence
    ≥PEAK_KEYCOL_M, OR a junction (deg≥3) dominating its DOMINANCE_RADIUS_M region by ≥HUB_RISE_M; bottom mirrors.
    """
    stations_df, adjacency, elev_by_name = station_track_graph(graph_dir)
    mpd = GeoConfig.METERS_PER_DEGREE_EQUATOR
    lat = stations_df[Schema.LAT].to_numpy(dtype=np.float64)
    lon = stations_df[Schema.LON].to_numpy(dtype=np.float64)
    names = stations_df[Schema.STATION_NAME].astype(str).to_numpy()
    elev = stations_df[Schema.ELEVATION_M].to_numpy(dtype=np.float64)
    xy = np.column_stack([lon * mpd * np.cos(np.radians(lat)), lat * mpd])
    tree = cKDTree(xy)
    radius = RailConfig.EXTREMA_DOMINANCE_RADIUS_M
    within = tree.query_ball_point(xy, radius)
    top_names: set[str] = set()
    bot_names: set[str] = set()
    for i, name in enumerate(names):
        neighbours = adjacency.get(name, set())
        if not neighbours:
            continue  # isolated (only boundary-clipped stubs) — never an extremum
        here = float(elev[i])
        near_elev = elev[within[i]]
        rise = here - float(near_elev.min())  # metres above the lowest station in the region
        drop = float(near_elev.max()) - here  # metres below the highest station in the region
        n_elev = [elev_by_name[nb] for nb in neighbours]
        degree = len(neighbours)
        top_prom = key_col_prominence(station_graph=adjacency, elev_by_id=elev_by_name, source=name, want_high=True)
        bot_prom = key_col_prominence(station_graph=adjacency, elev_by_id=elev_by_name, source=name, want_high=False)
        is_top = (here >= max(n_elev) and top_prom >= RailConfig.EXTREMA_PEAK_KEYCOL_M) or (
            degree >= 3 and rise >= RailConfig.EXTREMA_HUB_RISE_M
        )
        is_bot = (here <= min(n_elev) and bot_prom >= RailConfig.EXTREMA_VALLEY_KEYCOL_M) or (
            degree >= 3 and drop >= RailConfig.EXTREMA_HUB_DROP_M
        )
        if is_top and not is_bot:
            top_names.add(name)
        elif is_bot and not is_top:
            bot_names.add(name)
        elif is_top and is_bot:
            (top_names if rise >= drop else bot_names).add(name)
    # One marker per extremum NAME (drop duplicate-name platform rows so counts match the name graph).
    maxima_df = (
        stations_df[stations_df[Schema.STATION_NAME].isin(top_names)]
        .drop_duplicates(subset=Schema.STATION_NAME)
        .reset_index(drop=True)
    )
    minima_df = (
        stations_df[stations_df[Schema.STATION_NAME].isin(bot_names)]
        .drop_duplicates(subset=Schema.STATION_NAME)
        .reset_index(drop=True)
    )
    logger.info(f"Station-extrema scan: {len(maxima_df)} tops, {len(minima_df)} bottoms")
    return StationExtrema(
        maxima=station_markers(stations_df=maxima_df, want_high=True),
        minima=station_markers(stations_df=minima_df, want_high=False),
    )
