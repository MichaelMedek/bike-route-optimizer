"""Unified route-track builder — the single source of per-point time & elevation.

Walks the A* node path once, computing per-edge length, grade and adaptive speed
while accumulating ride time. GPX, stats, and elevation profile all derive from this
one structure, so total time and GPX end-timestamp agree by construction.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import networkx as nx

from bike_router.constants import CostConfig, GpxConfig, Mode, RailConfig, SpeedConfig
from bike_router.cost import road_tier, surface_tier
from bike_router.geo import haversine_distance_m
from bike_router.speed import effective_speed_kmh, kmh_to_ms


@dataclass(frozen=True)
class TrackPoint:
    """One point along the route: position, elevation, cumulative time, and the mode,
    condition, and speed of the edge arriving at it (start point takes the first edge's).
    """

    lat: float
    lon: float
    elevation_m: float
    elapsed_s: float
    mode: str
    surface_bad: bool  # pedalled segment on an unpaved/rough surface (tier != 0); False for rail
    road_bad: bool  # pedalled segment on a main road (road tier != 0); False for rail
    grade: float  # signed rise/run of the arriving edge (+ uphill, − downhill); 0 for the start
    speed_kmh: float  # segment speed (bike: adaptive; rail: RAIL_SPEED_KMH) — drives ribbon width


@dataclass(frozen=True)
class RouteStats:
    """Rolled-up distance / duration / climb for a portion of the route.

    The per-field format strings live HERE (single source): the CLI, the Streamlit
    metrics, and the PNG overlay all render via these properties so they never drift.
    """

    distance_km: float
    duration_min: float
    ascent_m: float
    descent_m: float

    @property
    def distance_str(self) -> str:
        return f"{self.distance_km:.1f} km"

    @property
    def duration_str(self) -> str:
        return f"{self.duration_min:.0f} min"

    @property
    def ascent_str(self) -> str:
        return f"+{self.ascent_m:.0f} m"

    @property
    def descent_str(self) -> str:
        return f"−{self.descent_m:.0f} m"  # unicode minus U+2212, one style everywhere

    @property
    def oneline(self) -> str:
        """Single-line summary: ``7.0 km · 24 min · +218 m / −26 m`` (CLI + PNG)."""
        return f"{self.distance_str} · {self.duration_str} · {self.ascent_str} / {self.descent_str}"

    def metric_pairs(self, *, duration_label: str) -> tuple[tuple[str, str], ...]:
        """(label, value) pairs for the four stat widgets (Streamlit st.metric rows)."""
        return (
            ("Distance", self.distance_str),
            (duration_label, self.duration_str),
            ("Ascent", self.ascent_str),
            ("Descent", self.descent_str),
        )


@dataclass(frozen=True)
class Track:
    """The full traversed route: ordered points + stats split bike-only vs bike+train.

    ``bike`` covers only the pedalled legs (what the rider physically cycles); ``total``
    covers the whole journey including train rides and boarding waits.
    """

    points: list[TrackPoint]
    bike: RouteStats
    total: RouteStats


def cheapest_edge(edges: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """The parallel edge A* would traverse (lowest stored cost)."""
    best_key = min(edges, key=lambda key: float(edges[key][CostConfig.EDGE_COST]))
    return edges[best_key]


def climb_totals(deltas: list[float]) -> tuple[float, float]:
    """(ascent, descent) in metres from per-edge Δelevations: gross up- vs down-sum.

    Ascent sums the positive deltas, descent the magnitude of the negative ones (NOT the
    net change). One source for both the whole-journey and the bike-only climb tallies.
    """
    ascent = sum(d for d in deltas if d > 0)
    descent = sum(-d for d in deltas if d < 0)
    return ascent, descent


def iter_route_edges(graph: nx.MultiDiGraph, node_path: list[int]) -> Iterator[tuple[int, int, dict[str, Any]]]:
    """Yield ``(node_a, node_b, cheapest_edge_data)`` for each hop on the A* path."""
    for node_a, node_b in zip(node_path[:-1], node_path[1:], strict=True):
        yield node_a, node_b, cheapest_edge(edges=graph.get_edge_data(node_a, node_b))


def edge_condition_speed(
    *, data: dict[str, Any], elev_source: float, elev_target: float, length_m: float
) -> tuple[bool, bool, float]:
    """(surface_bad, road_bad, speed_kmh) for one edge — single source for timing + colour.

    bike: surface_bad if surface tier != 0; road_bad if a main road; speed from the adaptive
    model. rail / station: never bad, fixed RAIL_SPEED_KMH / walking pace.
    """
    mode = data["mode"]
    if mode == Mode.BIKE:
        s_tier = surface_tier(surface=data.get("surface"))
        grade = (elev_target - elev_source) / length_m if length_m > 0 else 0.0
        speed_kmh = effective_speed_kmh(surface_tier=s_tier, grade=grade)
        return s_tier != 0, road_tier(highway=data.get("highway")) != 0, speed_kmh
    elif mode == Mode.RAIL:
        return False, False, RailConfig.RAIL_SPEED_KMH
    elif mode == Mode.STATION:
        return False, False, SpeedConfig.WALK_KMH  # short walk to the platform
    else:
        raise AssertionError(f"unknown edge mode: {mode!r}")


def classify_condition(*, mode: str, surface_bad: bool, road_bad: bool) -> str:
    """Canonical condition label for a route segment — the SINGLE branch point."""
    if mode == Mode.RAIL:
        return "train"
    elif surface_bad and road_bad:
        return "main road + unpaved"
    elif road_bad and not surface_bad:
        return "main road"
    elif surface_bad and not road_bad:
        return "unpaved"
    elif not surface_bad and not road_bad:
        return "good"
    else:
        raise AssertionError(f"unclassified segment: mode={mode!r} surface_bad={surface_bad} road_bad={road_bad}")


def build_track(graph: nx.MultiDiGraph, node_path: list[int]) -> Track:
    """Build the full route track with adaptive-speed timing from a node path.

    Bike edges use the surface/grade speed model and count toward ascent/descent; rail
    edges ride at RAIL_SPEED_KMH; each station edge adds half of BOARDING_WAIT_S, so board
    + alight sum to one full wait (mirrors the cost split). All leg times are DERIVED.
    """
    assert len(node_path) >= 2, "route must have >= 2 nodes"

    first = graph.nodes[node_path[0]]
    first_data = cheapest_edge(edges=graph.get_edge_data(node_path[0], node_path[1]))
    first_surface_bad, first_road_bad, first_speed = edge_condition_speed(
        data=first_data,
        elev_source=float(first["elevation"]),
        elev_target=float(graph.nodes[node_path[1]]["elevation"]),
        length_m=float(first_data["length"]),
    )
    first_len = float(first_data["length"])
    first_grade = (float(graph.nodes[node_path[1]]["elevation"]) - float(first["elevation"])) / first_len
    points = [
        TrackPoint(
            lat=first["y"],
            lon=first["x"],
            elevation_m=float(first["elevation"]),
            elapsed_s=0.0,
            mode=str(first_data["mode"]),
            surface_bad=first_surface_bad,
            road_bad=first_road_bad,
            grade=first_grade,
            speed_kmh=first_speed,
        )
    ]
    total_m = total_s = 0.0
    total_deltas: list[float] = []  # Δelevation of EVERY edge (whole-journey climb)
    bike_deltas: list[float] = []  # Δelevation of pedalled edges only
    bike_m = bike_s = 0.0  # only bike legs feed the avg-speed assert (rail is far faster)
    rail_speed_ms = kmh_to_ms(kmh=RailConfig.RAIL_SPEED_KMH)

    for node_a, node_b, data in iter_route_edges(graph=graph, node_path=node_path):
        length_m = float(data["length"])
        elev_a = float(graph.nodes[node_a]["elevation"])
        elev_b = float(graph.nodes[node_b]["elevation"])
        delta = elev_b - elev_a
        surface_bad, road_bad, speed_kmh = edge_condition_speed(
            data=data, elev_source=elev_a, elev_target=elev_b, length_m=length_m
        )
        total_deltas.append(delta)  # every edge feeds the whole-journey climb (same scope as total_m)

        if data["mode"] == Mode.BIKE:
            bike_deltas.append(delta)
            speed_ms = kmh_to_ms(kmh=speed_kmh)
            leg_s = length_m / speed_ms
            bike_m += length_m
            bike_s += leg_s
        elif data["mode"] == Mode.RAIL:
            leg_s = length_m / rail_speed_ms  # train ride time, derived from length
        elif data["mode"] == Mode.STATION:  # half the wait per station edge: board + alight = full wait
            leg_s = 0.5 * RailConfig.BOARDING_WAIT_S
        else:
            raise AssertionError(f"unknown edge mode: {data['mode']!r}")
        total_s += leg_s
        total_m += length_m

        target = graph.nodes[node_b]
        grade = delta / length_m
        points.append(
            TrackPoint(
                lat=target["y"],
                lon=target["x"],
                elevation_m=elev_b,
                elapsed_s=total_s,
                mode=str(data["mode"]),
                surface_bad=surface_bad,
                road_bad=road_bad,
                grade=grade,
                speed_kmh=speed_kmh,
            )
        )

    assert total_m > 0, "route distance must be positive"
    assert total_s > 0, "route duration must be positive"
    # sanity: average BIKE speed must sit between the walking floor and the best base
    # speed (rail legs are excluded — 80 km/h would trip it).
    if bike_s > 0:
        avg_kmh = (bike_m / GpxConfig.METERS_PER_KM) / (bike_s / GpxConfig.SECONDS_PER_HOUR)
        assert SpeedConfig.WALK_KMH - 1e-9 <= avg_kmh <= max(SpeedConfig.BASE_KMH_BY_TIER.values()) + 1e-9, (
            f"implausible average speed {avg_kmh:.1f} km/h"
        )
    # bike stats = pedalled legs only; total = whole journey (bike + rail climb), matching total_m.
    bike_ascent, bike_descent = climb_totals(deltas=bike_deltas)
    total_ascent, total_descent = climb_totals(deltas=total_deltas)
    bike_stats = RouteStats(
        distance_km=bike_m / GpxConfig.METERS_PER_KM,
        duration_min=bike_s / GpxConfig.SECONDS_PER_HOUR * GpxConfig.MINUTES_PER_HOUR,
        ascent_m=bike_ascent,
        descent_m=bike_descent,
    )
    total_stats = RouteStats(
        distance_km=total_m / GpxConfig.METERS_PER_KM,
        duration_min=total_s / GpxConfig.SECONDS_PER_HOUR * GpxConfig.MINUTES_PER_HOUR,
        ascent_m=total_ascent,  # whole journey, incl. the climb the train covers
        descent_m=total_descent,
    )
    return Track(points=points, bike=bike_stats, total=total_stats)


def densify_track(graph: nx.MultiDiGraph, node_path: list[int], track: Track) -> Track:
    """Expand the node-level track into the full 3D road polyline (no DEM at inference).

    Walks each edge's baked 3D geometry so the profile follows the true road/track;
    only station access-links contribute a straight segment. Each leg's time is spread by
    along-edge distance and ascent/descent are recomputed from the baked vertex elevations.
    """
    assert len(node_path) >= 2, "route must have >= 2 nodes to densify"
    assert len(track.points) == len(node_path), "track points must align with node path"
    out: list[TrackPoint] = []

    for index, (node_a, node_b, data) in enumerate(iter_route_edges(graph=graph, node_path=node_path)):
        mode = str(data["mode"])
        # Condition + speed are per-edge; the arriving point (index+1) carries the leg's.
        leg_point = track.points[index + 1]
        leg_surface_bad, leg_road_bad, leg_speed = leg_point.surface_bad, leg_point.road_bad, leg_point.speed_kmh
        leg_grade = leg_point.grade  # the leg's node-level gradient, shared by its dense vertices
        verts = edge_vertices_3d(graph=graph, node_a=node_a, node_b=node_b, data=data)
        t_start, t_end = track.points[index].elapsed_s, track.points[index + 1].elapsed_s
        seg_lengths = [
            haversine_distance_m(lat_a=verts[i][1], lon_a=verts[i][0], lat_b=verts[i + 1][1], lon_b=verts[i + 1][0])
            for i in range(len(verts) - 1)
        ]
        total = sum(seg_lengths) or 1.0
        cum = 0.0
        last_leg = index == len(node_path) - 2
        stop = len(verts) if last_leg else len(verts) - 1  # avoid duplicating shared node vertex
        for i in range(stop):
            lon, lat, elev = verts[i]
            out.append(
                TrackPoint(
                    lat=lat,
                    lon=lon,
                    elevation_m=elev,
                    elapsed_s=t_start + (t_end - t_start) * (cum / total),
                    mode=mode,
                    surface_bad=leg_surface_bad,
                    road_bad=leg_road_bad,
                    grade=leg_grade,
                    speed_kmh=leg_speed,
                )
            )
            if i < len(seg_lengths):
                cum += seg_lengths[i]

    # Stats are unchanged by densification (same legs) — carry both groups through.
    return Track(points=out, bike=track.bike, total=track.total)


def edge_vertices_3d(
    graph: nx.MultiDiGraph, node_a: int, node_b: int, data: dict[str, Any]
) -> list[tuple[float, float, float]]:
    """(lon, lat, elev) vertices of edge a→b from its baked 3D geometry.

    Bike and rail edges both carry a baked 3D LineString; only station access-links
    have no geometry, so fall back to a straight segment at the two node elevations.
    """
    geom = data.get("geometry")
    ea, eb = float(graph.nodes[node_a]["elevation"]), float(graph.nodes[node_b]["elevation"])
    if geom is None:
        return [
            (graph.nodes[node_a]["x"], graph.nodes[node_a]["y"], ea),
            (graph.nodes[node_b]["x"], graph.nodes[node_b]["y"], eb),
        ]
    coords = [(c[0], c[1], c[2] if len(c) > 2 else ea) for c in geom.coords]
    # Orient a→b by matching the first vertex to node_a's coords.
    ax, ay = graph.nodes[node_a]["x"], graph.nodes[node_a]["y"]
    if abs(coords[0][0] - ax) + abs(coords[0][1] - ay) > abs(coords[-1][0] - ax) + abs(coords[-1][1] - ay):
        coords = coords[::-1]
    return coords
