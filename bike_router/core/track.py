"""Unified route-track builder — the single source of per-point time, elevation & colour.

Walks the route's ordered edge list once, computing per-edge length, grade and adaptive speed while
accumulating time. GPX, stats, profile, and both colour scales all derive from this one structure.
"""

from dataclasses import dataclass

import numpy as np

from bike_router.core.constants import Condition, GpxConfig, Grade, GradeConfig, Mode, Palette, RailConfig, SpeedConfig
from bike_router.core.cost import road_tier, surface_tier, surface_weight
from bike_router.core.geo import haversine_vec
from bike_router.core.route_path import RouteEdge, RouteNode, RoutePath
from bike_router.core.speed import effective_speed_kmh, kmh_to_ms


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
    unreliable_elev: bool  # arriving edge's displayed elevation deviates far from baked terrain


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


def cumulative_km(points: "list[TrackPoint]") -> "np.ndarray":
    """Cumulative great-circle distance (km) at each track point — the ONE distance axis.

    Shared by the elevation profile (x-axis) and marker projection so both read the same
    km. Vectorized haversine over consecutive points; point 0 is at 0 km.
    """
    lats = np.array([p.lat for p in points], dtype=np.float64)
    lons = np.array([p.lon for p in points], dtype=np.float64)
    step_km = haversine_vec(lat_a=lats[:-1], lon_a=lons[:-1], lat_b=lats[1:], lon_b=lons[1:]) / 1000.0
    return np.concatenate(([0.0], np.cumsum(step_km)))


def project_markers_onto_track(
    *, track: "Track", markers: list[tuple[float, float, str]]
) -> list[tuple[float, float, str]]:
    """Place each (lat, lon, label) marker on the route at its nearest track point.

    Returns (distance_km, elevation_m, label) so the same named markers the map shows appear
    on the elevation profile at the right position + height. Nearest by great-circle distance.
    """
    dists = cumulative_km(points=track.points)
    plats = np.array([p.lat for p in track.points], dtype=np.float64)
    plons = np.array([p.lon for p in track.points], dtype=np.float64)
    placed: list[tuple[float, float, str]] = []
    for lat, lon, label in markers:
        idx = int(haversine_vec(lat_a=lat, lon_a=lon, lat_b=plats, lon_b=plons).argmin())
        placed.append((float(dists[idx]), track.points[idx].elevation_m, label))
    return placed


def climb_totals(deltas: list[float]) -> tuple[float, float]:
    """(ascent, descent) in metres from per-edge Δelevations: gross up- vs down-sum.

    Ascent sums the positive deltas, descent the magnitude of the negative ones (NOT the
    net change). One source for both the whole-journey and the bike-only climb tallies.
    """
    ascent = sum(d for d in deltas if d > 0)
    descent = sum(-d for d in deltas if d < 0)
    return ascent, descent


def edge_grade(*, elev_source: float, elev_target: float, length_m: float) -> float:
    """Signed rise/run of an edge (+ uphill, − downhill) — the ONE grade formula.

    length_m is a baked graph invariant (> 0); a zero would divide-raise, surfacing the bug.
    """
    return (elev_target - elev_source) / length_m


def edge_condition_speed(*, edge: RouteEdge, elev_source: float, elev_target: float) -> tuple[bool, bool, float]:
    """(surface_bad, road_bad, speed_kmh) for one edge — single source for timing + colour.

    bike: surface_bad if surface tier != 0; road_bad if a main road; speed from the adaptive
    model. rail / station: never bad, fixed RAIL_SPEED_KMH / walking pace.
    """
    if edge.mode == Mode.BIKE:
        # colour reads the discrete tier (paved vs unpaved / quiet vs main); speed reads the continuous weight.
        s_tier = surface_tier(surface=edge.surface)
        grade = edge_grade(elev_source=elev_source, elev_target=elev_target, length_m=edge.length_m)
        speed_kmh = effective_speed_kmh(surface_weight=surface_weight(surface=edge.surface), grade=grade)
        return s_tier != 0, road_tier(highway=edge.highway) != 0, speed_kmh
    elif edge.mode == Mode.RAIL:
        return False, False, RailConfig.RAIL_SPEED_KMH
    elif edge.mode == Mode.STATION:
        return False, False, SpeedConfig.WALK_KMH  # short walk to the platform
    else:
        raise AssertionError(f"unknown edge mode: {edge.mode!r}")


def classify_condition(*, mode: str, surface_bad: bool, road_bad: bool) -> str:
    """Canonical road-QUALITY label for a route segment — the SINGLE branch point."""
    if mode == Mode.RAIL:
        return Condition.TRAIN
    elif surface_bad and road_bad:
        return Condition.MAIN_ROAD_UNPAVED
    elif road_bad and not surface_bad:
        return Condition.MAIN_ROAD
    elif surface_bad and not road_bad:
        return Condition.UNPAVED
    elif not surface_bad and not road_bad:
        return Condition.GOOD
    else:
        raise AssertionError(f"unclassified segment: mode={mode!r} surface_bad={surface_bad} road_bad={road_bad}")


def classify_grade(*, mode: str, grade: float) -> str:
    """Canonical road-GRADE label for a segment — the SINGLE grade branch point.

    A train keeps its own "train" label (purple, no rider-felt grade). Bike/station: uphill at
    or above +MARGIN, downhill at or below −MARGIN, else flat (so only |grade| < MARGIN is flat).
    """
    if mode == Mode.RAIL:
        return Grade.TRAIN
    elif grade >= GradeConfig.MARGIN:
        return Grade.UPHILL
    elif grade <= -GradeConfig.MARGIN:
        return Grade.DOWNHILL
    else:
        return Grade.FLAT


def segment_color(*, mode: str, surface_bad: bool, road_bad: bool) -> list[int]:
    """RGB on the road-QUALITY scale — the single source both the 3D ribbon and PNG use.

    Delegates the branching to classify_condition (one source) and looks the colour up in
    Palette.CONDITION_COLORS, so colour and legend label can never disagree.
    """
    condition = classify_condition(mode=mode, surface_bad=surface_bad, road_bad=road_bad)
    return list(Palette.hex_to_rgb(hex_color=Palette.CONDITION_COLORS[condition]))


def grade_color(*, mode: str, grade: float) -> list[int]:
    """RGB on the road-GRADE scale (flat blue / uphill red / downhill green) — one source."""
    return list(Palette.hex_to_rgb(hex_color=Palette.GRADE_COLORS[classify_grade(mode=mode, grade=grade)]))


def _track_point(
    *, at: RouteNode, edge: RouteEdge, elev_from: float, elev_to: float, elapsed_s: float, unreliable: bool
) -> TrackPoint:
    """A TrackPoint at ``at``, carrying ``edge``'s condition/grade/speed — the ONE point builder.

    ``elev_from``/``elev_to`` are the edge's elevations in TRAVEL direction (node_a → node_b), so
    grade + speed reflect the direction ridden, not which endpoint the point sits at.
    """
    surface_bad, road_bad, speed_kmh = edge_condition_speed(edge=edge, elev_source=elev_from, elev_target=elev_to)
    # Invariant: ONLY bike edges may be flagged unreliable (trains legitimately tunnel/bridge).
    assert not unreliable or edge.mode == Mode.BIKE, f"only bike edges may be unreliable, got {edge.mode!r}"
    return TrackPoint(
        lat=at.lat,
        lon=at.lon,
        elevation_m=at.elevation_m,
        elapsed_s=elapsed_s,
        mode=edge.mode,
        surface_bad=surface_bad,
        road_bad=road_bad,
        grade=edge_grade(elev_source=elev_from, elev_target=elev_to, length_m=edge.length_m),
        speed_kmh=speed_kmh,
        unreliable_elev=unreliable,
    )


def build_track(route: RoutePath) -> Track:
    """Build the full route track with adaptive-speed timing from an ordered edge list.

    Bike edges use the surface/grade speed model and count toward ascent/descent; rail rides at
    RAIL_SPEED_KMH; each station edge adds half of BOARDING_WAIT_S (board + alight = one wait).
    """
    first_node, second_node, first_edge = next(route.iter_edges())
    # Start point: at the first node, oriented in the first edge's travel direction (a→b).
    points = [
        _track_point(
            at=first_node,
            edge=first_edge,
            elev_from=first_node.elevation_m,
            elev_to=second_node.elevation_m,
            elapsed_s=0.0,
            unreliable=edge_display_unreliable(node_a=first_node, node_b=second_node, edge=first_edge),
        )
    ]
    total_m = total_s = 0.0
    total_deltas: list[float] = []  # Δelevation of EVERY edge (whole-journey climb)
    bike_deltas: list[float] = []  # Δelevation of pedalled edges only
    bike_m = bike_s = 0.0  # only bike legs feed the avg-speed assert (rail is far faster)
    rail_speed_ms = kmh_to_ms(kmh=RailConfig.RAIL_SPEED_KMH)

    for node_a, node_b, edge in route.iter_edges():
        length_m = edge.length_m
        delta = node_b.elevation_m - node_a.elevation_m
        total_deltas.append(delta)  # every edge feeds the whole-journey climb (same scope as total_m)

        if edge.mode == Mode.BIKE:
            bike_deltas.append(delta)
            _s, _r, speed_kmh = edge_condition_speed(
                edge=edge, elev_source=node_a.elevation_m, elev_target=node_b.elevation_m
            )
            leg_s = length_m / kmh_to_ms(kmh=speed_kmh)
            bike_m += length_m
            bike_s += leg_s
        elif edge.mode == Mode.RAIL:
            leg_s = length_m / rail_speed_ms  # train ride time, derived from length
        elif edge.mode == Mode.STATION:  # half the wait per station edge: board + alight = full wait
            leg_s = 0.5 * RailConfig.BOARDING_WAIT_S
        else:
            raise AssertionError(f"unknown edge mode: {edge.mode!r}")
        total_s += leg_s
        total_m += length_m
        # The arriving point sits at node_b but carries THIS edge's grade/speed in its travel
        # direction (node_a → node_b), so a climb reads as uphill, not as the reverse descent.
        points.append(
            _track_point(
                at=node_b,
                edge=edge,
                elev_from=node_a.elevation_m,
                elev_to=node_b.elevation_m,
                elapsed_s=total_s,
                unreliable=edge_display_unreliable(node_a=node_a, node_b=node_b, edge=edge),
            )
        )

    assert total_m > 0, "route distance must be positive"
    assert total_s > 0, "route duration must be positive"
    # sanity: average BIKE speed must sit between the walking floor and the best (paved) base
    # speed (rail legs are excluded — 80 km/h would trip it).
    if bike_s > 0:
        avg_kmh = (bike_m / GpxConfig.METERS_PER_KM) / (bike_s / GpxConfig.SECONDS_PER_HOUR)
        assert SpeedConfig.WALK_KMH - 1e-9 <= avg_kmh <= SpeedConfig.BASE_KMH_AT_WEIGHT0 + 1e-9, (
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


def edge_vertices_3d(*, node_a: RouteNode, node_b: RouteNode, edge: RouteEdge) -> list[tuple[float, float, float]]:
    """(lon, lat, elev) vertices of an edge: REAL 2D polyline, elevation LINEAR node-to-node.

    z interpolates between the two node elevations by along-edge distance (the single elevation
    source the optimiser + stats use); a geometry-less hop (rail/station) is a straight segment.
    """
    ea, eb = node_a.elevation_m, node_b.elevation_m
    if edge.geometry is None:
        return [(node_a.lon, node_a.lat, ea), (node_b.lon, node_b.lat, eb)]
    xy = np.asarray(edge.geometry, dtype=np.float64)  # already oriented a→b, 2D lon/lat
    seg = haversine_vec(lat_a=xy[:-1, 1], lon_a=xy[:-1, 0], lat_b=xy[1:, 1], lon_b=xy[1:, 0])
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    frac = cum / (cum[-1] or 1.0)
    z = ea + (eb - ea) * frac
    return [(float(x), float(y), float(zi)) for (x, y), zi in zip(xy, z, strict=True)]


def edge_elevation_deviation_m(*, node_a: RouteNode, node_b: RouteNode, edge: RouteEdge) -> float:
    """Max |displayed − baked| elevation (m) along an edge — how far the linear display sinks/rises.

    The display interpolates elevation linearly between the two node heights; the graph baked the
    REAL terrain per vertex in ``edge.geometry_z``. Returns the largest gap; 0 when no baked z.
    """
    if edge.geometry_z is None or edge.geometry is None:
        return 0.0  # straight hop (no baked polyline) → nothing to deviate from; documented dual-state
    ea, eb = node_a.elevation_m, node_b.elevation_m
    xy = np.asarray(edge.geometry, dtype=np.float64)
    seg = haversine_vec(lat_a=xy[:-1, 1], lon_a=xy[:-1, 0], lat_b=xy[1:, 1], lon_b=xy[1:, 0])
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    linear_z = ea + (eb - ea) * (cum / (cum[-1] or 1.0))
    baked = np.asarray(edge.geometry_z, dtype=np.float64)
    gaps = np.abs(linear_z - baked)
    finite = gaps[np.isfinite(gaps)]
    return float(finite.max()) if finite.size else 0.0


def edge_display_unreliable(*, node_a: RouteNode, node_b: RouteNode, edge: RouteEdge) -> bool:
    """True iff a BIKE edge's displayed elevation deviates beyond GradeConfig.ELEVATION_DEVIATION_WARN_M.

    Only bike edges: trains legitimately tunnel/bridge, so a rail edge's linear-vs-baked gap is
    expected, not a display bug. Non-bike edges are never flagged (no banner, no gray).
    """
    if edge.mode != Mode.BIKE:
        return False
    return edge_elevation_deviation_m(node_a=node_a, node_b=node_b, edge=edge) > GradeConfig.ELEVATION_DEVIATION_WARN_M


def track_has_unreliable_elevation(*, track: Track) -> bool:
    """True iff any track point sits on an edge whose displayed elevation is unreliable (for the banner)."""
    return any(point.unreliable_elev for point in track.points)


def densify_track(route: RoutePath, track: Track) -> Track:
    """Expand the node-level track into the full 3D road polyline (no DEM at inference).

    Walks each edge's real 2D geometry so the profile follows the true road (station links stay
    straight); each leg's time spreads by along-edge distance, other fields carry from the node track.
    """
    assert len(track.points) == len(route.nodes), "track points must align with route nodes"
    out: list[TrackPoint] = []
    n_edges = len(route.edges)

    for index, (node_a, node_b, edge) in enumerate(route.iter_edges()):
        leg_point = track.points[index + 1]  # the arriving point carries this leg's condition/speed/grade
        verts = edge_vertices_3d(node_a=node_a, node_b=node_b, edge=edge)
        t_start, t_end = track.points[index].elapsed_s, track.points[index + 1].elapsed_s
        vxy = np.asarray([(v[0], v[1]) for v in verts], dtype=np.float64)
        seg_lengths = haversine_vec(lat_a=vxy[:-1, 1], lon_a=vxy[:-1, 0], lat_b=vxy[1:, 1], lon_b=vxy[1:, 0])
        total = float(seg_lengths.sum()) or 1.0
        cum = 0.0
        last_leg = index == n_edges - 1
        stop = len(verts) if last_leg else len(verts) - 1  # avoid duplicating shared node vertex
        for i in range(stop):
            lon, lat, elev = verts[i]
            out.append(
                TrackPoint(
                    lat=lat,
                    lon=lon,
                    elevation_m=elev,
                    elapsed_s=t_start + (t_end - t_start) * (cum / total),
                    mode=edge.mode,
                    surface_bad=leg_point.surface_bad,
                    road_bad=leg_point.road_bad,
                    grade=leg_point.grade,
                    speed_kmh=leg_point.speed_kmh,
                    unreliable_elev=leg_point.unreliable_elev,
                )
            )
            if i < len(seg_lengths):
                cum += seg_lengths[i]

    # Stats are unchanged by densification (same legs) — carry both groups through.
    return Track(points=out, bike=track.bike, total=track.total)
