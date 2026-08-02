"""track tests — the ONE per-point time/elevation/colour structure: geometry, timing, densify.

One test_<fn> per production symbol (exact-name mirror) and a TestFoo per dataclass. Folds the
former test_track_rail.py: bike timing, rail/station ride time + boarding wait, condition/grade
classification, both colour scales, and the 3D densify. Every original assertion is preserved.
"""

import numpy as np
import pytest

from bike_router.core.constants import GpxConfig, GradeConfig, Mode, NodeType, Palette, RailConfig, SpeedConfig
from bike_router.core.route_path import RouteEdge, RouteNode, RoutePath
from bike_router.core.track import (
    RouteStats,
    Track,
    TrackPoint,
    _track_point,
    build_track,
    classify_condition,
    classify_grade,
    climb_totals,
    cumulative_km,
    densify_track,
    edge_condition_speed,
    edge_display_unreliable,
    edge_elevation_deviation_m,
    edge_grade,
    edge_vertices_3d,
    grade_color,
    leg_km,
    project_markers_onto_track,
    segment_color,
    track_has_unreliable_elevation,
)
from tests.conftest import (
    make_condition_route,
    make_densify_detour_route,
    make_exchange_rail_route,
    make_line_route,
    make_rail_route,
)


def _rail_ride_s(*, rail_m: float) -> float:
    """Expected train ride time (s) for a rail distance — one source for the timing asserts."""
    return rail_m / (RailConfig.RAIL_SPEED_KMH * GpxConfig.METERS_PER_KM / GpxConfig.SECONDS_PER_HOUR)


def _bike_edge(*, surface: str, highway: str, length_m: float = 800.0) -> RouteEdge:
    """One bike RouteEdge with the given surface/road for the condition/speed contracts."""
    return RouteEdge(
        from_node=1, to_node=2, mode=Mode.BIKE, length_m=length_m, surface=surface, highway=highway, geometry=None
    )


# --- dataclasses -------------------------------------------------------------


class TestTrackPoint:
    def test_carries_position_time_and_edge_condition(self):
        pt = TrackPoint(
            lat=48.0,
            lon=8.0,
            elevation_m=100.0,
            elapsed_s=12.0,
            mode=Mode.BIKE,
            surface_bad=True,
            road_bad=False,
            grade=0.05,
            speed_kmh=18.0,
            unreliable_elev=False,
        )
        assert (pt.lat, pt.lon, pt.elevation_m, pt.elapsed_s) == (48.0, 8.0, 100.0, 12.0)
        assert pt.mode == Mode.BIKE and pt.surface_bad is True and pt.road_bad is False
        assert pt.grade == 0.05 and pt.speed_kmh == 18.0

    def test_is_frozen(self):
        pt = TrackPoint(
            lat=48.0,
            lon=8.0,
            elevation_m=0.0,
            elapsed_s=0.0,
            mode=Mode.BIKE,
            surface_bad=False,
            road_bad=False,
            grade=0.0,
            speed_kmh=1.0,
            unreliable_elev=False,
        )
        with pytest.raises(AttributeError):
            pt.elapsed_s = 1.0  # type: ignore[misc]


class TestRouteStats:
    def test_format_strings_are_single_source(self):
        # The CLI, Streamlit metrics, and PNG overlay all render via these properties, so
        # the format specs (and the unicode minus U+2212) live in ONE place.
        stats = RouteStats(distance_km=7.04, duration_min=23.6, ascent_m=218.4, descent_m=26.7)
        assert stats.distance_str == "7.0 km"
        assert stats.duration_str == "24 min"
        assert stats.ascent_str == "+218 m"
        assert stats.descent_str == "−27 m"  # unicode minus, rounded
        assert stats.oneline == "7.0 km · 24 min · +218 m / −27 m"
        assert stats.metric_pairs(duration_label="Ride time") == (
            ("Distance", "7.0 km"),
            ("Ride time", "24 min"),
            ("Ascent", "+218 m"),
            ("Descent", "−27 m"),
        )

    def test_is_frozen(self):
        with pytest.raises(AttributeError):
            RouteStats(distance_km=1.0, duration_min=1.0, ascent_m=0.0, descent_m=0.0).ascent_m = 9.0  # type: ignore[misc]


class TestTrack:
    def test_splits_bike_only_from_whole_journey(self):
        # A pure-bike route: the pedalled-only stats equal the whole-journey stats.
        track = build_track(route=make_line_route())
        assert track.bike == track.total
        assert len(track.points) == len(make_line_route().nodes)


# --- distance / climb --------------------------------------------------------


def test_leg_km():
    # Per-consecutive-point great-circle km (length n-1); cumulative_km is its running sum.
    track = build_track(route=make_line_route())
    legs = leg_km(points=track.points)
    assert len(legs) == len(track.points) - 1 and all(km > 0 for km in legs)
    assert cumulative_km(points=track.points)[-1] == pytest.approx(float(sum(legs)))  # cumsum == Σ legs


def test_cumulative_km():
    # The ONE distance axis: starts at 0, monotonic non-decreasing, ends positive for a real route.
    track = build_track(route=make_line_route())
    dists = cumulative_km(points=track.points)
    assert dists[0] == 0.0
    assert all(b >= a for a, b in zip(dists[:-1], dists[1:], strict=True))
    assert dists[-1] > 0.0


def test_project_markers_onto_track():
    # A marker at node 2's coords (8.01, 48.0) snaps to that node's distance + elevation (130 m).
    track = build_track(route=make_line_route())
    placed = project_markers_onto_track(track=track, markers=[(48.0, 8.01, "mid")])
    dist_km, elev_m, label = placed[0]
    assert label == "mid"
    assert elev_m == pytest.approx(130.0)
    assert dist_km == pytest.approx(cumulative_km(points=track.points)[1])


def test_climb_totals():
    # Gross up-/down-sum, NOT net: over a hill (net 0) each still reports the full height; [] → (0,0).
    ascent, descent = climb_totals(deltas=[+30.0, -30.0])  # 100 → 130 → 100 m
    assert ascent == 30.0 and descent == 30.0
    up, down = climb_totals(deltas=[+30.0, -10.0, +5.0, -25.0])  # rolling hills, net 0
    assert up == 35.0 and down == 35.0
    assert climb_totals(deltas=[]) == (0.0, 0.0)


# --- grade / condition / speed (single branch points) ------------------------


def test_edge_grade():
    # Signed rise/run: +uphill, −downhill, 0 flat; length is a baked > 0 invariant.
    assert edge_grade(elev_source=100.0, elev_target=150.0, length_m=1000.0) == pytest.approx(0.05)
    assert edge_grade(elev_source=150.0, elev_target=100.0, length_m=1000.0) == pytest.approx(-0.05)
    assert edge_grade(elev_source=100.0, elev_target=100.0, length_m=500.0) == 0.0


def test_edge_condition_speed():
    # bike: surface_bad iff tier != 0, road_bad iff main road, adaptive speed; rail/station never
    # bad and ride at fixed RAIL_SPEED / walking pace; an unknown mode fails loud.
    s_bad, r_bad, speed = edge_condition_speed(
        edge=_bike_edge(surface="asphalt", highway="residential"), elev_source=100.0, elev_target=100.0
    )
    assert s_bad is False and r_bad is False and speed == pytest.approx(SpeedConfig.BASE_KMH_AT_WEIGHT0)
    s_bad, r_bad, _ = edge_condition_speed(
        edge=_bike_edge(surface="grass", highway="primary"), elev_source=100.0, elev_target=100.0
    )
    assert s_bad is True and r_bad is True  # rough surface (grass) + main road (primary)

    rail_edge = RouteEdge(
        from_node=1, to_node=2, mode=Mode.RAIL, length_m=1.0, surface=None, highway=None, geometry=None
    )
    assert edge_condition_speed(edge=rail_edge, elev_source=0.0, elev_target=500.0) == (
        False,
        False,
        RailConfig.RAIL_SPEED_KMH,
    )
    station_edge = RouteEdge(
        from_node=1, to_node=2, mode=Mode.STATION, length_m=1.0, surface=None, highway=None, geometry=None
    )
    assert edge_condition_speed(edge=station_edge, elev_source=0.0, elev_target=0.0) == (
        False,
        False,
        SpeedConfig.WALK_KMH,
    )
    bad_edge = RouteEdge(from_node=1, to_node=2, mode="fly", length_m=1.0, surface=None, highway=None, geometry=None)
    with pytest.raises(AssertionError, match="unknown edge mode"):
        edge_condition_speed(edge=bad_edge, elev_source=0.0, elev_target=0.0)


def test_classify_condition():
    # The SINGLE quality branch: train wins; then the four bike combinations of surface/road badness.
    assert classify_condition(mode=Mode.RAIL, surface_bad=True, road_bad=True) == "train"
    assert classify_condition(mode=Mode.BIKE, surface_bad=True, road_bad=True) == "main road + unpaved"
    assert classify_condition(mode=Mode.BIKE, surface_bad=False, road_bad=True) == "main road"
    assert classify_condition(mode=Mode.BIKE, surface_bad=True, road_bad=False) == "unpaved"
    assert classify_condition(mode=Mode.BIKE, surface_bad=False, road_bad=False) == "good"


def test_classify_grade():
    # The SINGLE grade branch: a train keeps its own label; uphill/downhill at OR past ±MARGIN,
    # flat only STRICTLY inside (so exactly ±MARGIN slopes; ±1% < 2% margin is flat).
    assert classify_grade(mode=Mode.RAIL, grade=0.5) == "train"
    assert classify_grade(mode=Mode.BIKE, grade=GradeConfig.MARGIN) == "uphill"  # exactly at → sloped
    assert classify_grade(mode=Mode.BIKE, grade=-GradeConfig.MARGIN) == "downhill"
    assert classify_grade(mode=Mode.BIKE, grade=GradeConfig.MARGIN - 0.005) == "flat"  # just under → flat
    assert classify_grade(mode=Mode.BIKE, grade=0.0) == "flat"


def test_segment_color():
    # RGB on the quality scale via classify_condition → Palette.CONDITION_COLORS (label/colour agree).
    good = segment_color(mode=Mode.BIKE, surface_bad=False, road_bad=False)
    assert good == list(Palette.hex_to_rgb(hex_color=Palette.CONDITION_COLORS["good"]))
    train = segment_color(mode=Mode.RAIL, surface_bad=False, road_bad=False)
    assert train == list(Palette.hex_to_rgb(hex_color=Palette.CONDITION_COLORS["train"]))


def test_grade_color():
    # RGB on the grade scale via classify_grade → Palette.GRADE_COLORS (flat/uphill/downhill/train).
    downhill = grade_color(mode=Mode.BIKE, grade=-0.5)
    assert downhill == list(Palette.hex_to_rgb(hex_color=Palette.GRADE_COLORS["downhill"]))
    train = grade_color(mode=Mode.RAIL, grade=0.0)
    assert train == list(Palette.hex_to_rgb(hex_color=Palette.GRADE_COLORS["train"]))


def test_track_point():
    # The ONE point builder: carries its node position/elevation + the edge's condition/grade in
    # its TRAVEL direction (elev_from → elev_to), NOT the reversed direction of where it sits.
    at = RouteNode(osmid=1, lat=48.0, lon=8.0, elevation_m=100.0, node_type=NodeType.BIKE, station_name=None)
    edge = _bike_edge(surface="grass", highway="primary", length_m=1000.0)
    pt = _track_point(at=at, edge=edge, elev_from=100.0, elev_to=150.0, elapsed_s=42.0, unreliable=False)
    assert (pt.lat, pt.lon, pt.elevation_m, pt.elapsed_s) == (48.0, 8.0, 100.0, 42.0)
    assert pt.surface_bad is True and pt.road_bad is True  # grass (rough) + primary (main road)
    assert pt.grade == pytest.approx(0.05)  # climb 100→150 over 1000 m → +5% (uphill, not the reverse)
    # REGRESSION: an arriving point sits at node_b but the grade must follow the ridden direction
    # a→b — a climb reads +, never the reversed − (the "uphill shown as downhill" bug).
    climbing = _track_point(at=at, edge=edge, elev_from=100.0, elev_to=200.0, elapsed_s=0.0, unreliable=False)
    assert climbing.grade > 0  # ascending a→b → positive grade

    # INVARIANT: only a BIKE edge may be flagged unreliable — a bike edge can, a rail/station edge
    # asserts loudly (trains legitimately tunnel/bridge; the gray flag is bike-only at its emitter).
    bike_pt = _track_point(at=at, edge=edge, elev_from=100.0, elev_to=150.0, elapsed_s=0.0, unreliable=True)
    assert bike_pt.unreliable_elev is True
    rail_edge = RouteEdge(
        from_node=1, to_node=2, mode=Mode.RAIL, length_m=1000.0, surface=None, highway=None, geometry=None
    )
    with pytest.raises(AssertionError, match="only bike edges may be unreliable"):
        _track_point(at=at, edge=rail_edge, elev_from=100.0, elev_to=150.0, elapsed_s=0.0, unreliable=True)


# --- build_track (bike + rail + station timing) ------------------------------


def test_build_track():
    # Bike route: 3 nodes → 3 points, 1.6 km, +30/−30 climb, monotonic zero-based time, uphill slower,
    # real node elevations, plausible average speed. Rail route: ride time + half-boarding wait, the
    # whole-journey climb spans the train while bike-only excludes it, and the 80 km/h ride does not
    # trip the bike avg-speed assert. Exchange trip charges exactly ONE boarding despite the change.
    track = build_track(route=make_line_route())
    assert len(track.points) == 3
    assert track.total.distance_km == 1.6
    assert track.total.ascent_m == 30.0 and track.total.descent_m == 30.0
    assert track.bike == track.total  # pure bike
    elapsed = [p.elapsed_s for p in track.points]
    assert elapsed[0] == 0.0 and elapsed[0] < elapsed[1] < elapsed[2]
    assert track.total.duration_min == elapsed[-1] / GpxConfig.SECONDS_PER_HOUR * GpxConfig.MINUTES_PER_HOUR
    assert (elapsed[1] - elapsed[0]) > (elapsed[2] - elapsed[1])  # uphill 1→2 slower than downhill 2→3
    assert [round(p.elevation_m) for p in track.points] == [100, 130, 100]
    avg_kmh = track.total.distance_km / (track.total.duration_min / GpxConfig.MINUTES_PER_HOUR)
    assert SpeedConfig.WALK_KMH <= avg_kmh <= SpeedConfig.BASE_KMH_AT_WEIGHT0

    # rail route: one station edge (board) → half a wait, route ends on the train (no alight hop)
    rail = build_track(route=make_rail_route())
    assert rail.points[-1].elapsed_s == 0.5 * RailConfig.BOARDING_WAIT_S + _rail_ride_s(rail_m=7000.0)
    assert rail.total.ascent_m == pytest.approx(400.0) and rail.total.descent_m == 0.0  # station +5, ride +395
    assert rail.bike.ascent_m == 0.0 and rail.bike.descent_m == 0.0  # no pedalled edge
    assert rail.total.ascent_m != rail.bike.ascent_m  # the two rows differ on a train route
    assert rail.bike.distance_km == 0.0 and rail.total.distance_km == pytest.approx(7.08)
    assert rail.bike.duration_min < rail.total.duration_min  # 80 km/h ride, no avg-speed assert tripped

    # exchange trip: board at A + alight at C = ONE full wait; the A→B→C rail hop adds none
    exchange = make_exchange_rail_route()
    ex_track = build_track(route=exchange)
    assert ex_track.points[-1].elapsed_s == pytest.approx(
        RailConfig.BOARDING_WAIT_S + _rail_ride_s(rail_m=4000.0 + 3000.0)
    )
    assert exchange.nodes[-1].node_type == NodeType.BIKE
    assert ex_track.total.ascent_m == 0.0 and ex_track.total.descent_m == 0.0  # flat (all 100 m)

    # condition/speed baked per point: good quiet leg vs main-road leg
    cond = build_track(route=make_condition_route())
    assert cond.points[1].surface_bad is False and cond.points[1].road_bad is False
    assert cond.points[1].speed_kmh == 25.0
    assert cond.points[2].road_bad is True and cond.points[2].surface_bad is False  # primary, asphalt


# --- 3D densify --------------------------------------------------------------


def test_edge_vertices_3d():
    # Real 2D polyline with elevation interpolated LINEARLY node-to-node by along-edge distance;
    # a geometry-less hop is a straight two-node segment at the node elevations.
    node_a = RouteNode(osmid=1, lat=48.0, lon=8.0, elevation_m=100.0, node_type=NodeType.BIKE, station_name=None)
    node_b = RouteNode(osmid=2, lat=48.0, lon=8.02, elevation_m=200.0, node_type=NodeType.BIKE, station_name=None)
    geom = [(8.0, 48.0), (8.01, 48.0), (8.02, 48.0)]  # straight, midpoint halfway
    edge = RouteEdge(
        from_node=1, to_node=2, mode=Mode.BIKE, length_m=1500.0, surface="asphalt", highway="residential", geometry=geom
    )
    verts = edge_vertices_3d(node_a=node_a, node_b=node_b, edge=edge)
    assert [round(z) for _x, _y, z in verts] == [100, 150, 200]  # linear z 100→150→200

    straight = RouteEdge(
        from_node=1, to_node=2, mode=Mode.RAIL, length_m=1500.0, surface=None, highway=None, geometry=None
    )
    assert edge_vertices_3d(node_a=node_a, node_b=node_b, edge=straight) == [
        (8.0, 48.0, 100.0),
        (8.02, 48.0, 200.0),
    ]


def _elev_edge(*, geometry_z: list[float] | None, mode: str = Mode.BIKE) -> tuple:
    """(node_a, node_b, edge) climbing 100→200 over a 3-vertex geometry with the given baked z."""
    node_a = RouteNode(osmid=1, lat=48.0, lon=8.0, elevation_m=100.0, node_type=NodeType.BIKE, station_name=None)
    node_b = RouteNode(osmid=2, lat=48.0, lon=8.02, elevation_m=200.0, node_type=NodeType.BIKE, station_name=None)
    geom = [(8.0, 48.0), (8.01, 48.0), (8.02, 48.0)]
    edge = RouteEdge(
        from_node=1,
        to_node=2,
        mode=mode,
        length_m=1500.0,
        surface="asphalt",
        highway="residential",
        geometry=geom,
        geometry_z=geometry_z,
    )
    return node_a, node_b, edge


def test_edge_elevation_deviation_m():
    # Linear display z is 100/150/200; baked terrain dips to 60 at the midpoint → max gap 90 m.
    node_a, node_b, edge = _elev_edge(geometry_z=[100.0, 60.0, 200.0])
    assert edge_elevation_deviation_m(node_a=node_a, node_b=node_b, edge=edge) == pytest.approx(90.0)
    # matching terrain → 0; no baked z → 0 (nothing to compare)
    flat_a, flat_b, flat = _elev_edge(geometry_z=[100.0, 150.0, 200.0])
    assert edge_elevation_deviation_m(node_a=flat_a, node_b=flat_b, edge=flat) == pytest.approx(0.0)
    na, nb, no_z = _elev_edge(geometry_z=None)
    assert edge_elevation_deviation_m(node_a=na, node_b=nb, edge=no_z) == 0.0


def test_edge_display_unreliable():
    # A BIKE edge beyond the 50 m warn threshold → unreliable; a small dip stays reliable.
    big_a, big_b, big = _elev_edge(geometry_z=[100.0, 60.0, 200.0])  # 90 m gap > 50
    assert edge_display_unreliable(node_a=big_a, node_b=big_b, edge=big)
    small_a, small_b, small = _elev_edge(geometry_z=[100.0, 140.0, 200.0])  # 10 m gap < 50
    assert not edge_display_unreliable(node_a=small_a, node_b=small_b, edge=small)
    # RAIL edge with the SAME huge gap is NEVER flagged — trains legitimately tunnel/bridge.
    rail_a, rail_b, rail = _elev_edge(geometry_z=[100.0, 60.0, 200.0], mode=Mode.RAIL)
    assert not edge_display_unreliable(node_a=rail_a, node_b=rail_b, edge=rail)
    # STATION edge with the SAME huge gap is likewise never flagged — only bike edges can go gray.
    stn_a, stn_b, stn = _elev_edge(geometry_z=[100.0, 60.0, 200.0], mode=Mode.STATION)
    assert not edge_display_unreliable(node_a=stn_a, node_b=stn_b, edge=stn)


def test_track_has_unreliable_elevation():
    # True iff any track point flags an unreliable arriving edge.
    ok = TrackPoint(
        lat=48.0,
        lon=8.0,
        elevation_m=100.0,
        elapsed_s=0.0,
        mode=Mode.BIKE,
        surface_bad=False,
        road_bad=False,
        grade=0.0,
        speed_kmh=20.0,
        unreliable_elev=False,
    )
    bad = TrackPoint(
        lat=48.0,
        lon=8.0,
        elevation_m=100.0,
        elapsed_s=0.0,
        mode=Mode.BIKE,
        surface_bad=False,
        road_bad=False,
        grade=0.0,
        speed_kmh=20.0,
        unreliable_elev=True,
    )
    assert not track_has_unreliable_elevation(
        track=Track(points=[ok, ok], bike=RouteStats(1.0, 1.0, 0.0, 0.0), total=RouteStats(1.0, 1.0, 0.0, 0.0))
    )
    assert track_has_unreliable_elevation(
        track=Track(points=[ok, bad], bike=RouteStats(1.0, 1.0, 0.0, 0.0), total=RouteStats(1.0, 1.0, 0.0, 0.0))
    )


def test_densify_track():
    # Detour edge: keeps the real 2D eastward bulge yet interpolates z linearly (no vertex > 140),
    # spreads timing by distance, and carries stats through unchanged. A geometry-less rail hop
    # falls back to a straight segment at the two node elevations (still no DEM).
    route = make_densify_detour_route()
    stats = RouteStats(distance_km=3.0, duration_min=10.0, ascent_m=40.0, descent_m=0.0)
    track = Track(
        points=[
            TrackPoint(
                lat=48.00,
                lon=8.00,
                elevation_m=100.0,
                elapsed_s=0.0,
                mode=Mode.BIKE,
                surface_bad=False,
                road_bad=False,
                grade=0.0,
                speed_kmh=25.0,
                unreliable_elev=False,
            ),
            TrackPoint(
                lat=48.02,
                lon=8.00,
                elevation_m=140.0,
                elapsed_s=600.0,
                mode=Mode.BIKE,
                surface_bad=False,
                road_bad=False,
                grade=0.0,
                speed_kmh=18.0,
                unreliable_elev=False,
            ),
        ],
        bike=stats,
        total=stats,
    )
    dense = densify_track(route=route, track=track)
    assert len(dense.points) == 3  # three real polyline vertices
    assert dense.points[0].elapsed_s == 0.0 and dense.points[-1].elapsed_s == 600.0
    assert dense.total.distance_km == 3.0
    assert max(p.lon for p in dense.points) > 8.02  # the eastward 2D bulge is present
    assert max(p.elevation_m for p in dense.points) == pytest.approx(140.0)
    assert dense.points[0].elevation_m == pytest.approx(100.0)
    assert all(100.0 - 1e-6 <= p.elevation_m <= 140.0 + 1e-6 for p in dense.points)
    assert dense.total.ascent_m == pytest.approx(40.0) and dense.total.descent_m == pytest.approx(0.0)

    # geometry-less rail hop → straight segment at node elevations
    nodes = [
        RouteNode(osmid=1, lat=48.0, lon=8.0, elevation_m=100.0, node_type=NodeType.RAIL, station_name="A"),
        RouteNode(osmid=2, lat=48.0, lon=8.1, elevation_m=400.0, node_type=NodeType.RAIL, station_name="B"),
    ]
    rail_route = RoutePath(
        nodes=nodes,
        edges=[
            RouteEdge(
                from_node=1, to_node=2, mode=Mode.RAIL, length_m=8000.0, surface=None, highway=None, geometry=None
            )
        ],
    )
    rail_stats = RouteStats(distance_km=8.0, duration_min=6.0, ascent_m=0.0, descent_m=0.0)
    rail_track = Track(
        points=[
            TrackPoint(
                lat=48.0,
                lon=8.0,
                elevation_m=100.0,
                elapsed_s=0.0,
                mode=Mode.RAIL,
                surface_bad=False,
                road_bad=False,
                grade=0.0,
                speed_kmh=80.0,
                unreliable_elev=False,
            ),
            TrackPoint(
                lat=48.0,
                lon=8.1,
                elevation_m=400.0,
                elapsed_s=360.0,
                mode=Mode.RAIL,
                surface_bad=False,
                road_bad=False,
                grade=0.0,
                speed_kmh=80.0,
                unreliable_elev=False,
            ),
        ],
        bike=rail_stats,
        total=rail_stats,
    )
    rail_dense = densify_track(route=rail_route, track=rail_track)
    assert [round(p.elevation_m) for p in rail_dense.points] == [100, 400]
    assert all(np.isfinite(p.elevation_m) for p in rail_dense.points)
