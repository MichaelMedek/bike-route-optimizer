"""Track-builder rail/station timing + 3D-densify tests (no DEM at inference)."""

import numpy as np
import pytest

from bike_router.core.constants import GpxConfig, Mode, NodeType, RailConfig
from bike_router.core.route_path import RouteEdge, RouteNode, RoutePath
from bike_router.core.track import RouteStats, Track, TrackPoint, build_track, densify_track
from tests.conftest import (
    make_condition_route,
    make_densify_detour_route,
    make_exchange_rail_route,
    make_rail_route,
)


def _rail_ride_s(*, rail_m: float) -> float:
    """Expected train ride time (s) for a rail distance — one source for the timing asserts."""
    return rail_m / (RailConfig.RAIL_SPEED_KMH * GpxConfig.METERS_PER_KM / GpxConfig.SECONDS_PER_HOUR)


def test_build_track_rail_derives_ride_time_and_boarding_wait():
    track = build_track(route=make_rail_route())
    # One station edge (board at 2) → half a wait; the route ends on the train (no alight hop).
    expected_s = 0.5 * RailConfig.BOARDING_WAIT_S + _rail_ride_s(rail_m=7000.0)
    assert track.points[-1].elapsed_s == expected_s
    # The WHOLE-journey climb spans every edge (like total distance): station hop 200→205 m
    # (+5) then the rail ride 205→600 m (+395) = +400 m total, all downhill-free here.
    assert track.total.ascent_m == pytest.approx(400.0) and track.total.descent_m == 0.0
    # The BIKE-only climb excludes rail/station: this route has NO pedalled edges → 0.
    assert track.bike.ascent_m == 0.0 and track.bike.descent_m == 0.0
    # bike-only vs total split: this route has NO pedalled (Mode.BIKE) edges — just a
    # station-access hop + the train — so bike distance is 0 while total spans the 7 km ride.
    assert track.bike.distance_km == 0.0
    assert track.total.distance_km == pytest.approx(7.08)
    assert track.bike.duration_min < track.total.duration_min


def test_build_track_total_climb_includes_train_bike_climb_excludes_it():
    # Regression: the "bike + train" ascent must span the WHOLE journey (incl. the climb
    # the train covers), while "bike only" counts just pedalled edges. make_rail_route climbs
    # 200→205 m on the station hop and 205→600 m on the rail ride: total +400 m, bike-only +0.
    track = build_track(route=make_rail_route())
    assert track.total.ascent_m == pytest.approx(400.0)  # whole journey, train climb included
    assert track.bike.ascent_m == 0.0  # no pedalled edge → no bike climb
    assert track.total.ascent_m != track.bike.ascent_m  # the two rows MUST differ on a train route


def test_build_track_rail_does_not_trip_avg_speed_assert():
    # 80 km/h rail alone would exceed the 25 km/h bike ceiling — must not assert.
    track = build_track(route=make_rail_route())
    assert track.total.distance_km == pytest.approx(7.08)  # 80 m station + 7000 m rail, completed w/o assert


def test_build_track_exchange_trip_charges_boarding_exactly_once():
    # bike → A → B(exchange, degree-3) → C → bike. Each of the two station edges (board at A,
    # alight at C) carries HALF the wait; the A→B→C rail hop through the exchange adds none.
    # So total time = ONE full boarding wait (½ + ½) + the two rail rides (4000 m + 3000 m).
    route = make_exchange_rail_route()
    track = build_track(route=route)
    ride_s = _rail_ride_s(rail_m=4000.0 + 3000.0)
    # exactly ONE boarding wait despite the mid-trip change at the degree-3 exchange node B
    assert track.points[-1].elapsed_s == pytest.approx(RailConfig.BOARDING_WAIT_S + ride_s)
    assert route.nodes[-1].node_type == NodeType.BIKE
    assert track.total.ascent_m == 0.0 and track.total.descent_m == 0.0  # this route is flat (all 100 m)


def test_build_track_sets_condition_and_speed_per_point():
    # A good quiet bike leg → not bad; a main-road leg → road_bad True (surface still good).
    track = build_track(route=make_condition_route())
    # point[1] arrives via the good quiet leg; point[2] via the main-road leg.
    assert track.points[1].surface_bad is False and track.points[1].road_bad is False
    assert track.points[1].speed_kmh == 25.0
    assert track.points[2].road_bad is True  # primary → main road
    assert track.points[2].surface_bad is False  # asphalt → surface still good


def test_densify_track_follows_real_2d_polyline_with_interpolated_elevation():
    # The fixture's edge 1→2 detours EAST; node elevations are 100→140. densify keeps the real
    # 2D bulge yet interpolates z linearly between the nodes (single source: same elevation the
    # optimiser + stats use), so no vertex exceeds 140.
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
            ),
        ],
        bike=stats,
        total=stats,
    )
    dense = densify_track(route=route, track=track)
    assert len(dense.points) == 3  # the three real polyline vertices
    assert dense.points[0].elapsed_s == 0.0 and dense.points[-1].elapsed_s == 600.0  # timing preserved
    assert dense.total.distance_km == 3.0  # stats carried over unchanged
    assert max(p.lon for p in dense.points) > 8.02  # the eastward 2D bulge is present
    # z is LINEAR node-to-node (100→140): no vertex exceeds 140.
    assert max(p.elevation_m for p in dense.points) == pytest.approx(140.0)
    assert dense.points[0].elevation_m == pytest.approx(100.0)
    assert all(100.0 - 1e-6 <= p.elevation_m <= 140.0 + 1e-6 for p in dense.points)
    assert dense.total.ascent_m == pytest.approx(40.0) and dense.total.descent_m == pytest.approx(0.0)


def test_densify_track_straight_hop_without_geometry():
    # Rail/station edges have no geometry → densify falls back to a straight segment
    # at the two node elevations (still no DEM).
    nodes = [
        RouteNode(osmid=1, lat=48.0, lon=8.0, elevation_m=100.0, node_type=NodeType.RAIL, station_name="A"),
        RouteNode(osmid=2, lat=48.0, lon=8.1, elevation_m=400.0, node_type=NodeType.RAIL, station_name="B"),
    ]
    route = RoutePath(
        nodes=nodes,
        edges=[
            RouteEdge(
                from_node=1, to_node=2, mode=Mode.RAIL, length_m=8000.0, surface=None, highway=None, geometry=None
            )
        ],
    )
    stats = RouteStats(distance_km=8.0, duration_min=6.0, ascent_m=0.0, descent_m=0.0)
    track = Track(
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
            ),
        ],
        bike=stats,
        total=stats,
    )
    dense = densify_track(route=route, track=track)
    assert [round(p.elevation_m) for p in dense.points] == [100, 400]  # straight hop at node elevations
    assert all(np.isfinite(p.elevation_m) for p in dense.points)
