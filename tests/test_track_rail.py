"""Track-builder rail/transfer timing + 3D-densify tests (no DEM at inference)."""

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from bike_router.constants import GpxConfig, Mode, RailConfig
from bike_router.track import Track, TrackPoint, build_track, densify_track


def _rail_graph() -> nx.MultiDiGraph:
    """S(bike) → station A → station B (rail) with transfer legs.

    Node 1 bike, nodes 2/3 stations. Transfer 1→2 enters a station (boarding wait);
    rail 2→3 rides at RAIL_SPEED_KMH. All leg times are DERIVED in build_track from
    length + is_station — nothing time-related is stored on the edges.
    """
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(1, x=8.00, y=48.0, elevation=200.0, is_station=False)
    graph.add_node(2, x=8.001, y=48.0, elevation=205.0, is_station=True)  # station A
    graph.add_node(3, x=8.10, y=48.0, elevation=600.0, is_station=True)  # station B
    # boarding transfer (enters station 2 → wait applied), then the rail ride
    graph.add_edge(1, 2, key=0, length=80.0, surface=None, highway=None, mode=Mode.TRANSFER, custom_cost=80.0)
    graph.add_edge(2, 3, key=0, length=7000.0, surface=None, highway=None, mode=Mode.RAIL, custom_cost=7000.0)
    return graph


def test_build_track_rail_derives_ride_time_and_boarding_wait():
    graph = _rail_graph()
    track = build_track(graph=graph, node_path=[1, 2, 3])
    rail_ride = 7000.0 / (RailConfig.RAIL_SPEED_KMH * GpxConfig.METERS_PER_KM / GpxConfig.SECONDS_PER_HOUR)
    expected_s = RailConfig.BOARDING_WAIT_S + rail_ride  # boarding at station 2 + ride 2→3
    assert track.points[-1].elapsed_s == expected_s
    # rail climb is NOT counted as pedalled ascent
    assert track.ascent_m == 0.0
    assert track.descent_m == 0.0


def test_build_track_rail_does_not_trip_avg_speed_assert():
    # 80 km/h rail alone would exceed the 25 km/h bike ceiling — must not assert.
    graph = _rail_graph()
    track = build_track(graph=graph, node_path=[1, 2, 3])
    assert track.distance_km > 0  # completed without AssertionError


def test_densify_track_follows_baked_3d_polyline_and_keeps_timing():
    # Edge 1→2 endpoints are due-north, but its baked geometry detours EAST and carries
    # per-vertex elevation (3D LineString). densify_track must emit those real vertices
    # and their baked elevations — no DEM involved.
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(1, x=8.00, y=48.00, elevation=100.0)
    graph.add_node(2, x=8.00, y=48.02, elevation=140.0)
    # 3D polyline: bulges east, rising to 200 m at the apex
    detour = LineString([(8.00, 48.00, 100.0), (8.03, 48.01, 200.0), (8.00, 48.02, 140.0)])
    graph.add_edge(
        1,
        2,
        key=0,
        length=3000.0,
        surface="asphalt",
        highway="residential",
        mode=Mode.BIKE,
        custom_cost=3000.0,
        geometry=detour,
    )
    track = Track(
        points=[
            TrackPoint(lat=48.00, lon=8.00, elevation_m=100.0, elapsed_s=0.0, mode=Mode.BIKE),
            TrackPoint(lat=48.02, lon=8.00, elevation_m=140.0, elapsed_s=600.0, mode=Mode.BIKE),
        ],
        distance_km=3.0,
        duration_min=10.0,
        ascent_m=40.0,  # node-level 100→140; densify must NOT recompute from noisy vertices
        descent_m=0.0,
    )
    dense = densify_track(graph=graph, node_path=[1, 2], track=track)
    assert len(dense.points) == 3  # the three real polyline vertices
    assert dense.points[0].elapsed_s == 0.0 and dense.points[-1].elapsed_s == 600.0  # timing preserved
    assert dense.distance_km == 3.0  # totals carried over
    assert max(p.lon for p in dense.points) > 8.02  # the eastward bulge is present
    assert max(p.elevation_m for p in dense.points) == 200.0  # baked apex elevation used in the profile
    # ascent/descent carried from the node-level track, NOT re-summed from the 200 m apex jitter
    assert dense.ascent_m == pytest.approx(40.0) and dense.descent_m == pytest.approx(0.0)


def test_densify_track_straight_hop_without_geometry():
    # Rail/transfer edges have no geometry → densify falls back to a straight segment
    # at the two node elevations (still no DEM).
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(1, x=8.0, y=48.0, elevation=100.0)
    graph.add_node(2, x=8.1, y=48.0, elevation=400.0)
    graph.add_edge(1, 2, key=0, length=8000.0, surface=None, highway=None, mode=Mode.RAIL, custom_cost=8000.0)
    track = Track(
        points=[
            TrackPoint(lat=48.0, lon=8.0, elevation_m=100.0, elapsed_s=0.0, mode=Mode.RAIL),
            TrackPoint(lat=48.0, lon=8.1, elevation_m=400.0, elapsed_s=360.0, mode=Mode.RAIL),
        ],
        distance_km=8.0,
        duration_min=6.0,
        ascent_m=0.0,
        descent_m=0.0,
    )
    dense = densify_track(graph=graph, node_path=[1, 2], track=track)
    assert [round(p.elevation_m) for p in dense.points] == [100, 400]  # straight hop at node elevations
    assert all(np.isfinite(p.elevation_m) for p in dense.points)
