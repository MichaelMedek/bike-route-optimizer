"""Track-builder tests — geometry, elevation, adaptive timing, rolled-up totals."""

from bike_router.constants import GpxConfig, SpeedConfig
from bike_router.track import build_track
from tests.conftest import make_line_graph


def test_track_points_and_totals():
    graph = make_line_graph()
    track = build_track(graph=graph, node_path=[1, 2, 3])
    # 3 nodes → 3 points; two 800 m edges → 1.6 km
    assert len(track.points) == 3
    assert abs(track.distance_km - 1.6) < 1e-6
    # 100→130→100 → +30 m / -30 m
    assert track.ascent_m == 30.0
    assert track.descent_m == 30.0


def test_track_timestamps_monotonic_and_start_zero():
    graph = make_line_graph()
    track = build_track(graph=graph, node_path=[1, 2, 3])
    elapsed = [point.elapsed_s for point in track.points]
    assert elapsed[0] == 0.0
    assert elapsed[0] < elapsed[1] < elapsed[2]
    assert track.duration_min == elapsed[-1] / GpxConfig.SECONDS_PER_HOUR * GpxConfig.MINUTES_PER_HOUR


def test_track_uphill_segment_is_slower_than_downhill():
    graph = make_line_graph()
    track = build_track(graph=graph, node_path=[1, 2, 3])
    uphill_s = track.points[1].elapsed_s - track.points[0].elapsed_s  # 1→2 climbs
    downhill_s = track.points[2].elapsed_s - track.points[1].elapsed_s  # 2→3 descends
    assert uphill_s > downhill_s  # same length, uphill takes longer


def test_track_elevations_are_real_node_values():
    graph = make_line_graph()
    track = build_track(graph=graph, node_path=[1, 2, 3])
    assert [round(point.elevation_m) for point in track.points] == [100, 130, 100]


def test_track_average_speed_within_bounds():
    graph = make_line_graph()
    track = build_track(graph=graph, node_path=[1, 2, 3])
    avg_kmh = track.distance_km / (track.duration_min / GpxConfig.MINUTES_PER_HOUR)
    assert SpeedConfig.WALK_KMH <= avg_kmh <= max(SpeedConfig.BASE_KMH_BY_TIER.values())
