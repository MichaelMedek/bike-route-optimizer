"""Track-builder tests — geometry, elevation, adaptive timing, rolled-up totals."""

from bike_router.core.constants import GpxConfig, SpeedConfig
from bike_router.core.track import RouteStats, build_track, climb_totals
from tests.conftest import make_line_route


def test_climb_totals_reports_gross_not_net_over_a_hill():
    # Start and end at the SAME elevation but go over a hill: net change is 0, yet the ride
    # still climbs then descends, so ascent/descent must each report the GROSS hill height.
    ascent, descent = climb_totals(deltas=[+30.0, -30.0])  # 100 → 130 → 100 m
    assert ascent == 30.0 and descent == 30.0  # NOT 0 (a plain sum of deltas would give 0)
    # rolling hills: +30 −10 +5 −25 → up-sum 35, down-sum 35 (net 0 again)
    up, down = climb_totals(deltas=[+30.0, -10.0, +5.0, -25.0])
    assert up == 35.0 and down == 35.0
    assert climb_totals(deltas=[]) == (0.0, 0.0)  # empty path → no climb


def test_route_stats_format_strings_are_single_source():
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


def test_track_points_and_totals():
    track = build_track(route=make_line_route())
    # 3 nodes → 3 points; two 800 m edges → 1.6 km
    assert len(track.points) == 3
    assert track.total.distance_km == 1.6  # two 800 m edges, exact in float
    # 100→130→100 → +30 m / -30 m
    assert track.total.ascent_m == 30.0
    assert track.total.descent_m == 30.0
    # pure-bike route: bike stats equal the totals
    assert track.bike == track.total


def test_track_timestamps_monotonic_and_start_zero():
    track = build_track(route=make_line_route())
    elapsed = [point.elapsed_s for point in track.points]
    assert elapsed[0] == 0.0
    assert elapsed[0] < elapsed[1] < elapsed[2]
    assert track.total.duration_min == elapsed[-1] / GpxConfig.SECONDS_PER_HOUR * GpxConfig.MINUTES_PER_HOUR


def test_track_uphill_segment_is_slower_than_downhill():
    track = build_track(route=make_line_route())
    uphill_s = track.points[1].elapsed_s - track.points[0].elapsed_s  # 1→2 climbs
    downhill_s = track.points[2].elapsed_s - track.points[1].elapsed_s  # 2→3 descends
    assert uphill_s > downhill_s  # same length, uphill takes longer


def test_track_elevations_are_real_node_values():
    track = build_track(route=make_line_route())
    assert [round(point.elevation_m) for point in track.points] == [100, 130, 100]


def test_track_average_speed_within_bounds():
    track = build_track(route=make_line_route())
    avg_kmh = track.total.distance_km / (track.total.duration_min / GpxConfig.MINUTES_PER_HOUR)
    assert SpeedConfig.WALK_KMH <= avg_kmh <= max(SpeedConfig.BASE_KMH_BY_TIER.values())
