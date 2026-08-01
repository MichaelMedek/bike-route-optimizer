"""GPX export tests — built from a Track, timestamps + elevations round-trip."""

import datetime as dt

import gpxpy

from bike_router.core.gpx_export import build_gpx
from bike_router.core.track import build_track, densify_track
from tests.conftest import make_line_route


def test_build_gpx():
    # One track point per node with its elevation; timestamps start at start_time, increase
    # monotonically, and the GPX end-time equals the track's total duration (consistent by build).
    track = build_track(route=make_line_route())
    points = (
        gpxpy.parse(build_gpx(track=track, start_time=None, track_name="Optimized bike route"))
        .tracks[0]
        .segments[0]
        .points
    )
    assert len(points) == 3 and [round(p.elevation) for p in points] == [100, 130, 100]

    start = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    stamped = (
        gpxpy.parse(build_gpx(track=track, start_time=start, track_name="Optimized bike route"))
        .tracks[0]
        .segments[0]
        .points
    )
    assert stamped[0].time == start and stamped[0].time < stamped[1].time < stamped[2].time
    total_s = (stamped[-1].time - stamped[0].time).total_seconds()
    assert abs(total_s - track.points[-1].elapsed_s) < 1e-6


def test_build_gpx_densified_track_monotonic_no_regression():
    # A DENSIFIED track (many interpolated points) still exports one GPX point per track point,
    # with non-decreasing timestamps (interpolated legs must never step backwards in time).
    route = make_line_route()
    dense = densify_track(route=route, track=build_track(route=route))
    start = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    points = (
        gpxpy.parse(build_gpx(track=dense, start_time=start, track_name="Optimized bike route"))
        .tracks[0]
        .segments[0]
        .points
    )
    assert len(points) == len(dense.points) >= 3  # densified beyond the 3 nodes
    times = [p.time for p in points]
    assert times == sorted(times)  # monotonic non-decreasing, no interpolation regression
