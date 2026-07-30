"""GPX export tests — built from a Track, timestamps + elevations round-trip."""

import datetime as dt

import gpxpy

from bike_router.core.gpx_export import build_gpx
from bike_router.core.track import build_track
from tests.conftest import make_line_route


def test_build_gpx():
    # One track point per node with its elevation; timestamps start at start_time, increase
    # monotonically, and the GPX end-time equals the track's total duration (consistent by build).
    track = build_track(route=make_line_route())
    points = gpxpy.parse(build_gpx(track=track)).tracks[0].segments[0].points
    assert len(points) == 3 and [round(p.elevation) for p in points] == [100, 130, 100]

    start = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    stamped = gpxpy.parse(build_gpx(track=track, start_time=start)).tracks[0].segments[0].points
    assert stamped[0].time == start and stamped[0].time < stamped[1].time < stamped[2].time
    total_s = (stamped[-1].time - stamped[0].time).total_seconds()
    assert abs(total_s - track.points[-1].elapsed_s) < 1e-6
