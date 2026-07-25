"""GPX export tests — built from a Track, timestamps + elevations round-trip."""

import datetime as dt

import gpxpy

from bike_router.gpx_export import build_gpx
from bike_router.track import build_track
from tests.conftest import make_line_graph


def _track():
    return build_track(graph=make_line_graph(), node_path=[1, 2, 3])


def test_build_gpx_roundtrips_points_and_elevation():
    xml = build_gpx(track=_track())
    points = gpxpy.parse(xml).tracks[0].segments[0].points
    assert len(points) == 3
    assert [round(p.elevation) for p in points] == [100, 130, 100]


def test_gpx_timestamps_increase_and_match_track():
    start = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    track = _track()
    points = gpxpy.parse(build_gpx(track=track, start_time=start)).tracks[0].segments[0].points
    assert points[0].time == start
    assert points[0].time < points[1].time < points[2].time
    # GPX end time equals the track's total duration (consistent by construction)
    total_s = (points[-1].time - points[0].time).total_seconds()
    assert abs(total_s - track.points[-1].elapsed_s) < 1e-6
