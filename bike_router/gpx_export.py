"""GPX track export via gpxpy.

Builds a standard track (GPX → GPXTrack → GPXTrackSegment → GPXTrackPoint) straight
from a Track (bike_router.track): every point carries its real DEM elevation and
the cumulative ride time from the surface/grade-adaptive speed model, so the GPX
end-timestamp equals the reported total duration by construction.
"""

import datetime as _dt

import gpxpy
import gpxpy.gpx

from bike_router.track import Track


def build_gpx(track: Track, start_time: _dt.datetime | None = None, track_name: str = "Optimized bike route") -> str:
    """Build GPX XML from a computed Track.

    Args:
        track: The route track (points carry elevation + cumulative elapsed time).
        start_time: Track start (defaults to now, UTC).
        track_name: GPX track name.

    Returns:
        GPX document as an XML string.
    """
    assert track.points, "track must have at least one point"
    start_time = start_time or _dt.datetime.now(_dt.UTC)

    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack(name=track_name)
    gpx.tracks.append(gpx_track)
    segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(segment)

    for point in track.points:
        segment.points.append(
            gpxpy.gpx.GPXTrackPoint(
                latitude=point.lat,
                longitude=point.lon,
                elevation=point.elevation_m,
                time=start_time + _dt.timedelta(seconds=point.elapsed_s),
            )
        )

    xml: str = gpx.to_xml()
    return xml
