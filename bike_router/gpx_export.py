"""GPX track export via gpxpy.

Builds a standard track (GPX → GPXTrack → GPXTrackSegment → GPXTrackPoint) where
every point carries real DEM elevation and a synthesized timestamp. Since no
actual ride happened, timestamps are derived from cumulative great-circle
distance divided by an assumed cycling speed, starting at a given time.
"""

import datetime as _dt

import gpxpy
import gpxpy.gpx

from bike_router.constants import GpxConfig
from bike_router.geo import haversine_distance_m


def build_gpx(
    coords_latlon: list[tuple[float, float]],
    elevations_m: list[float | None],
    start_time: _dt.datetime | None = None,
    track_name: str = "Optimized bike route",
) -> str:
    """Build GPX XML for the route.

    Args:
        coords_latlon: Ordered (lat, lon) points along the route.
        elevations_m: Elevation (metres) per point, same length as coords.
        start_time: Track start (defaults to now, UTC).
        track_name: GPX track name.

    Returns:
        GPX document as an XML string.
    """
    if len(coords_latlon) != len(elevations_m):
        raise ValueError("coords_latlon and elevations_m must be the same length")

    start_time = start_time or _dt.datetime.now(_dt.UTC)
    speed_ms = GpxConfig.SPEED_KMH * GpxConfig.METERS_PER_KM / GpxConfig.SECONDS_PER_HOUR

    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack(name=track_name)
    gpx.tracks.append(track)
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    elapsed_s = 0.0
    previous = None
    for (latitude, longitude), elevation in zip(coords_latlon, elevations_m, strict=True):
        if previous is not None:
            distance = haversine_distance_m(lat_a=previous[0], lon_a=previous[1], lat_b=latitude, lon_b=longitude)
            elapsed_s += distance / speed_ms
        point_time = start_time + _dt.timedelta(seconds=elapsed_s)
        segment.points.append(
            gpxpy.gpx.GPXTrackPoint(
                latitude=latitude,
                longitude=longitude,
                elevation=(None if elevation is None else float(elevation)),
                time=point_time,
            )
        )
        previous = (latitude, longitude)

    return gpx.to_xml()
