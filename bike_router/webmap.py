"""Pure helpers for the Streamlit 3D map viewer (app_webmap.py).

Kept out of the UI shell so the map wiring is unit-testable: turning a computed
Track into ribbon points, and computing the deck.gl camera for the default and
post-route views. No streamlit/pydeck imports here.
"""

import math
from dataclasses import dataclass

from bike_router.constants import WebMapConfig
from bike_router.geo import haversine_distance_m
from bike_router.track import Track


def route_ribbon_points(track: Track, float_above_m: float = WebMapConfig.RIBBON_FLOAT_ABOVE_M) -> list[list[float]]:
    """`[[lon, lat, elevation + float_above_m], ...]` for the route ribbon.

    Reads the Track's points directly (elevations are already finite — the pipeline
    fills DEM nodata during enrich_elevations), lifting each above the terrain mesh.

    Args:
        track: The computed route track from plan_route.
        float_above_m: Metres to lift the ribbon above the terrain mesh.
    """
    assert len(track.points) >= 2, "ribbon needs at least two points to draw"
    return [[point.lon, point.lat, point.elevation_m + float_above_m] for point in track.points]


@dataclass(frozen=True)
class ViewState:
    """A deck.gl camera pose."""

    latitude: float
    longitude: float
    zoom: float
    pitch: float
    bearing: float


def zoom_for_span_m(span_m: float) -> float:
    """Log2 zoom for a span: halving the span adds one zoom level, clamped.

    Args:
        span_m: The extent to frame, in metres.
    """
    assert span_m > 0, "span must be positive"
    raw = WebMapConfig.VIEWING_ZOOM + math.log2(WebMapConfig.ZOOM_SPAN_ANCHOR_M / span_m)
    lo = WebMapConfig.VIEWING_ZOOM - WebMapConfig.ZOOM_STEPS_OUT
    hi = WebMapConfig.VIEWING_ZOOM + WebMapConfig.ZOOM_STEPS_IN
    return max(lo, min(hi, raw))


def default_view_state() -> ViewState:
    """The opening camera: Freudenstadt, north-up, tilted."""
    return ViewState(
        latitude=WebMapConfig.DEFAULT_LAT,
        longitude=WebMapConfig.DEFAULT_LON,
        zoom=WebMapConfig.VIEWING_ZOOM,
        pitch=WebMapConfig.DEFAULT_PITCH,
        bearing=WebMapConfig.DEFAULT_BEARING,
    )


def route_view_state(start_latlon: tuple[float, float], end_latlon: tuple[float, float]) -> ViewState:
    """Camera framing the start→end midpoint, zoom from the direct-line span.

    Args:
        start_latlon: (lat, lon) of the start.
        end_latlon: (lat, lon) of the destination.
    """
    start_lat, start_lon = start_latlon
    end_lat, end_lon = end_latlon
    span_m = haversine_distance_m(lat_a=start_lat, lon_a=start_lon, lat_b=end_lat, lon_b=end_lon)
    return ViewState(
        latitude=(start_lat + end_lat) / 2.0,
        longitude=(start_lon + end_lon) / 2.0,
        zoom=zoom_for_span_m(span_m=span_m),
        pitch=WebMapConfig.DEFAULT_PITCH,
        bearing=WebMapConfig.DEFAULT_BEARING,
    )
