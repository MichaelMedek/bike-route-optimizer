"""Pure helpers for the Streamlit 3D map viewer (app_webmap.py).

Kept out of the UI shell so the map wiring is unit-testable: ribbon points, the
deck.gl camera for the default/post-route views, and the composition donut chart.
No streamlit imports here — these are pure builders the app shell merely calls.
"""

import math
from dataclasses import dataclass

import altair as alt
import pandas as pd

from bike_router.constants import Palette, RoadConfig, SurfaceConfig, WebMapConfig
from bike_router.geo import haversine_distance_m
from bike_router.track import Track, classify_condition


def _hex(rgb: tuple[int, int, int]) -> str:
    """RGB tuple → ``#rrggbb`` hex (Altair wants hex/CSS colours)."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# Donut label→colour maps, all derived from single sources (no re-typed labels):
# surface/road labels+colours live in SurfaceConfig/RoadConfig; mode in WebMapConfig.
SURFACE_DONUT_COLORS = dict(SurfaceConfig.TIER_LABEL_COLORS.values())
ROAD_DONUT_COLORS = dict(RoadConfig.TIER_LABEL_COLORS.values())
MODE_DONUT_COLORS = {label: _hex(rgb=rgb) for label, rgb in WebMapConfig.MODE_DONUT_COLORS.items()}


def composition_donut(title: str, by_km: dict[str, float], colors: dict[str, str]) -> alt.Chart:
    """Small interactive Altair donut of a km breakdown (hover → label / km / %).

    ``colors`` maps each category label to a hex colour so wedges are meaningful
    (green good / red bad; blue bike / purple train) rather than Altair defaults.
    """
    total = sum(by_km.values()) or 1.0
    frame = pd.DataFrame([{"category": label, "km": km, "pct": km / total * 100} for label, km in by_km.items()])
    domain = list(colors)
    chart: alt.Chart = (
        alt.Chart(frame, title=title)
        .mark_arc(innerRadius=30)
        .encode(
            theta=alt.Theta("km:Q", stack=True),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(domain=domain, range=[colors[label] for label in domain]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                alt.Tooltip("category:N", title="type"),
                alt.Tooltip("km:Q", format=".1f", title="km"),
                alt.Tooltip("pct:Q", format=".0f", title="%"),
            ],
        )
        .properties(height=180)
    )
    return chart


def segment_color(*, mode: str, surface_bad: bool, road_bad: bool) -> list[int]:
    """RGB for a route segment — the single source both the 3D ribbon and PNG use.

    Delegates the branching to classify_condition (one source) and looks the colour up in
    Palette.CONDITION_COLORS, so colour and legend label can never disagree.
    """
    condition = classify_condition(mode=mode, surface_bad=surface_bad, road_bad=road_bad)
    return list(Palette.hex_to_rgb(hex_color=Palette.CONDITION_COLORS[condition]))


def route_ribbon_segments(
    track: Track, float_above_m: float = WebMapConfig.RIBBON_FLOAT_ABOVE_M
) -> list[tuple[list[int], float, list[list[float]]]]:
    """Split the route into contiguous runs of one colour + width for rendering.

    Returns ``(color, width_m, points)`` runs where color comes from segment_color
    (condition/mode) and width_m from the segment speed (RIBBON_WIDTH_PER_KMH_M per
    km/h). Consecutive runs share their boundary point so the ribbon stays continuous.

    Args:
        track: The computed route track from plan_route (points carry mode/condition/speed).
        float_above_m: Metres to lift the ribbon above the terrain mesh.
    """
    assert len(track.points) >= 2, "ribbon needs at least two points to draw"
    segments: list[tuple[list[int], float, list[list[float]]]] = []
    for point in track.points:
        xyz = [point.lon, point.lat, point.elevation_m + float_above_m]
        color = segment_color(mode=point.mode, surface_bad=point.surface_bad, road_bad=point.road_bad)
        width = point.speed_kmh * WebMapConfig.RIBBON_WIDTH_PER_KMH_M
        # Every point extends the current run (bridging the seam keeps the ribbon
        # continuous); a colour OR width change then starts a new run.
        if segments:
            segments[-1][2].append(xyz)
        if not segments or segments[-1][0] != color or segments[-1][1] != width:
            segments.append((color, width, [xyz]))
    return segments


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
