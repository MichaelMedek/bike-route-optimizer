"""Pure helpers for the Streamlit 3D map viewer (app_webmap.py).

Kept out of the UI shell so the map wiring is unit-testable: ribbon points, the
deck.gl camera for the default/post-route views, and the composition donut chart.
No streamlit imports here — these are pure builders the app shell merely calls.
"""

import math
from dataclasses import dataclass

import altair as alt
import pandas as pd

from bike_router.constants import Mode, Palette, RoadConfig, SurfaceConfig, WebMapConfig
from bike_router.geo import haversine_distance_m
from bike_router.track import Track, TrackPoint, classify_condition


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


@dataclass(frozen=True)
class RibbonSegment:
    """One contiguous run of the route ribbon: colour, width, 3D points, and a hover tooltip."""

    color: list[int]
    width_m: float
    points: list[list[float]]
    tooltip: str


def place_label(*, name: str, elevation_m: float) -> str:
    """``Name (739 m)`` — the ONE marker/tooltip label format (start, end, stations, legs)."""
    return f"{name} ({elevation_m:.0f} m)"


def _segment_tooltip(point: TrackPoint) -> str:
    """Hover text for a pedalled ribbon segment: surface · road · gradient · est. speed."""
    surface = "unpaved" if point.surface_bad else "paved"
    road = "main road" if point.road_bad else "quiet way"
    grade_pct = point.grade * 100
    slope = f"{grade_pct:+.0f}% {'uphill' if grade_pct > 0.5 else 'downhill' if grade_pct < -0.5 else 'flat'}"
    return f"{surface} · {road} · {slope} · ~{point.speed_kmh:.0f} km/h"


def ribbon_width_m(speed_kmh: float) -> float:
    """Ribbon width from speed, INVERSELY (fluid-dynamics): half the speed → double the width.

    Slow spots read as fat/congested pipes, fast spots as thin fast-flowing ones — the eye is
    drawn to the slow parts. Anchored at RIBBON_REF_SPEED_KMH → RIBBON_REF_WIDTH_M.
    """
    assert speed_kmh > 0, "speed must be positive to size the ribbon"
    return WebMapConfig.RIBBON_REF_WIDTH_M * WebMapConfig.RIBBON_REF_SPEED_KMH / speed_kmh


def route_ribbon_segments(
    track: Track,
    float_above_m: float = WebMapConfig.RIBBON_FLOAT_ABOVE_M,
    rail_tooltips: list[str] | None = None,
) -> list[RibbonSegment]:
    """Split the route into contiguous runs sharing one colour + width + tooltip, for rendering.

    Colour comes from segment_color (condition/mode); width from ribbon_width_m (∝ 1/speed, so
    slow spots are wider). A pedalled segment's tooltip describes it (surface/road/gradient/speed);
    a train run shows its whole-leg label from ``rail_tooltips`` (one per train ride, in order).
    Consecutive runs share their boundary point so the ribbon stays continuous.

    Args:
        track: The computed route track from plan_route (points carry mode/condition/speed/grade).
        float_above_m: Metres to lift the ribbon above the terrain mesh.
        rail_tooltips: Whole-leg hover text per train ride (rail_leg_tooltips); None → generic.
    """
    assert len(track.points) >= 2, "ribbon needs at least two points to draw"
    tips = rail_tooltips or []
    segments: list[RibbonSegment] = []
    rail_run = -1  # index into tips; bumped when a fresh train run starts
    prev_mode = None
    for point in track.points[1:]:  # each point is the FAR end of one edge (point 0 has no edge)
        xyz = [point.lon, point.lat, point.elevation_m + float_above_m]
        color = segment_color(mode=point.mode, surface_bad=point.surface_bad, road_bad=point.road_bad)
        width = ribbon_width_m(speed_kmh=point.speed_kmh)
        if point.mode == str(Mode.RAIL):
            if prev_mode != str(Mode.RAIL):
                rail_run += 1
            tooltip = tips[rail_run] if rail_run < len(tips) else "Train"
        else:
            tooltip = _segment_tooltip(point=point)
        prev_mode = point.mode
        # First point of the ribbon anchors the first run; then a colour/width/tooltip change
        # opens a new run, and every point extends the current run (shared seam = continuous).
        if not segments:
            start = track.points[0]
            segments.append(
                RibbonSegment(
                    color=color,
                    width_m=width,
                    points=[[start.lon, start.lat, start.elevation_m + float_above_m]],
                    tooltip=tooltip,
                )
            )
        segments[-1].points.append(xyz)
        if segments[-1].color != color or segments[-1].width_m != width or segments[-1].tooltip != tooltip:
            segments.append(RibbonSegment(color=color, width_m=width, points=[xyz], tooltip=tooltip))
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
