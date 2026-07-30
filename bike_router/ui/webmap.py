"""Pure helpers for the Streamlit 3D map viewer (app_webmap.py).

Kept out of the UI shell so the map wiring is unit-testable: ribbon points, the
deck.gl camera for the default/post-route views, and the composition donut chart.
No streamlit imports here — these are pure builders the app shell merely calls.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from bike_router.core.constants import Mode, Palette, WebMapConfig
from bike_router.core.geo import haversine_vec
from bike_router.core.simplify import place_label  # single source; re-exported for the app shell
from bike_router.core.track import (
    RouteStats,
    Track,
    TrackPoint,
    classify_condition,
    classify_grade,
    cumulative_km,
    grade_color,
    segment_color,
)

if TYPE_CHECKING:
    from bike_router.core.pipeline import RouteResult

__all__ = ["place_label"]  # noqa: F822 — re-export so app_webmap imports it from here


def _hex(rgb: tuple[int, int, int]) -> str:
    """RGB tuple → ``#rrggbb`` hex (Altair wants hex/CSS colours)."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# Donut label→colour maps, derived from single sources (no re-typed labels): the road-QUALITY
# and road-GRADE scales come from Palette; the mode donut from WebMapConfig.
QUALITY_DONUT_COLORS = {label: Palette.CONDITION_COLORS[label] for label in ("good", "unpaved", "main road")}
GRADE_DONUT_COLORS = {label: Palette.GRADE_COLORS[label] for label in ("flat", "uphill", "downhill")}
MODE_DONUT_COLORS = {label: _hex(rgb=rgb) for label, rgb in WebMapConfig.MODE_DONUT_COLORS.items()}


def _bike_km_by(track: Track, bucket_of: "Callable[[TrackPoint], str]") -> dict[str, float]:
    """Bike km per bucket, ``bucket_of`` mapping each pedalled point to its bucket label.

    Each point (bar the start) is the far end of one edge; its leg length is the great-circle
    gap from the previous point (vectorized once). Rail/station points are skipped. One loop
    serves both the road-QUALITY and road-GRADE donuts (single source, no duplication).
    """
    lats = np.array([p.lat for p in track.points], dtype=np.float64)
    lons = np.array([p.lon for p in track.points], dtype=np.float64)
    leg_km = haversine_vec(lat_a=lats[:-1], lon_a=lons[:-1], lat_b=lats[1:], lon_b=lons[1:]) / 1000.0
    by_km: dict[str, float] = {}
    for point, km in zip(track.points[1:], leg_km, strict=True):
        if point.mode != str(Mode.BIKE):
            continue
        bucket = bucket_of(point)
        by_km[bucket] = by_km.get(bucket, 0.0) + float(km)
    return by_km


def condition_km(track: Track) -> dict[str, float]:
    """Bike km per road-QUALITY bucket (good / unpaved / main road) — the unified donut source.

    Bucketed via classify_condition; "main road" and "main road + unpaved" fold into "main
    road" (the dominant hazard).
    """

    def _bucket(p: TrackPoint) -> str:
        condition = classify_condition(mode=p.mode, surface_bad=p.surface_bad, road_bad=p.road_bad)
        return "main road" if condition in ("main road", "main road + unpaved") else condition

    return _bike_km_by(track, _bucket)


def grade_km(track: Track) -> dict[str, float]:
    """Bike km per road-GRADE bucket (flat / uphill / downhill) via classify_grade (one source)."""
    return _bike_km_by(track, lambda p: classify_grade(mode=p.mode, grade=p.grade))


def composition_donut(title: str, by_km: dict[str, float], colors: dict[str, str]) -> alt.Chart:
    """Small interactive Altair donut of a km breakdown (hover → label / km / %).

    ``colors`` maps each category label to a hex colour so wedges are meaningful
    (blue good / red bad; blue bike / purple train) rather than Altair defaults.
    """
    total = sum(by_km.values())
    assert total > 0, "composition donut needs a non-empty km breakdown"
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


def elevation_profile_chart(track: Track, markers: list[tuple[float, float, str]] | None = None) -> go.Figure:
    """Plotly elevation profile: x = distance (km), y = elevation (m); line coloured bike vs train.

    One Scatter per contiguous MODE run, coloured bike-route blue / train-path purple with the
    SAME MODE_DONUT_COLORS as the "By mode" donut. ``markers`` (distance_km, elevation_m, label)
    from project_markers_onto_track — the SAME named points the map shows (start/end, stations,
    waypoints) — are overlaid so the profile and map agree. Distance is the shared cumulative_km.
    """

    def _label(mode: str) -> str:
        rail, bike = WebMapConfig.MODE_DONUT_LABELS[Mode.RAIL], WebMapConfig.MODE_DONUT_LABELS[Mode.BIKE]
        return rail if mode == str(Mode.RAIL) else bike

    dists = cumulative_km(points=track.points)
    elevs = [p.elevation_m for p in track.points]
    labels = [_label(mode=p.mode) for p in track.points]

    fig = go.Figure()
    seen_legend: set[str] = set()
    run_start = 0
    for i in range(1, len(labels) + 1):
        # Close a run at a mode change or the end; include the boundary point so runs join.
        if i == len(labels) or labels[i] != labels[run_start]:
            label = labels[run_start]
            end = i if i == len(labels) else i + 1  # share the boundary vertex with the next run
            fig.add_trace(
                go.Scatter(
                    x=dists[run_start:end],
                    y=elevs[run_start:end],
                    mode="lines",
                    line={"color": MODE_DONUT_COLORS[label], "width": 2},
                    name=label,
                    legendgroup=label,
                    showlegend=label not in seen_legend,  # one legend entry per mode
                    hovertemplate="%{x:.1f} km · %{y:.0f} m<extra></extra>",
                )
            )
            seen_legend.add(label)
            run_start = i

    if markers:
        # The named waypoints/stations/endpoints, at their correct distance + elevation on the line.
        fig.add_trace(
            go.Scatter(
                x=[d for d, _e, _lab in markers],
                y=[e for _d, e, _lab in markers],
                text=[lab for _d, _e, lab in markers],
                mode="markers+text",
                textposition="top center",
                marker={"size": 8, "color": _hex(rgb=WebMapConfig.RAIL_COLOR)},
                name="waypoints",
                hovertemplate="%{text}<br>%{x:.1f} km · %{y:.0f} m<extra></extra>",
            )
        )

    lo, hi = min(elevs), max(elevs)
    pad = max((hi - lo) * 0.1, 5.0)  # headroom so the line isn't glued to the axes
    fig.update_layout(
        title="Elevation profile",
        height=220,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        plot_bgcolor="white",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.3},
        xaxis={"title": "Distance (km)", "showgrid": True, "gridcolor": "rgba(200,200,200,0.3)"},
        yaxis={
            "title": "Elevation (m)",
            "range": [lo - pad, hi + pad],
            "showgrid": True,
            "gridcolor": "rgba(200,200,200,0.3)",
        },
    )
    return fig


@dataclass(frozen=True)
class RibbonSegment:
    """One contiguous run of the route ribbon: colour, width, 3D points, and a hover tooltip."""

    color: list[int]
    width_m: float
    points: list[list[float]]
    tooltip: str


# The two ribbon colour scales the radio button toggles between.
QUALITY_SCALE = "quality"  # blue good / orange unpaved / red main road
GRADE_SCALE = "grade"  # blue flat / red uphill / green downhill


def _point_color(*, point: TrackPoint, scale: str) -> list[int]:
    """RGB for one point's arriving edge on the chosen scale (both single-sourced in track)."""
    if scale == QUALITY_SCALE:
        return segment_color(mode=point.mode, surface_bad=point.surface_bad, road_bad=point.road_bad)
    elif scale == GRADE_SCALE:
        return grade_color(mode=point.mode, grade=point.grade)
    else:
        raise ValueError(f"unknown ribbon colour scale: {scale!r}")


def _segment_tooltip(point: TrackPoint) -> str:
    """Hover text for a pedalled ribbon segment: surface · road · gradient · est. speed."""
    surface = "unpaved" if point.surface_bad else "paved"
    road = "main road" if point.road_bad else "quiet way"
    grade_pct = point.grade * 100
    if grade_pct > 0.5:
        direction = "uphill"
    elif grade_pct < -0.5:
        direction = "downhill"
    else:
        direction = "flat"
    slope = f"{grade_pct:+.0f}% {direction}"
    return f"{surface} · {road} · {slope} · ~{point.speed_kmh:.0f} km/h"


def ribbon_width_m(speed_kmh: float) -> float:
    """Ribbon width from speed like water in a pipe: flow (area×speed) is conserved.

    Width is the pipe DIAMETER, so cross-section area ∝ width²; conserving area×speed gives
    width ∝ 1/√speed — 4× slower → 2× wider (4× area). Anchored RIBBON_REF_SPEED_KMH → REF_WIDTH.
    """
    assert speed_kmh > 0, "speed must be positive to size the ribbon"
    return WebMapConfig.RIBBON_REF_WIDTH_M * math.sqrt(WebMapConfig.RIBBON_REF_SPEED_KMH / speed_kmh)


def route_ribbon_segments(
    track: Track,
    float_above_m: float = WebMapConfig.RIBBON_FLOAT_ABOVE_M,
    rail_tooltips: list[str] | None = None,
    color_scale: str = QUALITY_SCALE,
) -> list[RibbonSegment]:
    """Split the route into contiguous runs sharing one colour + width + tooltip, for rendering.

    Colour comes from the chosen scale (road-QUALITY or road-GRADE, the radio toggle). BIKE
    segments size their width by effort (∝ 1/√speed, so slow spots are wider); RAIL + STATION
    segments use the fixed RIBBON_REF_WIDTH_M. A pedalled segment's tooltip describes it
    (surface/road/gradient/speed); a train run shows its whole-leg label from ``rail_tooltips``
    (one per train ride, in order). Consecutive runs share their boundary point so the ribbon stays continuous.

    Args:
        track: The computed route track from plan_route (points carry mode/condition/speed/grade).
        float_above_m: Metres to lift the ribbon above the terrain mesh.
        rail_tooltips: Whole-leg hover text per train ride (rail_leg_tooltips); None → generic.
        color_scale: QUALITY_SCALE (blue/orange/red) or GRADE_SCALE (blue/green/red).
    """
    assert len(track.points) >= 2, "ribbon needs at least two points to draw"
    tips = rail_tooltips or []
    segments: list[RibbonSegment] = []
    rail_run = -1  # index into tips; bumped when a fresh train run starts
    prev_mode = None
    for point in track.points[1:]:  # each point is the FAR end of one edge (point 0 has no edge)
        xyz = [point.lon, point.lat, point.elevation_m + float_above_m]
        color = _point_color(point=point, scale=color_scale)
        # Bike segments size by effort (∝ 1/√speed); rail + station segments draw the fixed
        # reference width (a train's pace isn't rider effort; station hops are negligible links).
        pedalled = point.mode == str(Mode.BIKE)
        width = ribbon_width_m(speed_kmh=point.speed_kmh) if pedalled else WebMapConfig.RIBBON_REF_WIDTH_M
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
    """The opening camera: high above the Bodensee (DACH centre), zoomed out, north-up, tilted."""
    return ViewState(
        latitude=WebMapConfig.DEFAULT_LAT,
        longitude=WebMapConfig.DEFAULT_LON,
        zoom=WebMapConfig.DEFAULT_ZOOM,
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
    span_m = float(haversine_vec(lat_a=start_lat, lon_a=start_lon, lat_b=end_lat, lon_b=end_lon))
    return ViewState(
        latitude=(start_lat + end_lat) / 2.0,
        longitude=(start_lon + end_lon) / 2.0,
        zoom=zoom_for_span_m(span_m=span_m),
        pitch=WebMapConfig.DEFAULT_PITCH,
        bearing=WebMapConfig.DEFAULT_BEARING,
    )


# --- pure shell-decision logic (unit-tested here so app_webmap stays thin st.* wiring) --------

# Fixed button labels, defined ONCE — referenced by the buttons AND the help/caption text.
SET_LABEL = "📍 Set start & end"
COMPUTE_LABEL = "🧭 Compute route"


def compute_gate(
    *, start_latlon: object, origin: str, destination: str, start_resolved: object, end_resolved: object
) -> tuple[bool, str]:
    """(compute_enabled, help_text) for the two-button Set→Compute workflow.

    Compute is enabled ONLY when endpoints are set AND both boxes still hold the exact text
    Set resolved; editing either box (without re-Setting) disables it again. The three states
    map to distinct help strings — the whole gate decision, testable without Streamlit.
    """
    endpoints_set = start_latlon is not None
    endpoints_match = endpoints_set and origin == start_resolved and destination == end_resolved
    if not endpoints_set:
        return False, "Set a start and end first"
    elif not endpoints_match:
        return False, f"Start/End changed — press {SET_LABEL} again first"
    else:
        return True, "Plan the route for the current slider settings"


def endpoint_labels(
    *,
    start_latlon: tuple[float, float, float] | None,
    end_latlon: tuple[float, float, float] | None,
    origin: str,
    destination: str,
) -> tuple[str, str] | None:
    """(start, end) marker labels "Name (elev m)", or None when no endpoints are set yet."""
    if start_latlon is None or end_latlon is None:
        return None
    return (
        place_label(name=origin, elevation_m=start_latlon[2]),
        place_label(name=destination, elevation_m=end_latlon[2]),
    )


def map_remount_key(*, camera_epoch: int, color_scale: str, has_ribbon: bool) -> str:
    """The st_deckgl remount key: camera_epoch drives the only camera move; the colour scale +
    whether a ribbon exists fold in so a fresh route or a scale toggle remounts immediately.
    """
    return f"bike_map_{camera_epoch}_{color_scale}_{has_ribbon}"


def scale_label(scale: str) -> str:
    """Human label for a ribbon colour scale (the radio's format_func)."""
    return (
        "Road quality (good / unpaved / main road)"
        if scale == QUALITY_SCALE
        else "Road grade (flat / uphill / downhill)"
    )


def output_stat_rows(result: "RouteResult") -> tuple[tuple[str, RouteStats, str], ...]:
    """(caption, RouteStats, duration_label) rows for the stats panel — the tested selection.

    A train route shows the bike-vs-total split (two rows); a pure-bike route shows one "Route".
    """
    track = result.track
    if result.rail_legs:
        return (
            ("**Total** (bike + train)", track.total, "Time"),
            ("**Bike only**", track.bike, "Ride time"),
        )
    return (("**Route**", track.bike, "Ride time"),)


def output_donuts(result: "RouteResult") -> tuple[tuple[str, dict[str, float], dict[str, str]], ...]:
    """(title, km-breakdown, colours) for the three composition donuts — the tested spec.

    Road-QUALITY (good/unpaved/main), road-GRADE (flat/uphill/downhill), and by-mode — each
    from its single-source km function, so app_webmap merely renders what this returns.
    """
    track = result.track
    return (
        ("By quality", condition_km(track=track), QUALITY_DONUT_COLORS),
        ("By grade", grade_km(track=track), GRADE_DONUT_COLORS),
        ("By mode", result.composition.by_mode_km, MODE_DONUT_COLORS),
    )


def profile_markers(
    *,
    result: "RouteResult",
    start_latlon: tuple[float, float, float],
    end_latlon: tuple[float, float, float],
    start_name: str,
    end_name: str,
    village_of: "Callable[[float, float], str | None]",
) -> list[tuple[float, float, str]]:
    """(distance_km, elevation_m, label) for every named marker on the elevation profile.

    Endpoints use the typed start/end names, stations their station names, and each interior
    gmaps waypoint its nearest-village name via ``village_of`` (a reverse-geocoder; a None result
    drops that marker). Projected onto the track so profile + map agree — one shared assembler.
    """
    from bike_router.core.track import project_markers_onto_track

    markers = [(start_latlon[0], start_latlon[1], start_name), (end_latlon[0], end_latlon[1], end_name)]
    markers += [(lat, lon, label) for lat, lon, _elev, label in _station_marker_points(result=result)]
    markers += [(lat, lon, name) for lat, lon in result.waypoints if (name := village_of(lat, lon))]
    return project_markers_onto_track(track=result.track, markers=markers)


def _station_marker_points(*, result: "RouteResult") -> list[tuple[float, float, float, str]]:
    """Station markers for the route (delegates to the core single source)."""
    from bike_router.core.simplify import route_station_markers

    return route_station_markers(rail_legs=result.rail_legs)
