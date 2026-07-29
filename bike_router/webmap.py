"""Pure helpers for the Streamlit 3D map viewer (app_webmap.py).

Kept out of the UI shell so the map wiring is unit-testable: ribbon points, the
deck.gl camera for the default/post-route views, and the composition donut chart.
No streamlit imports here — these are pure builders the app shell merely calls.
"""

import math
from dataclasses import dataclass

import altair as alt
import pandas as pd
import plotly.graph_objects as go

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
    (blue good / red bad; blue bike / purple train) rather than Altair defaults.
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


def elevation_profile_chart(track: Track) -> go.Figure:
    """Plotly elevation profile: x = distance (km), y = elevation (m); line coloured bike vs train.

    A Plotly figure (Altair collapsed to a flat line inside the expander). One Scatter per
    contiguous MODE run, coloured bike-route blue / train-path purple with the SAME
    MODE_DONUT_COLORS as the "By mode" donut (station hops count as bike). Consecutive runs share
    their boundary point so the line stays continuous. Distance is the cumulative haversine gap.
    """

    def _label(mode: str) -> str:
        rail, bike = WebMapConfig.MODE_DONUT_LABELS[Mode.RAIL], WebMapConfig.MODE_DONUT_LABELS[Mode.BIKE]
        return rail if mode == str(Mode.RAIL) else bike

    # Cumulative distance + mode label per point.
    dists, elevs, labels = [], [], []
    dist_km = 0.0
    prev = track.points[0]
    for point in track.points:
        dist_km += haversine_distance_m(lat_a=prev.lat, lon_a=prev.lon, lat_b=point.lat, lon_b=point.lon) / 1000.0
        dists.append(dist_km)
        elevs.append(point.elevation_m)
        labels.append(_label(mode=point.mode))
        prev = point

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
) -> list[RibbonSegment]:
    """Split the route into contiguous runs sharing one colour + width + tooltip, for rendering.

    Colour comes from segment_color (condition/mode). BIKE segments size their width by effort
    (∝ 1/√speed, so slow spots are wider); RAIL + STATION segments use the fixed RIBBON_REF_WIDTH_M.
    A pedalled segment's tooltip describes it (surface/road/gradient/speed); a train run shows its
    whole-leg label from ``rail_tooltips`` (one per train ride, in order). Consecutive runs share
    their boundary point so the ribbon stays continuous.

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
    span_m = haversine_distance_m(lat_a=start_lat, lon_a=start_lon, lat_b=end_lat, lon_b=end_lon)
    return ViewState(
        latitude=(start_lat + end_lat) / 2.0,
        longitude=(start_lon + end_lon) / 2.0,
        zoom=zoom_for_span_m(span_m=span_m),
        pitch=WebMapConfig.DEFAULT_PITCH,
        bearing=WebMapConfig.DEFAULT_BEARING,
    )
