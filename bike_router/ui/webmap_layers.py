"""Pydeck layer builders for the 3D bike-route map (app_webmap.py).

Radically simplified from ski-resort-designer's `ui/terrain_layer.py`: 3D only.
A `TerrainLayer` (AWS Terrarium elevation tiles + OpenTopoMap texture, same as the
ski map) plus an optional route ribbon `PathLayer`. No 2D style, no clicks.
"""

from dataclasses import asdict
from pathlib import Path

import pydeck as pdk

from bike_router.core.constants import WebMapConfig
from bike_router.core.geo import haversine_distance_m
from bike_router.core.route_path import RoutePath
from bike_router.core.track import build_track, densify_track
from bike_router.ui.webmap import RibbonSegment, ViewState, route_ribbon_segments, zoom_for_span_m


def create_terrain_layer(mesh_max_error: float) -> pdk.Layer:
    """3D TerrainLayer: AWS Terrarium elevation meshed, OpenTopoMap texture draped."""
    return pdk.Layer(
        "TerrainLayer",
        elevation_data=WebMapConfig.TERRAIN_TILES_URL,
        elevation_decoder=WebMapConfig.TERRAIN_ELEVATION_DECODER,
        texture=WebMapConfig.TEXTURE_TILES_URL,
        mesh_max_error=mesh_max_error,
        id="terrain_3d",
        pickable=False,
    )


def create_route_ribbon_layers(segments: list[RibbonSegment]) -> list[pdk.Layer]:
    """ONE pickable PathLayer holding every contiguous run as a data row (path/color/width/tooltip).

    A single layer (not one-per-run) so deck.gl picking is uniform across the WHOLE ribbon — stacking
    single-datum PathLayers left only the first few runs hoverable. Accessors read each row's fields.

    Args:
        segments: RibbonSegment runs from webmap.route_ribbon_segments (color RGB, width_m,
            points ``[[lon, lat, z], ...]`` z-lifted, tooltip text).
    """
    data = [{"path": seg.points, "color": seg.color, "width": seg.width_m, "tooltip": seg.tooltip} for seg in segments]
    return [
        pdk.Layer(
            "PathLayer",
            data,
            get_path="path",
            get_color="color",
            get_width="width",
            width_min_pixels=WebMapConfig.RIBBON_MIN_PIXELS,
            cap_rounded=True,
            joint_rounded=True,
            id="route_ribbon",
            pickable=True,
        )
    ]


def _marker_layer(*, layer_id: str, markers: list[dict[str, object]], radius_m: float, min_pixels: int) -> pdk.Layer:
    """A pickable ScatterplotLayer of billboard markers, each row: position/color/tooltip.

    Shared by the start/end markers and the (smaller) station hop-on/hop-off markers. Every
    row's ``tooltip`` feeds the deck-level tooltip config. Drawn on top of the terrain mesh.
    """
    return pdk.Layer(
        "ScatterplotLayer",
        markers,
        get_position="position",
        get_fill_color="color",
        get_radius=radius_m,
        radius_min_pixels=min_pixels,
        stroked=True,
        get_line_color=[0, 0, 0],
        line_width_min_pixels=1,
        billboard=True,  # face the camera in the 3D tilted view
        parameters={"depthTest": False},  # draw ON TOP of the terrain mesh, never buried
        id=layer_id,
        pickable=True,
    )


def _marker_row(*, lat: float, lon: float, elev: float, color: list[int], tooltip: str) -> dict[str, object]:
    """One marker datum, lifted above the terrain like the ribbon (single source of the shape)."""
    return {"position": [lon, lat, elev + WebMapConfig.RIBBON_FLOAT_ABOVE_M], "color": color, "tooltip": tooltip}


def create_endpoint_layer(
    start: tuple[float, float, float], end: tuple[float, float, float], start_label: str, end_label: str
) -> pdk.Layer:
    """ScatterplotLayer with the start + end markers — the ONE blue, slightly bigger than waypoints.

    Each endpoint is ``(lat, lon, elevation_m)`` (snapped to its graph node), hovering above it. All
    markers share MARKER_COLOR (blue); role is told apart by SIZE (endpoints biggest), not colour.
    """
    blue = list(WebMapConfig.MARKER_COLOR)
    markers = [
        _marker_row(lat=start[0], lon=start[1], elev=start[2], color=blue, tooltip=start_label),
        _marker_row(lat=end[0], lon=end[1], elev=end[2], color=blue, tooltip=end_label),
    ]
    return _marker_layer(
        layer_id="route_endpoints",
        markers=markers,
        radius_m=WebMapConfig.ENDPOINT_RADIUS_M,
        min_pixels=WebMapConfig.ENDPOINT_MIN_PIXELS,
    )


def create_waypoint_layer(waypoints: list[tuple[float, float, float, str]]) -> pdk.Layer:
    """Blue round markers at each intermediate point — board/alight STATIONS AND gmaps WAYPOINTS.

    Same blue as the endpoints but SLIGHTLY SMALLER (they read as "along the way"). One layer for
    both kinds — a station and a village waypoint look identical, differing only in hover label.

    Args:
        waypoints: ``(lat, lon, elevation_m, label)`` per intermediate marker; the label is the
            shared "Name (elev m)" text shown on hover.
    """
    blue = list(WebMapConfig.MARKER_COLOR)
    markers = [
        _marker_row(lat=lat, lon=lon, elev=elev, color=blue, tooltip=label) for lat, lon, elev, label in waypoints
    ]
    return _marker_layer(
        layer_id="route_waypoints",
        markers=markers,
        radius_m=WebMapConfig.WAYPOINT_RADIUS_M,
        min_pixels=WebMapConfig.WAYPOINT_MIN_PIXELS,
    )


def build_deck(
    view: ViewState,
    ribbon_segments: list[RibbonSegment] | None,
    endpoints: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    endpoint_labels: tuple[str, str] | None = None,
    waypoints: list[tuple[float, float, float, str]] | None = None,
) -> pdk.Deck:
    """Assemble the Deck bottom→top: terrain, waypoints, endpoints, then the route ribbon.

    One deck-level tooltip (``{tooltip}``) serves every pickable layer — each ribbon segment,
    endpoint, and waypoint datum carries its own ``tooltip`` string (the proven pydeck idiom).
    """
    layers = [create_terrain_layer(mesh_max_error=1.0)]
    if waypoints:
        layers.append(create_waypoint_layer(waypoints=waypoints))
    if endpoints is not None:
        labels = endpoint_labels or ("Start", "End")
        layers.append(
            create_endpoint_layer(start=endpoints[0], end=endpoints[1], start_label=labels[0], end_label=labels[1])
        )
    if ribbon_segments is not None:
        layers.extend(create_route_ribbon_layers(segments=ribbon_segments))
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(**asdict(view)),
        map_provider=None,  # TerrainLayer is the basemap; no Mapbox style needed
        tooltip={"html": "{tooltip}", "style": {"backgroundColor": "rgba(255,255,255,0.95)", "color": "#333"}},
    )


def render_resort_html(
    *,
    routes: list[RoutePath],
    stations_latlon: list[tuple[float, float]],
    out_path: Path,
    lift_markers: list[tuple[float, float, float, str]] | None = None,
) -> None:
    """Write a standalone interactive 3D map (pydeck to_html) of a resort's lift+slope routes.

    Each route → Track → ribbon segments over the terrain deck; ``lift_markers`` places a hoverable
    marker per lift station (label e.g. "Lift: bottom → top"). Camera frames the stations' bbox.

    Args:
        routes: every lift + slope RoutePath to draw.
        stations_latlon: (lat, lon) of the resort's stations, for camera framing.
        out_path: HTML file to write.
        lift_markers: (lat, lon, elev, label) per lift station marker (hover text), or None.
    """
    segments: list[RibbonSegment] = []
    for route in routes:
        segments += route_ribbon_segments(track=densify_track(route=route, track=build_track(route=route)))
    lats = [lat for lat, _lon in stations_latlon]
    lons = [lon for _lat, lon in stations_latlon]
    span_m = haversine_distance_m(lat_a=min(lats), lon_a=min(lons), lat_b=max(lats), lon_b=max(lons)) or 1.0
    view = ViewState(
        latitude=(min(lats) + max(lats)) / 2.0,
        longitude=(min(lons) + max(lons)) / 2.0,
        zoom=zoom_for_span_m(span_m=span_m),
        pitch=WebMapConfig.DEFAULT_PITCH,
        bearing=WebMapConfig.DEFAULT_BEARING,
    )
    deck = build_deck(view=view, ribbon_segments=segments, waypoints=lift_markers)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    deck.to_html(str(out_path), open_browser=False)
