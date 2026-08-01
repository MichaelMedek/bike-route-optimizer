"""Pydeck layer builders for the 3D bike-route map.

A `TerrainLayer` (AWS Terrarium elevation tiles + OpenTopoMap texture) plus an optional route
ribbon `PathLayer`, endpoint/waypoint/top-station marker layers. 3D only, no 2D style.
"""

from dataclasses import asdict

import pydeck as pdk

from bike_router.core.constants import WebMapConfig
from bike_router.core.simplify import place_label
from bike_router.ui.webmap import RibbonSegment, ViewState


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


def create_top_station_layer(top_stations: list[tuple[float, float, float, str]]) -> pdk.Layer:
    """Rail-purple clickable markers at local-maximum ("top") rail stations — trip inspiration.

    Each row carries ``name`` (click-to-fill the Start box) + a "Name (elev m)" hover tooltip; clicks
    are read via the deck ``events=['click']`` return.

    Args:
        top_stations: ``(lat, lon, elevation_m, name)`` per local-maximum rail station.
    """
    purple = list(WebMapConfig.RAIL_COLOR)
    markers = [
        {
            **_marker_row(lat=lat, lon=lon, elev=elev, color=purple, tooltip=place_label(name=name, elevation_m=elev)),
            "name": name,
        }
        for lat, lon, elev, name in top_stations
    ]
    return _marker_layer(
        layer_id="top_stations",
        markers=markers,
        radius_m=WebMapConfig.ENDPOINT_RADIUS_M,
        min_pixels=WebMapConfig.ENDPOINT_MIN_PIXELS,
    )


def build_deck(
    view: ViewState,
    ribbon_segments: list[RibbonSegment] | None,
    *,
    endpoints: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    endpoint_labels: tuple[str, str] | None,
    waypoints: list[tuple[float, float, float, str]] | None,
    top_stations: list[tuple[float, float, float, str]] | None,
) -> pdk.Deck:
    """Assemble the Deck bottom→top: terrain, top stations, waypoints, endpoints, then the route ribbon.

    One deck-level tooltip (``{tooltip}``) serves every pickable layer — each ribbon segment,
    endpoint, and waypoint datum carries its own ``tooltip`` string (the proven pydeck idiom).
    """
    layers = [create_terrain_layer(mesh_max_error=1.0)]
    if top_stations:
        layers.append(create_top_station_layer(top_stations=top_stations))
    if waypoints:
        layers.append(create_waypoint_layer(waypoints=waypoints))
    if endpoints is not None:
        # endpoints and endpoint_labels are coupled at the caller (both gate on start_latlon set);
        # a present-endpoints / absent-labels state is drift, so fail loud rather than paint generics.
        assert endpoint_labels is not None, "endpoints set but endpoint_labels missing — coupled-state drift"
        layers.append(
            create_endpoint_layer(
                start=endpoints[0], end=endpoints[1], start_label=endpoint_labels[0], end_label=endpoint_labels[1]
            )
        )
    if ribbon_segments is not None:
        layers.extend(create_route_ribbon_layers(segments=ribbon_segments))
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(**asdict(view)),
        map_provider=None,  # TerrainLayer is the basemap; no Mapbox style needed
        tooltip={"html": "{tooltip}", "style": {"backgroundColor": "rgba(255,255,255,0.95)", "color": "#333"}},
    )
