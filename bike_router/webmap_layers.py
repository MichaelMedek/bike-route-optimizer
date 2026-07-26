"""Pydeck layer builders for the 3D bike-route map (app_webmap.py).

Radically simplified from ski-resort-designer's `ui/terrain_layer.py`: 3D only.
A `TerrainLayer` (AWS Terrarium elevation tiles + OpenTopoMap texture, same as the
ski map) plus an optional route ribbon `PathLayer`. No 2D style, no clicks.
"""

from dataclasses import asdict

import pydeck as pdk

from bike_router.constants import WebMapConfig
from bike_router.webmap import RibbonSegment, ViewState


def create_terrain_layer(mesh_max_error: float = 1.0) -> pdk.Layer:
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
    """One pickable PathLayer per contiguous run, in its condition colour + speed-scaled width.

    Each datum carries a ``tooltip`` string so the deck-level tooltip config shows the
    segment's surface/road/gradient/speed (or, for a train run, the whole train leg) on hover.

    Args:
        segments: RibbonSegment runs from webmap.route_ribbon_segments (color RGB, width_m,
            points ``[[lon, lat, z], ...]`` z-lifted, tooltip text).
    """
    return [
        pdk.Layer(
            "PathLayer",
            [{"path": seg.points, "color": seg.color, "tooltip": seg.tooltip}],
            get_path="path",
            get_color="color",
            get_width=seg.width_m,
            width_min_pixels=WebMapConfig.RIBBON_MIN_PIXELS,
            cap_rounded=True,
            joint_rounded=True,
            id=f"route_ribbon_{index}",
            pickable=True,
        )
        for index, seg in enumerate(segments)
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
    """ScatterplotLayer with the start (blue) and end (cyan) markers, each with a name+elev tooltip.

    Each endpoint is ``(lat, lon, elevation_m)`` (snapped to its graph node); the marker hovers
    RIBBON_FLOAT_ABOVE_M above that elevation. Colors match the debug PNG.
    """
    markers = [
        _marker_row(
            lat=start[0], lon=start[1], elev=start[2], color=list(WebMapConfig.START_COLOR), tooltip=start_label
        ),
        _marker_row(lat=end[0], lon=end[1], elev=end[2], color=list(WebMapConfig.END_COLOR), tooltip=end_label),
    ]
    return _marker_layer(
        layer_id="route_endpoints",
        markers=markers,
        radius_m=WebMapConfig.MARKER_RADIUS_M,
        min_pixels=WebMapConfig.MARKER_MIN_PIXELS,
    )


def create_station_layer(stations: list[tuple[float, float, float, str]]) -> pdk.Layer:
    """Rail-coloured markers at each hop-on/hop-off station, smaller than the start/end markers.

    Args:
        stations: ``(lat, lon, elevation_m, label)`` per boarded/alighted station; the label is
            the shared "Name (elev m)" text shown on hover.
    """
    rail = list(WebMapConfig.RAIL_COLOR)
    markers = [
        _marker_row(lat=lat, lon=lon, elev=elev, color=rail, tooltip=label) for lat, lon, elev, label in stations
    ]
    return _marker_layer(
        layer_id="route_stations",
        markers=markers,
        radius_m=WebMapConfig.STATION_MARKER_RADIUS_M,
        min_pixels=WebMapConfig.STATION_MARKER_MIN_PIXELS,
    )


def build_deck(
    view: ViewState,
    ribbon_segments: list[RibbonSegment] | None,
    endpoints: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    endpoint_labels: tuple[str, str] | None = None,
    stations: list[tuple[float, float, float, str]] | None = None,
) -> pdk.Deck:
    """Assemble the Deck bottom→top: terrain, stations, endpoints, then the route ribbon.

    One deck-level tooltip (``{tooltip}``) serves every pickable layer — each ribbon segment,
    endpoint, and station datum carries its own ``tooltip`` string (the proven pydeck idiom).
    """
    layers = [create_terrain_layer()]
    if stations:
        layers.append(create_station_layer(stations=stations))
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
