"""Streamlit 3D bike-route viewer — a thin UI shell over bike_router.

Single column, top → bottom: start/end boxes, Fetch button, three routing-penalty
sliders, Compute button, export controls, and a 3D map. All routing logic lives in
bike_router; this file only wires widgets to those functions and renders the map.

Run:  streamlit run app_webmap.py
"""

import tempfile
from pathlib import Path

import networkx as nx
import streamlit as st

from bike_router.constants import DEMConfig, GmapsConfig, RoutingDefaults, RoutingParams, WebMapConfig
from bike_router.corridor import build_corridor
from bike_router.cost import assign_edge_costs
from bike_router.elevation import DEMService, ensure_dem
from bike_router.geocoding import GeocodeError, geocode, make_geocode_fn
from bike_router.gmaps import build_gmaps_url
from bike_router.gpx_export import build_gpx
from bike_router.graph import build_bike_graph, enrich_elevations, snap_endpoints
from bike_router.naming import route_basename
from bike_router.plotting import plot_elevation_heatmap
from bike_router.routing import shortest_route
from bike_router.simplify import route_to_linestring, select_waypoints
from bike_router.track import Track, build_track
from bike_router.webmap import default_view_state, route_ribbon_points, route_view_state
from bike_router.webmap_layers import build_deck


def _init_state() -> None:
    """Seed session_state on first load (nothing fetched/computed yet)."""
    defaults = {
        "graph": None,
        "dem": None,
        "start_latlon": None,
        "end_latlon": None,
        "source": None,
        "target": None,
        "ribbon_points": None,
        "gmaps_url": None,
        "gpx_bytes": None,
        "png_bytes": None,
        "download_stem": None,
        "view": default_view_state(),
        "camera_epoch": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    assert st.session_state.camera_epoch >= 0, "camera_epoch must be a non-negative counter"


def _geocode_both(origin: str, destination: str) -> tuple[dict[str, tuple[float, float]], list[str]]:
    """Geocode start and end independently.

    Returns (resolved, invalid): `resolved` maps "Start"/"End" → (lat, lon) for the
    ones that were found; `invalid` lists the labels that could not be geocoded, so
    the caller can name exactly which field(s) are wrong.
    """
    geocode_fn = make_geocode_fn()
    resolved: dict[str, tuple[float, float]] = {}
    invalid: list[str] = []
    for label, place in (("Start", origin), ("End", destination)):
        try:
            resolved[label] = geocode(place=place, geocode_fn=geocode_fn)
        except GeocodeError:
            invalid.append(f"{label} ({place!r})")
    assert len(resolved) + len(invalid) == 2, "every endpoint must resolve or be flagged invalid"
    return resolved, invalid


def _fetch(start_latlon: tuple[float, float], end_latlon: tuple[float, float]) -> None:
    """Build & prepare the corridor graph from resolved endpoints (params-independent)."""
    dem = DEMService(dem_path=ensure_dem(dem_path=DEMConfig.EURODEM_PATH))
    corridor = build_corridor(start_latlon=start_latlon, dest_latlon=end_latlon)
    graph = build_bike_graph(polygon=corridor)
    source, target = snap_endpoints(graph=graph, start_latlon=start_latlon, dest_latlon=end_latlon)
    enrich_elevations(graph=graph, dem=dem)
    assert graph.number_of_nodes() > 0, "prepared graph must not be empty"
    assert source in graph and target in graph, "snapped endpoints must be graph nodes"

    st.session_state.graph = graph
    st.session_state.dem = dem
    st.session_state.start_latlon = start_latlon
    st.session_state.end_latlon = end_latlon
    st.session_state.source = source
    st.session_state.target = target
    st.session_state.ribbon_points = None  # stale route from a previous fetch
    st.session_state.view = route_view_state(start_latlon=start_latlon, end_latlon=end_latlon)
    st.session_state.camera_epoch += 1


def _png_bytes(graph: nx.MultiDiGraph, route_nodes: list[int], track: Track, params: RoutingParams) -> bytes:
    """Render the debug heatmap to a temp PNG and read it back as bytes."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        out_path = Path(tmp.name)
    plot_elevation_heatmap(graph=graph, route_nodes=route_nodes, track=track, params=params, out_path=str(out_path))
    data = out_path.read_bytes()
    out_path.unlink()
    assert data, "heatmap PNG must not be empty"
    return data


def _compute(origin: str, destination: str, params: RoutingParams) -> None:
    """Cost the cached graph for ``params``, route it, and build all artifacts."""
    graph = st.session_state.graph
    dem = st.session_state.dem
    assert graph is not None and dem is not None, "compute requires a fetched graph + DEM"
    assign_edge_costs(graph=graph, params=params)
    node_path = shortest_route(graph=graph, source=st.session_state.source, target=st.session_state.target)
    assert len(node_path) >= 2, "route must have at least a source and a target node"
    track = build_track(graph=graph, node_path=node_path)

    waypoints = select_waypoints(
        line=route_to_linestring(graph=graph, node_path=node_path), count=GmapsConfig.N_WAYPOINTS
    )

    ribbon_points = route_ribbon_points(graph=graph, node_path=node_path, dem=dem)
    assert len(ribbon_points) >= 2, "ribbon needs at least two points to draw"
    st.session_state.ribbon_points = ribbon_points
    st.session_state.gmaps_url = build_gmaps_url(waypoints_latlon=waypoints)
    st.session_state.gpx_bytes = build_gpx(track=track).encode("utf-8")
    st.session_state.png_bytes = _png_bytes(graph=graph, route_nodes=node_path, track=track, params=params)
    st.session_state.download_stem = route_basename(origin=origin, destination=destination)
    st.session_state.view = route_view_state(
        start_latlon=st.session_state.start_latlon, end_latlon=st.session_state.end_latlon
    )
    st.session_state.camera_epoch += 1


def main() -> None:
    st.set_page_config(page_title="Bike Route Optimizer", layout="centered")
    _init_state()
    st.title("🚲 Bike Route Optimizer")

    # 1. Start | End — the only side-by-side row.
    col_start, col_end = st.columns(2)
    origin = col_start.text_input("Start", value="Freudenstadt, Germany")
    destination = col_end.text_input("End", value="Pforzheim, Germany")

    # 2. Fetch the OSM bike graph for the corridor.
    if st.button("Fetch OpenStreetMap routes", use_container_width=True):
        with st.spinner("Looking up places…"):
            resolved, invalid = _geocode_both(origin=origin, destination=destination)
        if invalid:
            joined = " and ".join(invalid)
            st.toast(f"Could not find {joined}. Check for typos.", icon="⚠️")
            st.error(f"Could not find {joined}. Check the spelling (e.g. add a country, like 'Paris, France').")
        else:
            with st.spinner("Fetching OpenStreetMap data…"):
                _fetch(start_latlon=resolved["Start"], end_latlon=resolved["End"])

    # 3–5. Three independent penalty sliders (no callbacks; do not recompute).
    # Each answers: how many extra km are you willing to ride to avoid one unit of
    # the bad thing? Range 0 → MAX_EXTRA_KM, starting at the tuned default.
    uphill = st.slider(
        "Uphill penalty",
        0.0,
        RoutingDefaults.MAX_EXTRA_KM,
        value=RoutingDefaults.EXTRA_KM_PER_UPHILL_100M,
        step=0.1,
        help="Extra km you'd ride to avoid every 100 m of climbing. "
        "0 = ignore hills; higher = take longer, flatter detours.",
    )
    unpaved = st.slider(
        "Bad road (unpaved) penalty",
        0.0,
        RoutingDefaults.MAX_EXTRA_KM,
        value=RoutingDefaults.EXTRA_KM_PER_UNPAVED_KM,
        step=0.1,
        help="Extra km you'd ride to avoid 1 km of unpaved surface "
        "(counted double on rough/soft ground). 0 = don't mind gravel/dirt.",
    )
    main_road = st.slider(
        "Main road penalty",
        0.0,
        RoutingDefaults.MAX_EXTRA_KM,
        value=RoutingDefaults.EXTRA_KM_PER_MAIN_ROAD_KM,
        step=0.1,
        help="Extra km you'd ride to avoid 1 km on a busy main road (primary/secondary). 0 = don't mind main roads.",
    )

    # 6. Compute the route (only here — never on slider movement).
    needs_fetch = st.session_state.graph is None
    assert 0.0 <= uphill <= RoutingDefaults.MAX_EXTRA_KM, "uphill slider out of range"
    assert 0.0 <= unpaved <= RoutingDefaults.MAX_EXTRA_KM, "unpaved slider out of range"
    assert 0.0 <= main_road <= RoutingDefaults.MAX_EXTRA_KM, "main-road slider out of range"
    if st.button(
        "Compute route",
        use_container_width=True,
        disabled=needs_fetch,
        help="Fetch OpenStreetMap routes first, then adjust the sliders and compute."
        if needs_fetch
        else "Compute the route for the current slider settings.",
    ):
        params = RoutingParams(
            extra_km_per_uphill_100m=uphill,
            extra_km_per_unpaved_km=unpaved,
            extra_km_per_main_road_km=main_road,
        )
        with st.spinner("Computing route…"):
            _compute(origin=origin, destination=destination, params=params)
    if needs_fetch:
        st.caption("⬆️ Click **Fetch OpenStreetMap routes** first to enable **Compute route**.")

    # 7. Export controls (only once a route exists).
    if st.session_state.ribbon_points is not None:
        assert st.session_state.gpx_bytes and st.session_state.png_bytes and st.session_state.gmaps_url, (
            "a computed route must have all three export artifacts"
        )
        st.caption("Open in Google Maps (copy the link):")
        st.code(st.session_state.gmaps_url, language=None)
        stem = st.session_state.download_stem
        col_gpx, col_png = st.columns(2)
        col_gpx.download_button(
            "Download GPX",
            data=st.session_state.gpx_bytes,
            file_name=f"{stem}.gpx",
            mime="application/gpx+xml",
            use_container_width=True,
        )
        col_png.download_button(
            "Download PNG",
            data=st.session_state.png_bytes,
            file_name=f"{stem}.png",
            mime="image/png",
            use_container_width=True,
        )

    # 8. 3D map. camera_epoch in the key remounts deck.gl so the camera reframes.
    deck = build_deck(view=st.session_state.view, ribbon_points=st.session_state.ribbon_points)
    st.pydeck_chart(deck, height=WebMapConfig.MAP_HEIGHT_PX, key=f"bike_map_{st.session_state.camera_epoch}")


if __name__ == "__main__":
    main()
