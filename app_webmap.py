"""Streamlit 3D bike-route viewer — a thin UI shell over bike_router.

Single column, top → bottom: start/end boxes, routing sliders, "Compute route",
the 3D map, then stats + export controls. ALL routing logic lives in
bike_router.plan_route (what the CLI calls); this file only wires widgets.

Run:  streamlit run app_webmap.py
"""

import streamlit as st

from bike_router.constants import PARAM_SPECS, RoutingDefaults, RoutingParams, WebMapConfig
from bike_router.geocoding import GeocodeError, geocode_endpoint, make_geocode_fn
from bike_router.graph_store import download_graph_from_hf, snap_to_node
from bike_router.pipeline import plan_route
from bike_router.webmap import (
    MODE_DONUT_COLORS,
    ROAD_DONUT_COLORS,
    SURFACE_DONUT_COLORS,
    composition_donut,
    default_view_state,
    route_ribbon_segments,
    route_view_state,
)
from bike_router.webmap_layers import build_deck


def _download_graph_with_bar() -> None:
    """One-time prebuilt-graph download with an st.progress bar (the ONLY bar)."""
    bar = st.progress(0.0, text="Downloading map data…")

    def _progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Downloading map data… {done}/{total}")

    download_graph_from_hf(progress=_progress)
    bar.empty()


def main() -> None:
    st.set_page_config(page_title="Bike Route Optimizer", layout="centered")
    _download_graph_with_bar()
    # Seed session_state once (nothing set/computed yet). start_latlon gates Compute.
    for key, initial in {
        "start_latlon": None,
        "end_latlon": None,
        "result": None,
        "ribbon_segments": None,
        "view": default_view_state(),
        "camera_epoch": 0,
    }.items():
        st.session_state.setdefault(key, initial)
    st.title("🚲 Bike Route Optimizer")

    # 1. Start | End — the only side-by-side row.
    col_start, col_end = st.columns(2)
    origin = col_start.text_input("Start", placeholder="Start location")
    destination = col_end.text_input("End", placeholder="End location")

    # 2. Set start & end: geocode both, mark on the map (blue start / cyan end), and
    # recenter. Recentering lives ONLY here (bumps camera_epoch), never on Compute.
    # Start is geocoded first, so a bad Start raises before the End lookup.
    if st.button("📍 Set start & end", use_container_width=True):
        try:
            with st.spinner("Looking up places…"):
                geocode_fn = make_geocode_fn()
                start_ll = geocode_endpoint(place=origin, label="Start", geocode_fn=geocode_fn)
                end_ll = geocode_endpoint(place=destination, label="End", geocode_fn=geocode_fn)
                # Snap to the nearest graph node (routing is node-to-node); this also
                # gives each marker its baked terrain elevation, so it hovers correctly.
                start = snap_to_node(lat=start_ll[0], lon=start_ll[1])
                end = snap_to_node(lat=end_ll[0], lon=end_ll[1])
            st.session_state.update(
                start_latlon=start,  # (lat, lon, elevation_m)
                end_latlon=end,
                result=None,  # stale route from the previous endpoints
                ribbon_segments=None,
                view=route_view_state(start_latlon=start[:2], end_latlon=end[:2]),
                camera_epoch=st.session_state.camera_epoch + 1,
            )
        except GeocodeError as error:
            st.toast(f"Could not find {error}. Check the spelling.", icon="⚠️")

    # The currently-marked endpoints (colors match the map markers and the PNG).
    if st.session_state.start_latlon is not None:
        st.caption(f"🔵 Start: **{origin}**    🔷 End: **{destination}**")

    # 3+. One slider per routing knob, straight from the shared PARAM_SPECS (the same
    # source the CLI reads). Range 0 → MAX_EXTRA_KM, starting at each spec's default.
    slider_values = {
        spec.field: st.slider(
            spec.label, 0.0, RoutingDefaults.MAX_EXTRA_KM, value=spec.default, step=0.1, help=spec.help
        )
        for spec in PARAM_SPECS
    }

    # 4. Compute the route — draws the ribbon; does NOT recenter (step 2 owns the camera).
    needs_endpoints = st.session_state.start_latlon is None
    compute_help = "Set a start and end first" if needs_endpoints else "Plan the route for the current sliders"
    if st.button("🧭 Compute route", use_container_width=True, disabled=needs_endpoints, help=compute_help):
        try:
            params = RoutingParams(**slider_values)
            with st.spinner("Planning route…"):
                result = plan_route(origin=origin, destination=destination, params=params)
            # No camera_epoch bump → the map keeps the view set in step 2.
            st.session_state.update(result=result, ribbon_segments=route_ribbon_segments(track=result.track))
        except (ValueError, RuntimeError) as error:  # too short/long, out of coverage, or no route
            st.toast(str(error), icon="⚠️")
    if needs_endpoints:
        st.caption("⬆️ Set start & end first to enable **Compute route**.")

    # 5. 3D map. camera_epoch (bumped only by Set start & end) keys the remount.
    endpoints = (
        (st.session_state.start_latlon, st.session_state.end_latlon)
        if st.session_state.start_latlon is not None
        else None
    )
    deck = build_deck(view=st.session_state.view, ribbon_segments=st.session_state.ribbon_segments, endpoints=endpoints)
    st.pydeck_chart(deck, height=WebMapConfig.MAP_HEIGHT_PX, key=f"bike_map_{st.session_state.camera_epoch}")

    # 6. Stats + export controls BELOW the map, shown once a route exists.
    result = st.session_state.result
    if result is not None:
        track = result.track
        metrics = (
            ("Distance", f"{track.distance_km:.1f} km"),
            ("Ride time", f"{track.duration_min:.0f} min"),
            ("Ascent", f"+{track.ascent_m:.0f} m"),
            ("Descent", f"−{track.descent_m:.0f} m"),
        )
        for col, (label, value) in zip(st.columns(len(metrics)), metrics, strict=True):
            col.metric(label, value)

        # Composition: three interactive donuts (% by km), side by side across the width.
        comp = result.composition
        donuts = (
            ("By surface", comp.by_surface_km, SURFACE_DONUT_COLORS),
            ("By road", comp.by_road_km, ROAD_DONUT_COLORS),
            ("By mode", comp.by_mode_km, MODE_DONUT_COLORS),
        )
        for col, (title, by_km, colors) in zip(st.columns(len(donuts)), donuts, strict=True):
            col.altair_chart(composition_donut(title=title, by_km=by_km, colors=colors), use_container_width=True)

        st.caption("🗺️ Open in Google Maps (copy the link):")
        st.code(result.gmaps_url, language=None)
        downloads = ((result.gpx_path, "application/gpx+xml"), (result.png_path, "image/png"))
        for col, (path, mime) in zip(st.columns(len(downloads)), downloads, strict=True):
            col.download_button(
                f"Download {path.suffix.lstrip('.').upper()}",
                data=path.read_bytes(),
                file_name=path.name,
                mime=mime,
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
