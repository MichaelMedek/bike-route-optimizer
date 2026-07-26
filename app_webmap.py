"""Streamlit 3D bike-route viewer — a thin UI shell over bike_router.

Single column, top → bottom: start/end boxes, routing sliders, "Compute route",
the 3D map, then stats + export controls. ALL routing logic lives in
bike_router.plan_route (what the CLI calls); this file only wires widgets.

Run:  streamlit run app_webmap.py
"""

import streamlit as st

from bike_router.constants import PARAM_SPECS, RoutingDefaults, RoutingParams, WebMapConfig
from bike_router.geocoding import GeocodeError, geocode_endpoint, make_geocode_fn
from bike_router.graph_store import download_graph_from_hf
from bike_router.pipeline import plan_route
from bike_router.webmap import default_view_state, route_ribbon_segments, route_view_state
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

    # 2. Set start & end: geocode both, mark on the map (green start / red end), and
    # recenter. Recentering lives ONLY here (bumps camera_epoch), never on Compute.
    # Start is geocoded first, so a bad Start raises before the End lookup.
    if st.button("📍 Set start & end", use_container_width=True):
        try:
            with st.spinner("Looking up places…"):
                geocode_fn = make_geocode_fn()
                start_latlon = geocode_endpoint(place=origin, label="Start", geocode_fn=geocode_fn)
                end_latlon = geocode_endpoint(place=destination, label="End", geocode_fn=geocode_fn)
            st.session_state.update(
                start_latlon=start_latlon,
                end_latlon=end_latlon,
                result=None,  # stale route from the previous endpoints
                ribbon_segments=None,
                view=route_view_state(start_latlon=start_latlon, end_latlon=end_latlon),
                camera_epoch=st.session_state.camera_epoch + 1,
            )
        except GeocodeError as error:
            st.toast(f"Could not find {error}. Check the spelling (e.g. add a country like 'Paris, France').", icon="⚠️")

    # The currently-marked endpoints (colors match the map markers and the PNG).
    if st.session_state.start_latlon is not None:
        st.caption(f"🟢 Start: **{origin}**    🔴 End: **{destination}**")

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
    if st.button("Compute route", use_container_width=True, disabled=needs_endpoints):
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

        # Composition as PERCENT of route distance, by surface / road class / mode.
        comp = result.composition
        breakdowns = (("By surface", comp.by_surface_km), ("By road", comp.by_road_km), ("By mode", comp.by_mode_km))
        for col, (label, by_km) in zip(st.columns(len(breakdowns)), breakdowns, strict=True):
            col.caption(label)
            total = sum(by_km.values())
            col.write({k: f"{v / total * 100:.0f}%" for k, v in sorted(by_km.items(), key=lambda kv: -kv[1])})

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
