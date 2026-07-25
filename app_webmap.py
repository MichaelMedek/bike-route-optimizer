"""Streamlit 3D bike-route viewer — a thin UI shell over bike_router.

Single column, top → bottom: start/end boxes, Compute button, the 3D map, then
route stats + export controls. ALL routing logic lives in bike_router.plan_route
(exactly what the CLI calls); this file only wires widgets and renders the map.

Run:  streamlit run app_webmap.py
"""

import streamlit as st

from bike_router.constants import PARAM_SPECS, DEMConfig, RoutingDefaults, RoutingParams, WebMapConfig
from bike_router.elevation import ensure_dem
from bike_router.geocoding import GeocodeError
from bike_router.pipeline import plan_route
from bike_router.webmap import default_view_state, route_ribbon_points, route_view_state
from bike_router.webmap_layers import build_deck


def _init_state() -> None:
    """Seed session_state on first load (nothing computed yet)."""
    defaults = {
        "result": None,
        "ribbon_points": None,
        "view": default_view_state(),
        "camera_epoch": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    assert st.session_state.camera_epoch >= 0, "camera_epoch must be a non-negative counter"


def main() -> None:
    st.set_page_config(page_title="Bike Route Optimizer", layout="centered")
    _init_state()
    st.title("🚲 Bike Route Optimizer")

    # 1. Start | End — the only side-by-side row.
    col_start, col_end = st.columns(2)
    origin = col_start.text_input("Start", placeholder="e.g. Freudenstadt, Germany")
    destination = col_end.text_input("End", placeholder="e.g. Pforzheim, Germany")

    # 2–4. One slider per routing knob, straight from the shared PARAM_SPECS (the
    # same source the CLI reads). Each answers: how many extra km would you ride to
    # avoid one unit of the bad thing? Range 0 → MAX_EXTRA_KM, at the tuned default.
    slider_values = {
        spec.field: st.slider(
            spec.label, 0.0, RoutingDefaults.MAX_EXTRA_KM, value=spec.default, step=0.1, help=spec.help
        )
        for spec in PARAM_SPECS
    }

    # 5. Compute the route — the app's ONLY job is to call plan_route (like the CLI).
    if st.button("Compute route", use_container_width=True):
        try:
            params = RoutingParams(**slider_values)
            # Two ski-resort-style progress bars: DEM download (bytes) + graph build
            # (nodes contracted). Each shows a real percentage, no fake spinner.
            dem_bar = st.progress(0.0, text="Preparing terrain…")

            def _dem_progress(fraction: float) -> None:
                dem_bar.progress(fraction, text=f"Downloading terrain… {fraction * 100:.0f}%")

            dem_path = ensure_dem(dem_path=DEMConfig.EURODEM_PATH, progress_callback=_dem_progress)
            dem_bar.progress(1.0, text="Terrain ready")

            build_bar = st.progress(0.0, text="Building bike network…")

            def _build_progress(done: int, total: int) -> None:
                frac = done / total if total else 0.0
                build_bar.progress(frac, text=f"Building bike network… {frac * 100:.0f}% ({done}/{total} nodes)")

            result = plan_route(
                origin=origin, destination=destination, dem_path=dem_path, params=params, progress=_build_progress
            )
            dem_bar.empty()
            build_bar.empty()
            st.session_state.result = result
            st.session_state.ribbon_points = route_ribbon_points(track=result.track)
            first, last = result.track.points[0], result.track.points[-1]
            st.session_state.view = route_view_state(
                start_latlon=(first.lat, first.lon), end_latlon=(last.lat, last.lon)
            )
            st.session_state.camera_epoch += 1
        except GeocodeError as error:
            st.toast(f"Could not find {error}. Check the spelling (e.g. add a country like 'Paris, France').", icon="⚠️")
        except SystemExit as error:  # trip too short/long, or no route in corridor
            st.toast(str(error), icon="⚠️")

    # 6. 3D map. camera_epoch in the key remounts deck.gl so the camera reframes.
    deck = build_deck(view=st.session_state.view, ribbon_points=st.session_state.ribbon_points)
    st.pydeck_chart(deck, height=WebMapConfig.MAP_HEIGHT_PX, key=f"bike_map_{st.session_state.camera_epoch}")

    # 7. Stats + export controls BELOW the map (only once a route exists).
    result = st.session_state.result
    if result is not None:
        track = result.track
        col_dist, col_time, col_up, col_down = st.columns(4)
        col_dist.metric("Distance", f"{track.distance_km:.1f} km")
        col_time.metric("Ride time", f"{track.duration_min:.0f} min")
        col_up.metric("Ascent", f"+{track.ascent_m:.0f} m")
        col_down.metric("Descent", f"−{track.descent_m:.0f} m")

        st.caption("Open in Google Maps (copy the link):")
        st.code(result.gmaps_url, language=None)
        col_gpx, col_png = st.columns(2)
        col_gpx.download_button(
            "Download GPX",
            data=result.gpx_path.read_bytes(),
            file_name=result.gpx_path.name,
            mime="application/gpx+xml",
            use_container_width=True,
        )
        col_png.download_button(
            "Download PNG",
            data=result.png_path.read_bytes(),
            file_name=result.png_path.name,
            mime="image/png",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
