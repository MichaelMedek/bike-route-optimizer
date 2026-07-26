"""Streamlit 3D bike-route viewer — a thin UI shell over bike_router.

Single column, top → bottom: start/end boxes, routing sliders, "Compute route",
the 3D map, then stats + export controls. ALL routing/geocoding logic lives in
bike_router (what the CLI calls); this file only wires widgets and renders output.

Run:  streamlit run app_webmap.py
"""

import functools

import streamlit as st
from streamlit_searchbox import st_searchbox

from bike_router.constants import PARAM_SPECS, PhotonConfig, RoutingDefaults, RoutingParams, WebMapConfig
from bike_router.errors import BikeRouterError
from bike_router.geocoding import photon_autocomplete
from bike_router.graph_store import download_graph_from_hf, load_meta, load_rail_baseline
from bike_router.pipeline import plan_route, resolve_endpoints
from bike_router.simplify import format_bike_legs, format_rail_legs
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


@st.cache_data(ttl=300)
def _suggest(term: str, bbox: tuple[float, float, float, float]) -> list[str]:
    """Cached Photon place-label suggestions for a typed term, biased to the bbox."""
    return photon_autocomplete(term=term, bbox=bbox)


@st.cache_data
def _rail_baseline() -> list[list[list[float]]]:
    """All rail lines (lifted 3D polylines), loaded once — the always-on map baseline."""
    return load_rail_baseline()


def _render_route_output(result: object) -> None:
    """Stats, composition donuts, train legs, Google Maps links, and downloads."""
    track = result.track
    # Two metric rows: the whole journey (bike + train) and the pedalled part only.
    # Both render the SAME four stats via RouteStats.metric_pairs (single source of format).
    stat_rows = (
        ("**Total** (bike + train)", track.total, "Time"),
        ("**Bike only**", track.bike, "Ride time"),
    )
    for caption, stats, duration_label in stat_rows:
        st.caption(caption)
        pairs = stats.metric_pairs(duration_label=duration_label)
        for col, (label, value) in zip(st.columns(len(pairs)), pairs, strict=True):
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

    # Train rides: boarding + alighting station per ride, so the rider can search the
    # actual train in a railway app. Absent for a pure-bike route.
    if result.rail_legs:
        st.caption("🚆 Trains to catch (search these in your railway app):")
        for line in format_rail_legs(rail_legs=result.rail_legs):
            st.markdown(f"- {line}")

    # One bicycling link per pedalled leg; a train ride splits the route into more. The
    # label ("Bike Route N: from → to") is a caption; the code block holds ONLY the URL so
    # its copy-to-clipboard icon copies just the link.
    st.caption("🗺️ Open in Google Maps (one bike leg per train ride):")
    for label, leg in zip(format_bike_legs(bike_legs=result.bike_legs), result.bike_legs, strict=True):
        st.caption(f"**{label}**")
        st.code(leg.url, language=None)

    downloads = ((result.gpx_path, "application/gpx+xml"), (result.png_path, "image/png"))
    for col, (path, mime) in zip(st.columns(len(downloads)), downloads, strict=True):
        col.download_button(
            f"Download {path.suffix.lstrip('.').upper()}",
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            use_container_width=True,
        )


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

    # 1. Start | End — the only side-by-side row. Each is a search-as-you-type box:
    # type freely (CLI-parity); picking a suggestion fills the box with the corrected
    # name. When Photon is slow/unreachable it degrades to a plain box (no suggestions).
    bbox = tuple(load_meta()["bbox"])  # coverage box biases + limits suggestions
    suggest = functools.partial(_suggest, bbox=bbox)
    col_start, col_end = st.columns(2)
    with col_start:
        st_searchbox(suggest, key="start_box", placeholder="Start location", debounce=PhotonConfig.DEBOUNCE_MS)
    with col_end:
        st_searchbox(suggest, key="end_box", placeholder="End location", debounce=PhotonConfig.DEBOUNCE_MS)
    # Whatever text is in each box (typed, or the label of a picked suggestion).
    origin = st.session_state.get("start_box", {}).get("search", "")
    destination = st.session_state.get("end_box", {}).get("search", "")

    # 2. Set start & end: resolve_endpoints geocodes the box texts + snaps to the graph;
    # we mark them on the map and recenter. Recentering lives ONLY here (camera_epoch).
    if st.button("📍 Set start & end", use_container_width=True):
        try:
            with st.spinner("Looking up places…"):
                start, end = resolve_endpoints(origin=origin, destination=destination)
            st.session_state.update(
                start_latlon=start,  # (lat, lon, elevation_m)
                end_latlon=end,
                result=None,  # stale route from the previous endpoints
                ribbon_segments=None,
                view=route_view_state(start_latlon=start[:2], end_latlon=end[:2]),
                camera_epoch=st.session_state.camera_epoch + 1,
            )
        except BikeRouterError as error:
            st.toast(str(error), icon="⚠️")

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
        except BikeRouterError as error:  # too short/long, out of coverage, or no route
            st.toast(str(error), icon="⚠️")
    if needs_endpoints:
        st.caption("⬆️ Set start & end first to enable **Compute route**.")

    # 5. 3D map. camera_epoch (bumped only by Set start & end) keys the remount. The rail
    # baseline (all train lines) always draws, below the route, so trains are visible up front.
    endpoints = (
        (st.session_state.start_latlon, st.session_state.end_latlon)
        if st.session_state.start_latlon is not None
        else None
    )
    deck = build_deck(
        view=st.session_state.view,
        ribbon_segments=st.session_state.ribbon_segments,
        endpoints=endpoints,
        rail_baseline=_rail_baseline(),
    )
    st.pydeck_chart(deck, height=WebMapConfig.MAP_HEIGHT_PX, key=f"bike_map_{st.session_state.camera_epoch}")

    # 6. Stats + export controls BELOW the map, shown once a route exists.
    if st.session_state.result is not None:
        _render_route_output(result=st.session_state.result)


if __name__ == "__main__":
    main()
