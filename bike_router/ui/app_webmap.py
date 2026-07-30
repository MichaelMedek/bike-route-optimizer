"""Streamlit 3D bike-route viewer — a thin UI shell over bike_router.

Single column, top → bottom: start/end boxes, routing sliders, "Compute route",
the 3D map, then stats + export controls. ALL routing/geocoding logic lives in
bike_router (what the CLI calls); this file only wires widgets and renders output.

Run:  streamlit run app_webmap.py
"""

import streamlit as st
from streamlit_deckgl import st_deckgl

from bike_router.core.constants import PARAM_SPECS, RoutingDefaults, RoutingParams, WebMapConfig
from bike_router.core.errors import BikeRouterError
from bike_router.core.geocoding import photon_autocomplete
from bike_router.core.graph_store import download_graph_from_hf, load_meta
from bike_router.core.pipeline import plan_route, resolve_endpoints
from bike_router.core.simplify import (
    format_bike_legs,
    format_rail_legs,
    place_label,
    rail_leg_tooltips,
    route_station_markers,
)
from bike_router.ui.webmap import (
    GRADE_DONUT_COLORS,
    GRADE_SCALE,
    MODE_DONUT_COLORS,
    QUALITY_DONUT_COLORS,
    QUALITY_SCALE,
    composition_donut,
    condition_km,
    default_view_state,
    elevation_profile_chart,
    grade_km,
    route_ribbon_segments,
    route_view_state,
)
from bike_router.ui.webmap_layers import build_deck

# Fixed button labels, defined ONCE — referenced by the buttons AND the help/caption text
SET_LABEL = "📍 Set start & end"
COMPUTE_LABEL = "🧭 Compute route"


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


def _render_route_output(result: object) -> None:
    """Route output: stats + donuts in a collapsible box; trains, links, downloads always shown."""
    track = result.track
    # Collapsible: route stats + the three composition donuts — detail to expand, not the
    # primary call to action. Show the bike-vs-total split ONLY when a train is used; a
    # pure-bike route has one set of numbers, so just show "Route".
    with st.expander("📊 Stats & composition", expanded=False):
        stat_rows = (
            (("**Total** (bike + train)", track.total, "Time"), ("**Bike only**", track.bike, "Ride time"))
            if result.rail_legs
            else (("**Route**", track.bike, "Ride time"),)
        )
        for caption, stats, duration_label in stat_rows:
            st.caption(caption)
            pairs = stats.metric_pairs(duration_label=duration_label)
            for col, (label, value) in zip(st.columns(len(pairs)), pairs, strict=True):
                col.metric(label, value)

        comp = result.composition
        # Three donuts: the unified road-QUALITY donut, the road-GRADE donut, and by-mode.
        donuts = (
            ("By quality", condition_km(track=track), QUALITY_DONUT_COLORS),
            ("By grade", grade_km(track=track), GRADE_DONUT_COLORS),
            ("By mode", comp.by_mode_km, MODE_DONUT_COLORS),
        )
        for col, (title, by_km, colors) in zip(st.columns(len(donuts)), donuts, strict=True):
            col.altair_chart(composition_donut(title=title, by_km=by_km, colors=colors), width="stretch")

        # Below the donuts: the elevation profile (distance × elevation), line coloured by mode
        # (bike/train) like the "By mode" donut.
        st.plotly_chart(elevation_profile_chart(track=track), width="stretch")

    # Always visible: which trains to catch, the bike-leg Maps links, and the downloads —
    # the actionable output the rider actually leaves with.
    if result.rail_legs:
        st.caption("🚆 Trains to catch (look these up in your railway app):")
        for line in format_rail_legs(rail_legs=result.rail_legs):
            st.markdown(f"- {line}")

    # One Google Maps bicycling link per pedalled leg (a train ride splits the route into
    # separate legs). The label ("Bike Route N: from → to") is a caption; the code block
    # holds ONLY the URL so its copy-to-clipboard icon copies just the link.
    st.caption("🗺️ Bike legs in Google Maps (one link per leg):")
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
            width="stretch",
        )


def _place_input(*, field: str, label: str, placeholder: str, bbox: tuple[float, float, float, float]) -> str:
    """An editable place box (type/paste freely) with click-to-fill suggestions below it.

    The text_input is the SINGLE source of truth — its exact text is returned and later
    geocoded verbatim. Photon suggestions are a pure convenience: clicking one just fills
    the box (a normal edit the user can still change); typing/pasting anything is fine.
    Suggestions are hidden once the box text matches the last resolved endpoint (set via
    the button), and reappear the moment the user edits the box again.

    Args:
        field: session_state key for this box's text.
        label: visible field label.
        placeholder: greyed-out hint shown when empty.
        bbox: coverage box biasing the suggestions.
    """
    typed = st.text_input(label, key=field, placeholder=placeholder)
    if typed == st.session_state[f"{field}_resolved"]:
        return typed  # already resolved to this text → no stale suggestions under the box
    # Suggestions ranked by Photon relevance (OSM prominence + proximity to the bbox centre);
    # each is a button that fills the box on click. Deduped and index-keyed so a repeated
    # Photon label can't collide on the Streamlit element key. Never required, never blocks.
    seen: set[str] = set()
    for index, suggestion in enumerate(_suggest(term=typed, bbox=bbox)):
        if suggestion == typed or suggestion in seen:
            continue
        seen.add(suggestion)
        st.button(
            f"↳ {suggestion}",
            key=f"{field}_sug_{index}",
            on_click=_fill_box,
            kwargs={"field": field, "value": suggestion},
            width="stretch",
        )
    return typed


def _fill_box(*, field: str, value: str) -> None:
    """Fill a place box with a clicked suggestion (a normal edit; still freely editable)."""
    st.session_state[field] = value


def main() -> None:
    st.set_page_config(page_title="Bike Route Optimizer", layout="centered")
    _download_graph_with_bar()
    # Seed session_state once (nothing set/computed yet). start_latlon gates Compute.
    for key, initial in {
        "start_latlon": None,
        "end_latlon": None,
        "result": None,
        "stations": None,
        "start_box_resolved": None,  # exact box text Set resolved (gates Compute + hides suggestions)
        "end_box_resolved": None,
        "view": default_view_state(),
        "camera_epoch": 0,
    }.items():
        st.session_state.setdefault(key, initial)
    st.title("🚲 Bike Route Optimizer")

    # 1. Start | End — the only side-by-side row. Each is a plain editable box: type,
    # paste, or click a suggestion below it — the box text is the single source of truth
    # and gets geocoded EXACTLY as-is on submit (suggestions are help, never required).
    bbox = tuple(load_meta()["bbox"])  # coverage box biases + limits suggestions
    col_start, col_end = st.columns(2)
    with col_start:
        origin = _place_input(field="start_box", label="Start", placeholder="Start location", bbox=bbox)
    with col_end:
        destination = _place_input(field="end_box", label="End", placeholder="End location", bbox=bbox)

    # 2. Set start & end: resolve_endpoints geocodes the box texts + snaps to the graph;
    # we mark them on the map and recenter. Recentering lives ONLY here (camera_epoch).
    if st.button(
        SET_LABEL,
        width="stretch",
        help="Geocode the Start/End places",
    ):
        try:
            with st.spinner("Looking up places…"):
                start, end = resolve_endpoints(origin=origin, destination=destination)
            st.session_state.update(
                start_latlon=start,  # (lat, lon, elevation_m)
                end_latlon=end,
                start_box_resolved=origin,  # suppress suggestions until the box is edited again
                end_box_resolved=destination,
                result=None,  # stale route from the previous endpoints
                stations=None,
                view=route_view_state(start_latlon=start[:2], end_latlon=end[:2]),
                camera_epoch=st.session_state.camera_epoch + 1,
            )
        except BikeRouterError as error:
            st.toast(str(error), icon="⚠️")

    # The currently-marked endpoints (colors match the map markers and the PNG).
    if st.session_state.start_latlon is not None:
        st.caption(f"🔵 Start: **{origin}**    🔷 End: **{destination}**")

    # 3+. Routing knobs tucked into one collapsible container, straight from the shared
    # PARAM_SPECS (the same source the CLI reads). Range 0 → MAX_EXTRA_KM, each at its default.
    with st.expander("⚙️ Tuning", expanded=False):
        slider_values = {
            spec.field: st.slider(
                spec.label, 0.0, RoutingDefaults.MAX_EXTRA_KM, value=spec.default, step=0.1, help=spec.help
            )
            for spec in PARAM_SPECS
        }

    # 4. Compute the route — draws the ribbon; does NOT recenter (step 2 owns the camera).
    # Gated on the two-button workflow: enabled ONLY when both boxes still hold the EXACT text
    # Set resolved. Editing either box (without re-Setting) disables Compute again.
    endpoints_set = st.session_state.start_latlon is not None
    endpoints_match = (
        endpoints_set
        and origin == st.session_state.start_box_resolved
        and destination == st.session_state.end_box_resolved
    )
    # The three valid states are explicit branches; the else is unreachable (guard).
    if not endpoints_set:
        compute_help = "Set a start and end first"
    elif endpoints_set and not endpoints_match:
        compute_help = f"Start/End changed — press {SET_LABEL} again first"
    elif endpoints_set and endpoints_match:
        compute_help = "Plan the route for the current slider settings"
    else:
        raise AssertionError(f"unreachable compute state: set={endpoints_set} match={endpoints_match}")
    if st.button(COMPUTE_LABEL, width="stretch", disabled=not endpoints_match, help=compute_help):
        try:
            params = RoutingParams(**slider_values)
            with st.spinner("Planning route…"):
                result = plan_route(origin=origin, destination=destination, params=params)
            # No camera_epoch bump → the map keeps the view set in step 2. Store the result; the
            # ribbon is rebuilt at render time from the colour-scale radio (below the map).
            st.session_state.update(
                result=result,
                stations=route_station_markers(rail_legs=result.rail_legs),
            )
        except BikeRouterError as error:  # too short/long, out of coverage, or no route
            st.toast(str(error), icon="⚠️")
    if not endpoints_match:
        st.caption(f"⬆️ Press **{SET_LABEL}** first to enable **{COMPUTE_LABEL}**.")

    _render_map(origin=origin, destination=destination)

    # 6. Stats + export controls BELOW the map, shown once a route exists.
    if st.session_state.result is not None:
        _render_route_output(result=st.session_state.result)


def _render_map(*, origin: str, destination: str) -> None:
    """Render the 3D map: endpoints, the colour-scale radio, and the route ribbon.

    camera_epoch (bumped only by Set start & end) drives the only camera move; the colour
    scale + whether a ribbon exists fold into the remount key so a fresh route or a scale
    toggle shows immediately without moving the view.
    """
    endpoints = (
        (st.session_state.start_latlon, st.session_state.end_latlon)
        if st.session_state.start_latlon is not None
        else None
    )
    endpoint_labels = (
        (
            place_label(name=origin, elevation_m=st.session_state.start_latlon[2]),
            place_label(name=destination, elevation_m=st.session_state.end_latlon[2]),
        )
        if endpoints is not None
        else None
    )
    # Ribbon colour scale: a radio ABOVE the map so its value flows straight into the ribbon build.
    result = st.session_state.result
    color_scale = QUALITY_SCALE
    if result is not None:
        color_scale = st.radio(
            "Ribbon colour",
            options=(QUALITY_SCALE, GRADE_SCALE),
            format_func=lambda s: "Road quality (good / unpaved / main road)"
            if s == QUALITY_SCALE
            else "Road grade (flat / uphill / downhill)",
            key="color_scale",
            horizontal=True,
        )
    ribbon = (
        route_ribbon_segments(
            track=result.track, rail_tooltips=rail_leg_tooltips(rail_legs=result.rail_legs), color_scale=color_scale
        )
        if result is not None
        else None
    )
    deck = build_deck(
        view=st.session_state.view,
        ribbon_segments=ribbon,
        endpoints=endpoints,
        endpoint_labels=endpoint_labels,
        stations=st.session_state.stations,
    )
    map_key = f"bike_map_{st.session_state.camera_epoch}_{color_scale}_{ribbon is not None}"
    st_deckgl(deck, key=map_key, height=WebMapConfig.MAP_HEIGHT_PX)


if __name__ == "__main__":
    main()
