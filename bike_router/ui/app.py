"""Streamlit 3D bike-route viewer — all app logic (the root app_webmap.py is a main() shell).

Start/end boxes, routing sliders, "Compute route", the 3D map, then stats + export controls. ALL
routing/geocoding logic lives in bike_router.core; this module only wires widgets and renders output.
"""

import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from streamlit_deckgl import st_deckgl
from streamlit_js_eval import get_geolocation

from bike_router.core.constants import (
    LOG_FORMAT,
    PARAM_SPECS,
    ST_PRIMARY,
    ST_SECONDARY,
    START_LABEL,
    GraphConfig,
    PhotonConfig,
    RoutingDefaults,
    RoutingParams,
    SessionKey,
    WebMapConfig,
)
from bike_router.core.errors import BikeRouterError
from bike_router.core.geocoding import (
    autocomplete_with_stations,
    box_display_label,
    default_http_get,
    latlon_box_value,
    nearest_place_name,
)
from bike_router.core.graph_store import download_graph_from_hf, load_meta, top_stations
from bike_router.core.pipeline import RouteResult, plan_route, resolve_endpoints
from bike_router.core.simplify import format_bike_legs, format_rail_legs, rail_leg_tooltips
from bike_router.ui.webmap import (
    COMPUTE_LABEL,
    GRADE_SCALE,
    QUALITY_SCALE,
    SET_LABEL,
    composition_donut,
    compute_gate,
    default_view_state,
    elevation_profile_chart,
    endpoint_labels,
    flattened_view,
    map_click_start_pending,
    map_remount_key,
    map_waypoint_markers,
    output_donuts,
    output_stat_rows,
    profile_markers,
    route_ribbon_segments,
    route_view_state,
    scale_label,
    station_click_pending,
    swapped_endpoint_state,
)
from bike_router.ui.webmap_layers import build_deck

logger = logging.getLogger(__name__)


def download_graph_with_bar() -> None:
    """One-time prebuilt-graph download with an st.progress bar (the ONLY bar)."""
    bar = st.progress(0.0, text="Downloading map data…")

    def _progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Downloading map data… {done}/{total}")

    download_graph_from_hf(target_dir=GraphConfig.GRAPH_DIR, progress=_progress)
    bar.empty()


@st.cache_data(ttl=300)  # type: ignore[misc]  # untyped external decorator (streamlit unstubbed in the mypy env)
def suggest(term: str, bbox: tuple[float, float, float, float]) -> tuple[str | None, list[str]]:
    """Cached Photon suggestions for a typed term: (red-button "<place> Bahnhof" pick, places)."""
    return autocomplete_with_stations(term=term, bbox=bbox, limit=PhotonConfig.LIMIT, http_get=default_http_get)


@st.cache_data(ttl=3600)  # type: ignore[misc]  # untyped external decorator; one cached batch per route
def village_names(waypoints: tuple[tuple[float, float], ...]) -> dict[tuple[float, float], str | None]:
    """Reverse-geocode every gmaps waypoint to its village name CONCURRENTLY (one Photon call each).

    A thread pool collapses ~1-round-trip-per-waypoint to ~1 total; returns a {(lat, lon): name|None}
    map so callers name points with a pure dict lookup, no network in loops.
    """
    with ThreadPoolExecutor(max_workers=max(1, len(waypoints))) as pool:
        names = pool.map(lambda ll: nearest_place_name(lat=ll[0], lon=ll[1], http_get=default_http_get), waypoints)
    return dict(zip(waypoints, names, strict=True))


def village_lookup(result: RouteResult) -> Callable[[float, float], str | None]:
    """A village_of(lat, lon) callable backed by the concurrently-prefetched, cached name map."""
    names = village_names(tuple(result.waypoints))
    return lambda lat, lon: names.get((lat, lon))


def render_route_output(result: RouteResult) -> None:
    """Route output: stats + donuts in a collapsible box; trains, links, downloads always shown."""
    track = result.track
    # Collapsible: route stats + the three composition donuts. Show the bike-vs-total split ONLY
    # when a train is used; a pure-bike route has one set of numbers, so just show "Route".
    with st.expander("📊 Stats & composition", expanded=False):
        for caption, stats, duration_label in output_stat_rows(result):
            st.caption(caption)
            pairs = stats.metric_pairs(duration_label=duration_label)
            for col, (label, value) in zip(st.columns(len(pairs)), pairs, strict=True):
                col.metric(label, value)

        for col, (title, by_km, colors) in zip(st.columns(3), output_donuts(result), strict=True):
            col.altair_chart(composition_donut(title=title, by_km=by_km, colors=colors), width="stretch")

        # Below the donuts: the elevation profile with the SAME named markers the map shows.
        markers = profile_markers(
            result=result,
            start_latlon=st.session_state.start_latlon,
            end_latlon=st.session_state.end_latlon,
            start_name=st.session_state.start_box_resolved,
            end_name=st.session_state.end_box_resolved,
            village_of=village_lookup(result),
        )
        st.plotly_chart(elevation_profile_chart(track=track, markers=markers), width="stretch")

    # Always visible: which trains to catch, the bike-leg Maps links, and the downloads.
    if result.rail_legs:
        st.caption("🚆 Trains to catch (look these up in your railway app):")
        for line in format_rail_legs(rail_legs=result.rail_legs):
            st.markdown(f"- {line}")

    # One Google Maps bicycling link per pedalled leg; the code block holds ONLY the URL so its
    # copy icon copies just the link.
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


def fill_box(field: str, value: str) -> None:
    """Fill a place box with a clicked suggestion (a normal edit; still freely editable)."""
    st.session_state[field] = value


def place_input(field: str, label: str, placeholder: str, bbox: tuple[float, float, float, float]) -> str:
    """An editable place box (type/paste freely) with click-to-fill suggestions below it.

    Returns the box text STRIPPED (the single source of truth, geocoded verbatim); the red station
    pick fills a "lat, lon (Name)" value (exact coords) but shows its readable name, then settlements.
    """
    typed: str = st.text_input(label, key=field, placeholder=placeholder).strip()
    if typed == st.session_state[f"{field}_resolved"]:
        return typed  # already resolved to this text → no stale suggestions under the box
    bahnhof, places = suggest(term=typed, bbox=bbox)
    seen: set[str] = set()
    if bahnhof is not None and bahnhof != typed:
        seen.add(bahnhof)
        st.button(
            f"🚉 {box_display_label(bahnhof)}",  # readable name; the button FILLS the exact-coords value
            key=f"{field}_sug_bahnhof",
            type=ST_PRIMARY,  # red button, first position
            on_click=fill_box,
            kwargs={"field": field, "value": bahnhof},
            width="stretch",
        )
    for index, suggestion in enumerate(places):
        if suggestion == typed or suggestion in seen:
            continue
        seen.add(suggestion)
        st.button(
            f"↳ {suggestion}",
            key=f"{field}_sug_{index}",
            on_click=fill_box,
            kwargs={"field": field, "value": suggestion},
            width="stretch",
        )
    return typed


def _recenter_on_endpoints(start: tuple[float, float, float], end: tuple[float, float, float]) -> None:
    """Reframe the map straight-down on the start→end span and bump the camera epoch (one remount).

    The SINGLE recenter path: both Set (new endpoints) and Compute (fresh route) call this so the
    map always reframes to a fresh top-down view; camera_epoch drives the only camera move.
    """
    st.session_state.update(
        view=route_view_state(start_latlon=start[:2], end_latlon=end[:2]),
        camera_epoch=st.session_state.camera_epoch + 1,
    )


def apply_pending_start(box_value: str) -> None:
    """Stash a Start-box value for the next render, then rerun — the ONE pending-start path.

    A widget key can't be written after its widget renders, so GPS, top-station clicks and map
    clicks all funnel their "lat, lon [(Name)]" box value through here (stash + rerun), no copies.
    """
    st.session_state._pending_start = box_value
    st.rerun()


def set_endpoints() -> None:
    """Set-button callback: geocode the box texts, mark them resolved, recenter the map.

    Runs as an on_click callback (BEFORE the rerun), so marking the boxes resolved clears their
    suggestions instantly. Recentering is the shared _recenter_on_endpoints helper (also used by Compute).
    """
    origin, destination = st.session_state.start_box, st.session_state.end_box
    st.session_state.update(start_box_resolved=origin, end_box_resolved=destination)
    try:
        start, end = resolve_endpoints(origin=origin, destination=destination, graph_dir=GraphConfig.GRAPH_DIR)
    except BikeRouterError as error:
        logger.warning(f"Set endpoints failed to geocode {origin!r} → {destination!r}: {error}")
        st.toast(str(error), icon="⚠️")
        return
    logger.info(f"Set endpoints: {origin!r}={start[:2]} → {destination!r}={end[:2]} (recenter epoch bump)")
    st.session_state.update(
        start_latlon=start,  # (lat, lon, elevation_m)
        end_latlon=end,
        result=None,  # stale route from the previous endpoints
    )
    _recenter_on_endpoints(start=start, end=end)


def toggle_top_stations() -> None:
    """Toggle the rail-purple top-station inspiration markers on the map."""
    shown = not st.session_state.get("show_top_stations", False)
    st.session_state.show_top_stations = shown
    logger.info(f"Top stations {'shown (map flattened for clicks)' if shown else 'hidden'}")


def request_gps() -> None:
    """My-location button callback: arm the GPS read (the component runs on the following render)."""
    st.session_state.gps_requested = True


def arm_map_click_start() -> None:
    """Map-click button callback: TOGGLE 'the next empty-map click sets Start' (red while armed)."""
    st.session_state.arm_map_click_start = not st.session_state.get("arm_map_click_start", False)


def capture_gps() -> None:
    """When armed, read the browser position and stash it as a "lat, lon" literal for the Start box.

    get_geolocation() returns the fix on the rerun AFTER permission is granted, so it runs here in
    the main body (not a callback); the coords flow through the shared apply_pending_start path.
    """
    if not st.session_state.get("gps_requested"):
        return
    location = get_geolocation()
    if location is None:
        return  # still waiting on the browser permission prompt — component reruns when answered
    st.session_state.gps_requested = False
    coords = location.get("coords") if isinstance(location, dict) else None
    if not coords:
        st.toast("Couldn't read your location (permission denied or unavailable).", icon="⚠️")
        return
    lat, lon, accuracy = coords["latitude"], coords["longitude"], coords.get("accuracy", 0.0)
    box_value = latlon_box_value(lat=lat, lon=lon, name=None)  # same box-coord format as picks
    st.toast(f"📍 Location set as Start (±{accuracy:.0f} m accuracy).", icon="📍")
    logger.info(f"GPS fix → Start box {box_value!r} (±{accuracy:.0f} m)")
    apply_pending_start(box_value=box_value)


@st.cache_data(ttl=3600)  # type: ignore[misc]  # untyped external decorator; one whole-graph scan, cached
def top_station_markers() -> list[tuple[float, float, float, str]]:
    """Local-maximum rail stations across the coverage area (cached — a one-off whole-graph scan)."""
    tops = top_stations(graph_dir=GraphConfig.GRAPH_DIR)
    if not tops:
        logger.warning("Top-station scan found no local-maximum rail stations — the map will show none")
    else:
        logger.info(f"Top-station scan: {len(tops)} local-maximum rail stations")
    return tops


def swap_endpoints() -> None:
    """Swap Start ↔ End in one callback (before the rerun, the sanctioned way to mutate widget keys)."""
    st.session_state.update(swapped_endpoint_state(st.session_state.to_dict()))


def configure_logging() -> None:
    """Configure ONLY the bike_router package logger (never root), INFO by default.

    Own StreamHandler + propagate=False, so our level applies to our code alone and Streamlit's root
    handler is untouched; set BIKE_ROUTER_DEBUG=1 for DEBUG. Idempotent across reruns.
    """
    level = logging.DEBUG if os.environ.get("BIKE_ROUTER_DEBUG") == "1" else logging.INFO
    package_logger = logging.getLogger("bike_router")
    package_logger.setLevel(level)
    package_logger.propagate = False  # our level applies to our code alone; root/libraries stay default
    if not package_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        package_logger.addHandler(handler)


def seed_state() -> None:
    """Seed session_state defaults ONCE, then apply any pending Start-box fill (top-station/GPS click).

    A widget key can't be written after its widget renders, so a click stashes into _pending_start;
    we apply it to start_box HERE, before the box is instantiated this run.
    """
    for key, initial in {
        SessionKey.START_LATLON: None,
        SessionKey.END_LATLON: None,
        SessionKey.RESULT: None,
        SessionKey.START_BOX_RESOLVED: None,  # exact box text Set resolved (gates Compute + hides suggestions)
        SessionKey.END_BOX_RESOLVED: None,
        "view": default_view_state(),
        "camera_epoch": 0,
        "show_top_stations": False,  # rail-purple top-station inspiration markers toggle
        "gps_requested": False,  # armed by "My location", read on the next render
        "arm_map_click_start": False,  # armed by the 🎯 button, consumed by the next empty-map click
    }.items():
        st.session_state.setdefault(key, initial)
    if st.session_state.get("_pending_start") is not None:
        pending = st.session_state.pop("_pending_start")
        st.session_state.start_box = pending
        logger.debug(f"Applied pending Start-box fill: {pending!r}")


def render_controls() -> tuple[str, str]:
    """Draw the Start/End boxes (+ swap) and the Set + GPS/map-click/top-stations row; return (origin, dest).

    The three Start-setters are ICON-ONLY and live in one horizontal container so they stay side-by-side
    as a group at any width (never one-per-row on mobile), dropping below Set together when space runs out.
    """
    bbox = tuple(load_meta(graph_dir=GraphConfig.GRAPH_DIR)["bbox"])  # coverage box biases + limits suggestions
    col_start, col_swap, col_end = st.columns([1, 0.18, 1])
    with col_start:
        origin = place_input(field=SessionKey.START_BOX, label=START_LABEL, placeholder="Start location", bbox=bbox)
    with col_swap:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)  # drop below the label
        st.button("⇄", help="Swap Start and End", on_click=swap_endpoints, width="stretch")
    with col_end:
        destination = place_input(field=SessionKey.END_BOX, label="End", placeholder="End location", bbox=bbox)

    # Set is the wide primary; the three Start-setters (GPS / map-click / top-stations) sit in ONE
    # horizontal group that never stacks — beside Set on desktop, dropping below as a unit on mobile.
    # 📍 GPS fills directly; 🎯 and 🚞 are ARM toggles (red while armed) whose next map click sets Start.
    map_armed = st.session_state.get("arm_map_click_start", False)
    tops_armed = st.session_state.get("show_top_stations", False)
    with st.container(horizontal=True, gap="small"):
        st.button(SET_LABEL, width="stretch", help="Geocode the Start/End places", on_click=set_endpoints)
        with st.container(horizontal=True, gap="small", width="content"):
            st.button(
                "📍",
                help="Use my current GPS location as Start (asks the browser for permission)",
                on_click=request_gps,
            )
            st.button(
                "🎯",
                type=ST_PRIMARY if map_armed else ST_SECONDARY,
                help="Arm, then click empty map (top-down) to set Start there; click again to disarm",
                on_click=arm_map_click_start,
            )
            st.button(
                "🚞",
                type=ST_PRIMARY if tops_armed else ST_SECONDARY,
                help="Arm rail-station markers, then click one to start a downhill trip; click again to hide",
                on_click=toggle_top_stations,
            )
    capture_gps()  # if armed by the button, read the browser fix → stash into the Start box (reruns)
    return origin, destination


def compute_button(origin: str, destination: str) -> None:
    """Routing sliders + the Compute button: plan the route on click and store it in session_state."""
    with st.expander("⚙️ Tuning", expanded=False):
        slider_values = {
            spec.field: st.slider(
                spec.label, 0.0, RoutingDefaults.MAX_EXTRA_KM, value=spec.default, step=0.1, help=spec.help
            )
            for spec in PARAM_SPECS
        }

    enabled, compute_help = compute_gate(
        start_latlon=st.session_state.start_latlon,
        origin=origin,
        destination=destination,
        start_resolved=st.session_state.start_box_resolved,
        end_resolved=st.session_state.end_box_resolved,
    )
    if st.button(COMPUTE_LABEL, width="stretch", disabled=not enabled, help=compute_help):
        try:
            params = RoutingParams(**slider_values)
            logger.info(f"Compute route {origin!r} → {destination!r} with {params}")
            with st.spinner("Planning route…"):
                result = plan_route(
                    origin=origin, destination=destination, params=params, graph_dir=GraphConfig.GRAPH_DIR
                )
            st.session_state.update(result=result)
            logger.info(
                f"Route computed: {len(result.track.points)} points, {result.track.total.distance_km:.1f} km, "
                f"{len(result.rail_legs)} rail leg(s)"
            )
            # Reframe fresh straight-down on the computed route — same recenter path as Set.
            _recenter_on_endpoints(start=st.session_state.start_latlon, end=st.session_state.end_latlon)
        except BikeRouterError as error:  # too short/long, out of coverage, or no route
            logger.warning(f"Compute route failed for {origin!r} → {destination!r}: {error}")
            st.toast(str(error), icon="⚠️")
    if not enabled:
        st.caption(f"⬆️ Press **{SET_LABEL}** first to enable **{COMPUTE_LABEL}**.")


def render_map(origin: str, destination: str) -> None:
    """Render the 3D map: endpoints, the colour-scale radio, and the route ribbon.

    camera_epoch (bumped by Set and Compute) drives the only camera move; the colour scale + ribbon presence
    fold into the remount key so a fresh route or scale toggle shows without moving the view.
    """
    endpoints = (
        (st.session_state.start_latlon, st.session_state.end_latlon)
        if st.session_state.start_latlon is not None
        else None
    )
    labels = endpoint_labels(
        start_latlon=st.session_state.start_latlon,
        end_latlon=st.session_state.end_latlon,
        origin=origin,
        destination=destination,
    )
    result = st.session_state.result
    color_scale = QUALITY_SCALE
    if result is not None:
        color_scale = st.radio(
            "Route colour",
            options=(QUALITY_SCALE, GRADE_SCALE),
            format_func=scale_label,
            key="color_scale",
            horizontal=True,
        )
    ribbon = (
        route_ribbon_segments(
            track=result.track,
            float_above_m=WebMapConfig.RIBBON_FLOAT_ABOVE_M,
            rail_tooltips=rail_leg_tooltips(rail_legs=result.rail_legs),
            color_scale=color_scale,
        )
        if result is not None
        else None
    )
    waypoints = map_waypoint_markers(result=result, village_of=village_lookup(result)) if result is not None else None
    tops = top_station_markers() if st.session_state.get("show_top_stations", False) else None
    # deck.gl picking is unreliable under pitch, so WHENEVER a click must be caught (top-station markers
    # shown OR map-click armed) flatten the camera to top-down — the one gate both arm-buttons share.
    top_down = tops is not None or st.session_state.get("arm_map_click_start", False)
    view = flattened_view(st.session_state.view) if top_down else st.session_state.view
    deck = build_deck(
        view=view,
        ribbon_segments=ribbon,
        endpoints=endpoints,
        endpoint_labels=labels,
        waypoints=waypoints,
        top_stations=tops,
    )
    map_key = map_remount_key(
        camera_epoch=st.session_state.camera_epoch, top_down=top_down, has_ribbon=ribbon is not None
    )
    event = st_deckgl(deck, key=map_key, height=WebMapConfig.MAP_HEIGHT_PX, events=["click"])
    handle_top_station_click(event=event)
    handle_map_click_start(event=event)


def handle_top_station_click(event: object) -> None:
    """Stash a clicked top-station's name for the Start box, then rerun (else no-op).

    We must NOT write start_box here (its widget already rendered); it goes through the shared
    apply_pending_start (stash + rerun); the last-applied marker dedups the re-returned event.
    """
    name = station_click_pending(event=event, last_applied=st.session_state.get("_last_station_click"))
    if name is not None:
        logger.info(f"Top-station clicked → filling Start box with {name!r}")
        st.session_state._last_station_click = name
        st.session_state.show_top_stations = False  # auto-disarm: markers hide, button returns to white
        apply_pending_start(box_value=name)


def handle_map_click_start(event: object) -> None:
    """When the map-click button armed it, stash an empty-map click's coords as Start, then rerun.

    Disarms on a hit so only ONE click sets Start; shares the apply_pending_start stash+rerun path,
    and dedups the re-returned event against the last-applied marker (like the top-station handler).
    """
    box_value = map_click_start_pending(
        event=event,
        armed=st.session_state.get("arm_map_click_start", False),
        last_applied=st.session_state.get("_last_map_click"),
    )
    if box_value is not None:
        logger.info(f"Map clicked → filling Start box with {box_value!r}")
        st.session_state._last_map_click = box_value
        st.session_state.arm_map_click_start = False
        apply_pending_start(box_value=box_value)
