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
    parse_latlon,
)
from bike_router.core.graph_store import download_graph_from_hf, load_meta, snap_to_node
from bike_router.core.pipeline import RouteResult, plan_route, resolve_endpoints
from bike_router.core.rail_extrema import StationExtrema, station_extrema
from bike_router.core.simplify import format_bike_legs, format_rail_legs, rail_leg_tooltips
from bike_router.ui.webmap import (
    COMPUTE_LABEL,
    GRADE_SCALE,
    QUALITY_SCALE,
    composition_donut,
    default_view_state,
    elevation_profile_chart,
    endpoint_markers,
    flattened_view,
    map_click_pending,
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
    """Cached Photon suggestions for a typed term: (red-button "<place> Bahnhof" pick, place box values)."""
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

        donuts = output_donuts(result)
        for col, (title, by_km, colors) in zip(st.columns(len(donuts)), donuts, strict=True):
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

    Returns the box text STRIPPED; every suggestion (station pick + settlements) is a ``"lat, lon (Name)"``
    box string, so a click fills EXACT coords (immediate marker, no re-geocode) while showing the readable name.
    """
    typed: str = st.text_input(label, key=field, placeholder=placeholder).strip()
    if typed == st.session_state[f"{field}_resolved"]:
        return typed  # already resolved to this text → no stale suggestions under the box
    bahnhof, places = suggest(term=typed, bbox=bbox)
    seen: set[str] = set()
    if bahnhof is not None and bahnhof != typed:
        seen.add(bahnhof)
        st.button(
            f"🚉 {box_display_label(value=bahnhof)}",  # readable name; the button FILLS the exact-coords value
            key=f"{field}_sug_bahnhof",
            type=ST_PRIMARY,  # red button, first position
            on_click=fill_box,
            kwargs={"field": field, "value": bahnhof},
            width="stretch",
        )
    for index, box_value in enumerate(places):
        if box_value == typed or box_value in seen:
            continue
        seen.add(box_value)
        st.button(
            f"↳ {box_display_label(value=box_value)}",  # readable name; FILLS the exact-coords box value
            key=f"{field}_sug_{index}",
            on_click=fill_box,
            kwargs={"field": field, "value": box_value},
            width="stretch",
        )
    return typed


def _recenter_on_endpoints(start: tuple[float, float, float], end: tuple[float, float, float]) -> None:
    """Reframe the map straight-down on the start→end span and bump the camera epoch (one remount).

    The ONLY camera move: Compute calls this to reframe on the fresh route. Phase-1 picks never do —
    reconcile_endpoints places markers without touching the view, so the camera stays put until Compute.
    """
    st.session_state.update(
        view=route_view_state(start_latlon=start[:2], end_latlon=end[:2]),
        camera_epoch=st.session_state.camera_epoch + 1,
    )


def apply_pending_box(*, field: str, box_value: str) -> None:
    """Stash a place-box value for the next render, then rerun — the ONE pending-box path.

    A widget key can't be written after its widget renders, so GPS, extremum-station clicks and map
    clicks all funnel their "lat, lon [(Name)]" box value through here (stash + rerun), no copies.
    """
    st.session_state[f"_pending_{field}"] = box_value
    st.rerun()


@st.cache_data(ttl=3600)  # type: ignore[misc]  # untyped external decorator; one cached snap per coords string
def snap_box(coords: str) -> tuple[float, float, float]:
    """Local (no-network) nearest-node snap for a coords box value → (lat, lon, elevation_m), cached.

    Args:
        coords: A "lat, lon [(Name)]" box literal (the exact string, so the cache key is stable).
    """
    latlon = parse_latlon(place=coords)
    assert latlon is not None, f"snap_box needs a coords literal, got {coords!r}"
    return snap_to_node(lat=latlon[0], lon=latlon[1], graph_dir=GraphConfig.GRAPH_DIR)


def reconcile_endpoints() -> None:
    """Phase 1: keep each endpoint's marker in sync with its box, WITHOUT moving the camera.

    A box holding a coords literal (GPS/map/station/suggestion pick) snaps locally to a marker; free
    text clears it (no marker). An off-graph pick toasts and clears. Never bumps camera_epoch — no recenter.
    """
    for box_key, resolved_key, latlon_key in (
        (SessionKey.START_BOX, SessionKey.START_BOX_RESOLVED, SessionKey.START_LATLON),
        (SessionKey.END_BOX, SessionKey.END_BOX_RESOLVED, SessionKey.END_LATLON),
    ):
        box = st.session_state[box_key]
        if box == st.session_state[resolved_key]:
            continue  # unchanged since last reconcile → marker already correct
        coords = parse_latlon(place=box)
        if coords is None:  # free text (or empty) → no immediate marker; Compute resolves it later
            st.session_state.update({latlon_key: None, resolved_key: None, SessionKey.RESULT: None})
            continue
        try:
            snapped = snap_box(coords=box)
        except BikeRouterError as error:  # a pick far from any graph node — fail loud to the user
            logger.warning(f"Reconcile snap failed for {box!r}: {error}")
            st.toast(str(error), icon="⚠️")
            st.session_state.update({latlon_key: None, resolved_key: None})
            continue
        st.session_state.update({latlon_key: snapped, resolved_key: box, SessionKey.RESULT: None})


def toggle_station_extrema() -> None:
    """Toggle the station-extrema markers (green tops → Start, red bottoms → End) on the map."""
    shown = not st.session_state.get("show_station_extrema", False)
    st.session_state.show_station_extrema = shown
    logger.info(f"Station extrema {'shown (map flattened for clicks)' if shown else 'hidden'}")


def request_gps() -> None:
    """My-location button callback: arm the GPS read (the component runs on the following render)."""
    st.session_state.gps_requested = True


def arm_map_click(target: str) -> None:
    """Map-click button callback: arm the next empty-map click to fill ``target`` (toggles off if re-clicked)."""
    current = st.session_state.get("map_click_target")
    st.session_state.map_click_target = None if current == target else target


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
    apply_pending_box(field=SessionKey.START_BOX, box_value=box_value)


@st.cache_data(ttl=3600)  # type: ignore[misc]  # untyped external decorator; one whole-graph scan, cached
def station_extrema_markers() -> StationExtrema:
    """Green local-max + red local-min station markers from ONE cached whole-graph scan."""
    return station_extrema(graph_dir=GraphConfig.GRAPH_DIR)


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
    """Seed session_state defaults ONCE, then apply any pending place-box fill (station/GPS/map click).

    A widget key can't be written after its widget renders, so a click stashes into _pending_<box>;
    we apply it to that box HERE, before the box is instantiated this run.
    """
    for key, initial in {
        SessionKey.START_BOX: "",  # widget keys, pre-seeded so reconcile_endpoints can read them strictly
        SessionKey.END_BOX: "",
        SessionKey.START_LATLON: None,
        SessionKey.END_LATLON: None,
        SessionKey.RESULT: None,
        SessionKey.START_BOX_RESOLVED: None,  # exact box text last resolved (hides suggestions + gates re-snap)
        SessionKey.END_BOX_RESOLVED: None,
        "view": default_view_state(),
        "camera_epoch": 0,
        "show_station_extrema": False,  # green-max (top→Start) / red-min (bottom→End) station markers toggle
        "gps_requested": False,  # armed by "My location", read on the next render
        "map_click_target": None,  # START_BOX / END_BOX armed by 🚩/🏁, consumed by the next empty-map click
    }.items():
        st.session_state.setdefault(key, initial)
    for field in (SessionKey.START_BOX, SessionKey.END_BOX):
        if st.session_state.get(f"_pending_{field}") is not None:
            pending = st.session_state.pop(f"_pending_{field}")
            st.session_state[field] = pending
            logger.debug(f"Applied pending fill to {field}: {pending!r}")


def render_controls() -> tuple[str, str]:
    """Draw the Start/End boxes (+ swap) and the four action buttons; return (origin, destination).

    Reconciles endpoints first so a picked box shows its marker immediately; the four setters share the
    full width in one row — 📍 GPS→Start, 🚩 pick Start / 🏁 pick End on the map, 🚞 station highs/lows.
    """
    reconcile_endpoints()  # phase 1: sync markers to any coords-carrying box, no camera move
    bbox = tuple(load_meta(graph_dir=GraphConfig.GRAPH_DIR)["bbox"])  # coverage box biases + limits suggestions
    col_start, col_swap, col_end = st.columns([1, 0.18, 1])
    with col_start:
        origin = place_input(field=SessionKey.START_BOX, label=START_LABEL, placeholder="Start location", bbox=bbox)
    with col_swap:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)  # drop below the label
        st.button("⇄", help="Swap Start and End", on_click=swap_endpoints, width="stretch")
    with col_end:
        destination = place_input(field=SessionKey.END_BOX, label="End", placeholder="End location", bbox=bbox)

    # The four setters share the full width in one equal-column row. 📍 fills Start from GPS directly;
    # 🚩/🏁 are ARM toggles (red while armed) whose next empty-map click sets Start/End; 🚞 shows stations.
    target = st.session_state.get("map_click_target")
    extrema_armed = st.session_state.get("show_station_extrema", False)
    col_gps, col_pick_start, col_pick_end, col_stations = st.columns(4)
    col_gps.button(
        "📍 My location",
        width="stretch",
        help="Use my current GPS location as Start (asks the browser for permission)",
        on_click=request_gps,
    )
    col_pick_start.button(
        "🚩 Pick start",
        type=ST_PRIMARY if target == SessionKey.START_BOX else ST_SECONDARY,
        width="stretch",
        help="Arm, then click empty map (top-down) to set Start there; click again to disarm",
        on_click=arm_map_click,
        kwargs={"target": SessionKey.START_BOX},
    )
    col_pick_end.button(
        "🏁 Pick end",
        type=ST_PRIMARY if target == SessionKey.END_BOX else ST_SECONDARY,
        width="stretch",
        help="Arm, then click empty map (top-down) to set End there; click again to disarm",
        on_click=arm_map_click,
        kwargs={"target": SessionKey.END_BOX},
    )
    col_stations.button(
        "🚞 Top/Bottom Stations",
        type=ST_PRIMARY if extrema_armed else ST_SECONDARY,
        width="stretch",
        help="Show station highs (green→Start) & lows (red→End); click again to hide",
        on_click=toggle_station_extrema,
    )
    capture_gps()  # if armed by the button, read the browser fix → stash into the Start box (reruns)
    return origin, destination


def compute_button(origin: str, destination: str) -> None:
    """Routing sliders + the Compute button: resolve the boxes, plan the route, recenter (Set folded in).

    Compute is the ONLY phase-2 action: it geocodes any still-free-text box (coords picks parse instantly),
    stores the snapped endpoints, plans, then recenters — enabled whenever both boxes are non-empty.
    """
    with st.expander("⚙️ Tuning", expanded=False):
        slider_values = {
            spec.field: st.slider(
                spec.label, 0.0, RoutingDefaults.MAX_EXTRA_KM, value=spec.default, step=0.1, help=spec.help
            )
            for spec in PARAM_SPECS
        }

    enabled = bool(origin and destination)
    if st.button(COMPUTE_LABEL, width="stretch", disabled=not enabled, help="Plan the route for the current settings"):
        try:
            params = RoutingParams(**slider_values)
            logger.info(f"Compute route {origin!r} → {destination!r} with {params}")
            with st.spinner("Planning route…"):
                start, end = resolve_endpoints(origin=origin, destination=destination, graph_dir=GraphConfig.GRAPH_DIR)
                result = plan_route(
                    origin=origin, destination=destination, params=params, graph_dir=GraphConfig.GRAPH_DIR
                )
            st.session_state.update(
                start_latlon=start,  # (lat, lon, elevation_m) — mark both boxes resolved so markers/suggestions agree
                end_latlon=end,
                start_box_resolved=origin,
                end_box_resolved=destination,
                result=result,
            )
            logger.info(
                f"Route computed: {len(result.track.points)} points, {result.track.total.distance_km:.1f} km, "
                f"{len(result.rail_legs)} rail leg(s)"
            )
            _recenter_on_endpoints(start=start, end=end)  # the ONE camera move
        except BikeRouterError as error:  # bad geocode, too short/long, out of coverage, or no route
            logger.warning(f"Compute route failed for {origin!r} → {destination!r}: {error}")
            st.toast(str(error), icon="⚠️")
    if not enabled:
        st.caption(f"⬆️ Enter a Start and End to enable **{COMPUTE_LABEL}**.")


def render_map(origin: str, destination: str) -> None:
    """Render the 3D map: endpoints, the colour-scale radio, and the route ribbon.

    camera_epoch (bumped only by Compute) drives the one camera move; colour scale, ribbon presence, and
    the endpoint-marker count fold into the remount key so markers/routes show without moving the view.
    """
    endpoints = endpoint_markers(
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
    extrema = station_extrema_markers() if st.session_state.get("show_station_extrema", False) else None
    maxima = extrema.maxima if extrema is not None else None
    minima = extrema.minima if extrema is not None else None
    # deck.gl picking is unreliable under pitch, so WHENEVER a click must be caught (extrema markers
    # shown OR a map-click button armed) flatten the camera to top-down — the one gate the arm-buttons share.
    top_down = extrema is not None or st.session_state.get("map_click_target") is not None
    view = flattened_view(st.session_state.view) if top_down else st.session_state.view
    deck = build_deck(
        view=view,
        ribbon_segments=ribbon,
        endpoints=endpoints,
        waypoints=waypoints,
        maxima=maxima,
        minima=minima,
    )
    map_key = map_remount_key(
        camera_epoch=st.session_state.camera_epoch,
        top_down=top_down,
        has_ribbon=ribbon is not None,
        endpoint_count=len(endpoints),
    )
    event = st_deckgl(deck, key=map_key, height=WebMapConfig.MAP_HEIGHT_PX, events=["click"])
    handle_station_click(event=event)
    handle_map_click(event=event)


def handle_station_click(event: object) -> None:
    """Stash a clicked extremum-station's value into Start (green max) or End (red min), then rerun.

    We must NOT write the box here (its widget already rendered); it goes through the shared
    apply_pending_box (stash + rerun); markers stay shown so both ends can be picked in one arming.
    """
    pending = station_click_pending(event=event, last_applied=st.session_state.get("_last_station_click"))
    if pending is not None:
        box_value, field = pending
        logger.info(f"Station clicked → filling {field} with {box_value!r}")
        st.session_state._last_station_click = box_value
        apply_pending_box(field=field, box_value=box_value)


def handle_map_click(event: object) -> None:
    """When a map-click button armed a target box, stash an empty-map click's coords there, then rerun.

    Disarms on a hit so only ONE click sets the box; shares the apply_pending_box stash+rerun path,
    and dedups the re-returned event against the last-applied marker (like the station handler).
    """
    target = st.session_state.get("map_click_target")
    if target is None:
        return  # no 🚩/🏁 armed — an empty-map click sets nothing
    box_value = map_click_pending(
        event=event,
        target=target,
        last_applied=st.session_state.get("_last_map_click"),
    )
    if box_value is not None:
        logger.info(f"Map clicked → filling {target} with {box_value!r}")
        st.session_state._last_map_click = box_value
        st.session_state.map_click_target = None
        apply_pending_box(field=target, box_value=box_value)
