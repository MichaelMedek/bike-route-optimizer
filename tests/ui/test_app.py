"""app tests — the Streamlit web UI, driven headlessly by streamlit's AppTest (no browser).

One test_<fn> per production symbol (exact-name mirror). AppTest.from_file runs the REAL
app_webmap.py shell (→ bike_router.ui.app.run_app) against the committed fixture graph, so the
whole widget flow executes with zero network; geocode/plan are mocked at the pipeline source.
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

import app_webmap
from bike_router.core import constants
from bike_router.core.composition import route_composition
from bike_router.core.track import build_track
from bike_router.ui import app
from tests.conftest import FIXTURE_GRAPH_DIR, make_line_route

_APP_FILE = str(Path(app_webmap.__file__).resolve())
_START, _END = "Freudenstadt, Germany", "Baiersbronn, Germany"


class _State(dict):
    """A session_state stand-in supporting BOTH item and attribute access (like Streamlit's)."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def _clear_caches() -> None:
    """Drop the app's @st.cache_data results so each run starts from a cold, deterministic cache."""
    app.suggest.clear()
    app.village_names.clear()
    app.top_station_markers.clear()


def _run() -> AppTest:
    """A freshly-run AppTest of the real app, pointed at the committed fixture graph (no network)."""
    _clear_caches()
    harness = AppTest.from_file(_APP_FILE, default_timeout=60)
    harness.run()
    return harness


def _click(harness: AppTest, label: str) -> None:
    """Click the first button whose label matches, then rerun."""
    for button in harness.button:
        if button.label == label:
            button.click().run()
            return
    raise AssertionError(f"no button labelled {label!r} (have {[b.label for b in harness.button]})")


@pytest.fixture
def fixture_graph():
    """Point every default GRAPH_DIR read at the committed fixture graph (no network)."""
    with patch.object(constants.GraphConfig, "GRAPH_DIR", FIXTURE_GRAPH_DIR):
        yield


# --- whole-page render (covers run_app, seed_state, render_controls, download_graph_with_bar, etc.) ---


def test_page_renders(fixture_graph):
    # The whole page (app_webmap.main → the ui.app helpers) renders with no exception: title, both
    # place boxes, the action buttons.
    at = _run()
    assert not at.exception
    assert at.title[0].value == "🚲 Bike Route Optimizer"
    assert len(at.text_input) == 2
    labels = [b.label for b in at.button]
    assert "🔎 Set start & end" in labels and "🚞" in labels and "🧭 Compute route" in labels


def test_download_graph_with_bar():
    # Wraps the HF download in a progress bar; the download is stubbed (no network).
    with patch.object(app, "download_graph_from_hf") as dl, patch.object(app, "st") as fake_st:
        fake_st.progress.return_value = MagicMock()
        app.download_graph_with_bar()
    dl.assert_called_once()


def test_seed_state(fixture_graph):
    # Seeds the gating defaults; a stashed _pending_start is applied to start_box before it renders.
    at = _run()
    assert at.session_state["start_latlon"] is None and at.session_state["camera_epoch"] == 0
    at.session_state["_pending_start"] = "Titisee Bahnhof"
    at.run()
    assert at.session_state["start_box"] == "Titisee Bahnhof" and "_pending_start" not in at.session_state


def test_render_controls(fixture_graph):
    # Draws the two place boxes + the swap and the three distinct Start-setter icons (GPS/map/rail).
    at = _run()
    labels = [b.label for b in at.button]
    assert "⇄" in labels and "📍" in labels and "🎯" in labels and "🚞" in labels


def test_compute_button(fixture_graph):
    # Set → Compute plans a route (mocked planner) and stores it in session_state.
    at = _run()
    track = build_track(route=make_line_route())
    result = SimpleNamespace(
        track=track,
        rail_legs=[],
        bike_legs=[SimpleNamespace(url="https://maps.google/x")],
        waypoints=[],
        composition=route_composition(track=track),
        gpx_path=Path("/tmp/r.gpx"),
        png_path=Path("/tmp/r.png"),
    )
    with (
        patch.object(app, "resolve_endpoints", return_value=((48.0, 8.0, 300.0), (48.02, 8.0, 300.0))),
        patch.object(app, "plan_route", return_value=result) as planned,
    ):
        at.text_input(key="start_box").set_value(_START)
        at.text_input(key="end_box").set_value(_END).run()
        _click(at, "🔎 Set start & end")
        _click(at, "🧭 Compute route")
    planned.assert_called_once()
    assert at.session_state["result"] is result


def test_render_route_output(fixture_graph, tmp_path):
    # After a computed route, the stats/donuts/links/downloads block renders (the bike-legs caption).
    at = _run()
    track = build_track(route=make_line_route())
    gpx, png = tmp_path / "r.gpx", tmp_path / "r.png"
    gpx.write_text("<gpx/>")
    png.write_bytes(b"\x89PNG")
    result = SimpleNamespace(
        track=track,
        rail_legs=[],
        bike_legs=[SimpleNamespace(url="https://maps.google/x", from_place="Start", to_place="End")],
        waypoints=[],
        composition=route_composition(track=track),
        gpx_path=gpx,
        png_path=png,
    )
    with (
        patch.object(app, "resolve_endpoints", return_value=((48.0, 8.0, 300.0), (48.02, 8.0, 300.0))),
        patch.object(app, "plan_route", return_value=result),
    ):
        at.text_input(key="start_box").set_value(_START)
        at.text_input(key="end_box").set_value(_END).run()
        _click(at, "🔎 Set start & end")
        _click(at, "🧭 Compute route")
    assert not at.exception
    assert any("Google Maps" in c.value for c in at.caption)


def test_render_map(fixture_graph):
    # Turning top stations on flattens the camera + renders the map with no exception.
    at = _run()
    _click(at, "🚞")
    assert at.session_state["show_top_stations"] is True and not at.exception


# --- place-input + suggestions ------------------------------------------------


def test_place_input(fixture_graph):
    # Typing a station term surfaces the RED "🚉 <place> Bahnhof" pick FIRST, then settlements.
    at = _run()
    at.text_input(key="start_box").set_value("Freudenstadt").run()
    labels = [b.label for b in at.button]
    assert labels[0].startswith("🚉") and "bahnhof" in labels[0].lower()


def test_fill_box(fixture_graph):
    # Clicking the red Bahnhof suggestion fills the Start box (the fill_box on_click callback).
    at = _run()
    at.text_input(key="start_box").set_value("Freudenstadt").run()
    at.button[0].click().run()
    assert "bahnhof" in at.session_state["start_box"].lower()


def test_suggest():
    # Cached (bahnhof, places) split from the shared geocoding helper; a station term → a Bahnhof pick.
    _clear_caches()
    bahnhof, places = app.suggest("Freudenstadt", (7.5, 47.4, 9.9, 49.8))
    assert bahnhof is not None and "bahnhof" in bahnhof.lower()
    assert isinstance(places, list)


# --- endpoint callbacks -------------------------------------------------------


def test_set_endpoints(fixture_graph):
    # Set geocodes both boxes (mocked on the app module), snaps latlons, bumps the camera epoch.
    at = _run()
    with patch.object(app, "resolve_endpoints", return_value=((48.0, 8.0, 300.0), (48.4, 8.6, 500.0))):
        at.text_input(key="start_box").set_value(_START)
        at.text_input(key="end_box").set_value(_END).run()
        _click(at, "🔎 Set start & end")
    assert at.session_state["start_latlon"] == (48.0, 8.0, 300.0) and at.session_state["camera_epoch"] == 1


def test_set_endpoints_bad_place_toasts(fixture_graph):
    # A geocode failure leaves latlon unset (the error is toasted, not raised).
    at = _run()
    boom = MagicMock(side_effect=app.BikeRouterError("nope"))
    with patch.object(app, "resolve_endpoints", boom):
        at.text_input(key="start_box").set_value("Zzz Nowhere")
        at.text_input(key="end_box").set_value(_END).run()
        _click(at, "🔎 Set start & end")
    assert at.session_state["start_latlon"] is None


def test_swap_endpoints(fixture_graph):
    # ⇄ swaps the two boxes' texts in one callback.
    at = _run()
    at.text_input(key="start_box").set_value("AAA")
    at.text_input(key="end_box").set_value("BBB").run()
    _click(at, "⇄")
    assert at.session_state["start_box"] == "BBB" and at.session_state["end_box"] == "AAA"


def test_toggle_top_stations(fixture_graph):
    # 🚞 flips show_top_stations on, then off.
    at = _run()
    _click(at, "🚞")
    assert at.session_state["show_top_stations"] is True
    _click(at, "🚞")
    assert at.session_state["show_top_stations"] is False


# --- GPS ----------------------------------------------------------------------


def test_request_gps():
    # The button callback just arms the GPS read for the next render.
    with patch.object(app, "st") as fake_st:
        fake_st.session_state = _State()
        app.request_gps()
        assert fake_st.session_state["gps_requested"] is True


def test_capture_gps():
    # When armed, a browser fix is stashed into _pending_start as a "lat, lon" literal, then reruns.
    with patch.object(app, "st") as fake_st, patch.object(app, "get_geolocation") as geo:
        fake_st.session_state = _State(gps_requested=True)
        fake_st.rerun = MagicMock(side_effect=RuntimeError("rerun"))
        geo.return_value = {"coords": {"latitude": 48.4, "longitude": 8.4, "accuracy": 12.0}}
        with pytest.raises(RuntimeError, match="rerun"):
            app.capture_gps()
    assert fake_st.session_state["_pending_start"] == "48.40000, 8.40000"


def test_capture_gps_not_armed():
    # Not armed → no-op (the geolocation component is never rendered).
    with patch.object(app, "st") as fake_st, patch.object(app, "get_geolocation") as geo:
        fake_st.session_state = _State(gps_requested=False)
        app.capture_gps()
    geo.assert_not_called()


# --- top-station click + map helpers -----------------------------------------


def test_handle_top_station_click():
    # A simulated top-station click stashes the "lat, lon (Name Bahnhof)" pending value (exact coords
    # from the marker's position), then reruns.
    fake_state = _State()
    with patch.object(app, "st") as fake_st:
        fake_st.session_state = fake_state
        fake_st.rerun = MagicMock(side_effect=RuntimeError("rerun"))
        with pytest.raises(RuntimeError, match="rerun"):
            app.handle_top_station_click(
                event={
                    "name": "Sauldorf",
                    "position": [9.0, 47.9, 600.0],
                    "eventType": app.WebMapConfig.DECK_CLICK_EVENT,
                }
            )
    assert fake_state["_pending_start"] == "47.90000, 9.00000 (Sauldorf Bahnhof)"


def test_apply_pending_start():
    # The ONE stash+rerun path: writes the box value to _pending_start and reruns (GPS/station/map share it).
    with patch.object(app, "st") as fake_st:
        fake_st.session_state = _State()
        fake_st.rerun = MagicMock(side_effect=RuntimeError("rerun"))
        with pytest.raises(RuntimeError, match="rerun"):
            app.apply_pending_start(box_value="48.0, 8.0")
    assert fake_st.session_state["_pending_start"] == "48.0, 8.0"


def test_recenter_on_endpoints():
    # The ONE recenter path (Set + Compute): sets a fresh view and bumps the camera epoch by one.
    with patch.object(app, "st") as fake_st:
        fake_st.session_state = _State(camera_epoch=2)
        app._recenter_on_endpoints(start=(48.0, 8.0, 300.0), end=(48.4, 8.6, 500.0))
    assert fake_st.session_state["camera_epoch"] == 3 and fake_st.session_state["view"] is not None


def test_arm_map_click_start():
    # The 🎯 button callback just arms the next-empty-map-click flag.
    with patch.object(app, "st") as fake_st:
        fake_st.session_state = _State()
        app.arm_map_click_start()
        assert fake_st.session_state["arm_map_click_start"] is True


def test_handle_map_click_start():
    # Armed + an empty-map click → stash "lat, lon" as Start, disarm, rerun; unarmed → no-op.
    click = app.WebMapConfig.DECK_CLICK_EVENT
    event = {"coordinate": [9.0, 47.9], "eventType": click}
    with patch.object(app, "st") as fake_st:
        fake_st.session_state = _State(arm_map_click_start=True)
        fake_st.rerun = MagicMock(side_effect=RuntimeError("rerun"))
        with pytest.raises(RuntimeError, match="rerun"):
            app.handle_map_click_start(event=event)
    assert fake_st.session_state["_pending_start"] == "47.90000, 9.00000"
    assert fake_st.session_state["arm_map_click_start"] is False
    # unarmed → no stash, no rerun
    with patch.object(app, "st") as fake_st:
        fake_st.session_state = _State(arm_map_click_start=False)
        fake_st.rerun = MagicMock(side_effect=RuntimeError("rerun"))
        app.handle_map_click_start(event=event)  # must NOT raise
    assert "_pending_start" not in fake_st.session_state


def test_top_station_markers(fixture_graph):
    # A cached whole-graph scan → (lat, lon, elev, name) tuples for local-maximum rail stations.
    _clear_caches()
    tops = app.top_station_markers()
    assert isinstance(tops, list)
    assert all(len(t) == 4 and isinstance(t[3], str) for t in tops)


def test_village_names(fixture_graph):
    # Concurrent reverse-geocode → {(lat, lon): name|None}; keys are preserved 1:1.
    pts = ((48.46, 8.41), (48.49, 8.40))
    _clear_caches()
    with patch.object(app, "nearest_place_name", side_effect=["Freudenstadt", None]):
        names = app.village_names(pts)
    assert set(names.keys()) == set(pts)


def test_village_lookup():
    # Builds a village_of(lat, lon) closure over the cached reverse-geocode map.
    result = SimpleNamespace(waypoints=[(48.46, 8.41)])
    with patch.object(app, "village_names", return_value={(48.46, 8.41): "Freudenstadt"}):
        lookup = app.village_lookup(result)
    assert lookup(48.46, 8.41) == "Freudenstadt" and lookup(0.0, 0.0) is None


def test_configure_logging():
    # Configures ONLY the bike_router package logger at INFO (info trail visible, not just warnings),
    # with its own handler + propagate=False; idempotent — reruns don't stack handlers.
    pkg = logging.getLogger("bike_router")
    saved = (pkg.level, pkg.propagate, list(pkg.handlers))
    try:
        pkg.handlers.clear()
        with patch.object(app, "st"):
            app.configure_logging()
            assert pkg.level == logging.INFO and pkg.propagate is False  # info by default, no root cascade
            assert len(pkg.handlers) == 1
            app.configure_logging()  # idempotent: still one handler, not stacked
            assert len(pkg.handlers) == 1
    finally:
        pkg.setLevel(saved[0])
        pkg.propagate = saved[1]
        pkg.handlers[:] = saved[2]
