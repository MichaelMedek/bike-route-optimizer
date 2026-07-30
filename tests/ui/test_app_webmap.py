"""Streamlit app-shell tests — drive app_webmap headlessly via AppTest.

The shell (widgets, the two-button Set→Compute gate, the colour-scale radio, donuts, map)
runs end-to-end with only the NETWORK seams patched (graph download, geocode, plan_route),
so every branch of the UI logic executes without hitting HF/Nominatim or a real dataset.
"""

import contextlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import streamlit_deckgl
from streamlit.testing.v1 import AppTest

from bike_router.core import geocoding, graph_store, pipeline
from bike_router.core.composition import route_composition
from bike_router.core.pipeline import RouteResult
from bike_router.core.simplify import BikeLeg
from bike_router.core.track import build_track
from bike_router.ui import app_webmap
from tests.conftest import make_composition_route, make_line_route

_BBOX = (7.0, 47.0, 10.0, 49.0)


def _result(route) -> RouteResult:
    """A RouteResult over a fixture RoutePath — real track/composition + real (tiny) artifact files.

    The download buttons call ``path.read_bytes()``, so the gpx/png must exist on disk.
    """
    tmp = Path(tempfile.mkdtemp())
    gpx, png = tmp / "r.gpx", tmp / "r.png"
    gpx.write_text("<gpx/>")
    png.write_bytes(b"\x89PNG\r\n")
    return RouteResult(
        track=build_track(route=route),
        gpx_path=gpx,
        png_path=png,
        bike_legs=[BikeLeg(url="https://maps/x", from_place="A", to_place="B")],
        rail_legs=[],
        composition=route_composition(route=route),
    )


@contextlib.contextmanager
def _stubbed(*, route):
    """Patch every network/dataset seam at its SOURCE module.

    AppTest re-imports the app script fresh on each run, so patching the app's local names
    wouldn't stick — we patch bike_router.core.* / the deck component where they're defined,
    which the freshly-imported ``from … import name`` then binds to.
    """
    patches = [
        patch.object(graph_store, "download_graph_from_hf", lambda target_dir=None, progress=None: target_dir),
        patch.object(graph_store, "load_meta", lambda graph_dir=None: {"bbox": list(_BBOX)}),
        patch.object(geocoding, "photon_autocomplete", lambda term, bbox: []),
        patch.object(
            pipeline,
            "resolve_endpoints",
            lambda origin, destination: ((48.0, 8.0, 300.0), (48.4, 8.6, 500.0)),
        ),
        patch.object(pipeline, "plan_route", lambda origin, destination, params: _result(route)),
        patch.object(streamlit_deckgl, "st_deckgl", lambda deck, key, height: None),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


def _run_to_route(at: AppTest) -> AppTest:
    """Type endpoints, press Set, then Compute — leaving a planned route in session_state."""
    at.run()
    at.text_input(key="start_box").set_value("Freudenstadt").run()
    at.text_input(key="end_box").set_value("Pforzheim").run()
    next(b for b in at.button if "Set" in b.label).click().run()
    next(b for b in at.button if "Compute" in b.label).click().run()
    return at


def test_app_boots_and_compute_disabled_before_set():
    """Cold boot renders the title; Compute is disabled until Set resolves the endpoints."""
    with _stubbed(route=make_line_route()):
        at = AppTest.from_file(app_webmap.__file__, default_timeout=30).run()
    assert not at.exception
    assert any("Bike Route Optimizer" in t.value for t in at.title)
    assert next(b for b in at.button if "Compute" in b.label).disabled


def test_set_then_compute_produces_route_output():
    """Set → Compute drives the full flow: endpoints resolve, a route plans, output renders."""
    with _stubbed(route=make_composition_route()):
        at = _run_to_route(AppTest.from_file(app_webmap.__file__, default_timeout=30))
    assert not at.exception
    assert at.session_state["start_latlon"] == (48.0, 8.0, 300.0)
    assert at.session_state["result"] is not None


def test_ribbon_colour_radio_toggles_scale():
    """Once a route exists, the colour-scale radio switches quality ↔ grade without error."""
    with _stubbed(route=make_composition_route()):
        at = _run_to_route(AppTest.from_file(app_webmap.__file__, default_timeout=30))
        at.radio(key="color_scale").set_value(app_webmap.GRADE_SCALE).run()
    assert not at.exception
    assert at.session_state["color_scale"] == app_webmap.GRADE_SCALE
