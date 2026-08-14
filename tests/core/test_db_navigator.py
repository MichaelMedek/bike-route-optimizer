"""Unit tests for the bahn.de deep-link builder (db_navigator).

Every DB call goes through an injectable stub getter (url, body, timeout) → JSON, so nothing hits the
network; the degrade paths use a stub that raises a urllib error.
"""

import datetime
import os
import urllib.error

import pytest

from bike_router.core import db_navigator
from bike_router.core.constants import DbNavigatorConfig
from bike_router.core.db_navigator import (
    DbGetter,
    _leg_products,
    build_bahn_leg,
    build_label,
    build_link,
    default_db_get,
    resolve_halt,
    verify_connection,
)

# A resolved (station_id, name, eva) triple as orte returns it (EVA embedded as "@L=...@").
_BERLIN_ID = "A=1@O=Berlin Hbf@X=13369549@Y=52525589@U=80@L=8011160@p=1@i=U×008065969@"
_MUNICH_ID = "A=1@O=München Hbf@X=11558339@Y=48140229@U=80@L=8000261@p=1@i=U×008020347@"
_WHEN = datetime.datetime(2026, 8, 20, 8, 30, 0)


def _orte_hit(station_id: str, name: str) -> list[dict[str, str]]:
    """A one-element orte reply for `name`."""
    return [{"id": station_id, "name": name}]


def _connection(products: list[tuple[str, str]]) -> dict[str, object]:
    """A journey connection whose PUBLICTRANSPORT legs carry the given (mittelText, produktGattung)."""
    return {
        "verbindungsAbschnitte": [
            {
                "abfahrt": {"sollzeit": "2026-08-20T08:44:00"},
                "ankunft": {"sollzeit": "2026-08-20T09:15:00"},
                "verkehrsmittel": {"typ": "PUBLICTRANSPORT", "mittelText": mittel, "produktGattung": gattung},
            }
            for mittel, gattung in products
        ]
    }


# --- DB getter seam ----------------------------------------------------------


class TestDbGetter:
    def test_protocol_is_satisfied_by_a_keyword_callable(self):
        # The seam is a keyword-callable (url, body, timeout) → JSON; a plain fn implements it.
        def getter(*, url: str, body: dict[str, object] | None, timeout: float) -> object:
            return {"ok": url, "posted": body is not None}

        fn: DbGetter = getter  # a conforming callable type-checks as the Protocol
        assert fn(url="https://x", body=None, timeout=1.0) == {"ok": "https://x", "posted": False}


def test_default_db_get(monkeypatch):
    # body=None → GET (no data); body given → POST with JSON-encoded data. Parses the JSON reply.
    import urllib.request

    captured: list[urllib.request.Request] = []

    class _Resp:
        def __init__(self, payload: bytes):
            self._payload = payload
            self.headers: dict[str, str] = {}

        def read(self) -> bytes:
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        captured.append(request)
        payload = b'{"verbindungen": []}' if request.data is not None else b'["got"]'
        return _Resp(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(db_navigator.time, "sleep", lambda _s: None)  # no real inter-call wait offline

    assert default_db_get(url="https://o", body=None, timeout=2.0) == ["got"]
    assert default_db_get(url="https://j", body={"a": 1}, timeout=2.0) == {"verbindungen": []}
    get_request, post_request = captured
    assert get_request.data is None and post_request.data == b'{"a": 1}'
    assert get_request.headers["User-agent"] == DbNavigatorConfig.USER_AGENT  # urllib title-cases header keys


# --- resolve ----------------------------------------------------------------


def test_resolve_halt():
    # An EVA-bearing top hit → (id VERBATIM, name, eva); a hit without "@L=" or an empty reply → None.
    stub: DbGetter = lambda *, url, body, timeout: _orte_hit(_BERLIN_ID, "Berlin Hbf")  # noqa: E731
    assert resolve_halt(name="Berlin Hbf", http_get=stub) == (_BERLIN_ID, "Berlin Hbf", "8011160")

    no_eva: DbGetter = lambda *, url, body, timeout: [{"id": "A=1@O=x@", "name": "x"}]  # noqa: E731
    assert resolve_halt(name="x", http_get=no_eva) is None

    empty: DbGetter = lambda *, url, body, timeout: []  # noqa: E731
    assert resolve_halt(name="nowhere", http_get=empty) is None


# --- connection products / label --------------------------------------------


def test_leg_products():
    # Only PUBLICTRANSPORT legs count; a walk segment (no such typ) is skipped.
    connection = _connection([("RB33", "REGIONAL"), ("S1", "SBAHN")])
    connection["verbindungsAbschnitte"].append({"verkehrsmittel": {"typ": "WALK"}})
    assert _leg_products(connection=connection) == [("RB33", "REGIONAL"), ("S1", "SBAHN")]


def test_build_label():
    # ≤2 legs show all; >2 elide to "first › … › last". Time is first dep + last arr.
    two = _connection([("RB33", "REGIONAL"), ("RE4", "REGIONAL")])
    assert build_label(connection=two) == "08:44 → 09:15 · RB33 › RE4"
    many = _connection([("S3", "SBAHN"), ("S5", "SBAHN"), ("MEX17", "REGIONAL"), ("S1", "SBAHN")])
    assert build_label(connection=many) == "08:44 → 09:15 · S3 › … › S1"


# --- verify -----------------------------------------------------------------


def test_verify_connection():
    # POSTs to the journey endpoint and returns the first connection; None when the reply has none.
    connection = _connection([("RB33", "REGIONAL")])
    hit: DbGetter = lambda *, url, body, timeout: {"verbindungen": [connection]}  # noqa: E731
    assert verify_connection(board_eva="8011160", alight_eva="8000261", when=_WHEN, http_get=hit) is connection

    none: DbGetter = lambda *, url, body, timeout: {"verbindungen": []}  # noqa: E731
    assert verify_connection(board_eva="8011160", alight_eva="8000261", when=_WHEN, http_get=none) is None


# --- link -------------------------------------------------------------------


def test_build_link():
    # Hash-fragment deep link: the essential station ids, the vm regional filter, ISO dep time, hza=D.
    board = (_BERLIN_ID, "Berlin Hbf", "8011160")
    alight = (_MUNICH_ID, "München Hbf", "8000261")
    url = build_link(board=board, alight=alight, when=_WHEN)
    assert url.startswith(f"{DbNavigatorConfig.SUCHE_URL}#")
    assert f"vm={DbNavigatorConfig.VM_CODES.replace(',', '%2C')}" in url or "vm=03" in url
    assert "hza=D" in url and "hd=2026-08-20T08%3A30%3A00" in url
    assert "soid=" in url and "zoid=" in url


# --- compose (build_bahn_leg) -----------------------------------------------


def test_build_bahn_leg():
    # Happy path: resolve both stops, verify a connection, return (deep link, label).
    connection = _connection([("RB33", "REGIONAL")])

    def stub(*, url, body, timeout):
        if body is None:  # orte GET — name is in the query
            return _orte_hit(_MUNICH_ID, "München Hbf") if "nchen" in url else _orte_hit(_BERLIN_ID, "Berlin Hbf")
        return {"verbindungen": [connection]}  # journey POST

    url, label = build_bahn_leg(board_name="Berlin Hbf", alight_name="München Hbf", when=_WHEN, http_get=stub)
    assert url.startswith(f"{DbNavigatorConfig.SUCHE_URL}#") and label == "08:44 → 09:15 · RB33"


def test_build_bahn_leg_degrades_on_request_error():
    # Any network error (e.g. an Akamai 403) → the bare-search fallback link + "bahn.de" label, never a crash.
    def boom(*, url, body, timeout):
        raise urllib.error.HTTPError(url, 403, "Forbidden", hdrs={}, fp=None)  # type: ignore[arg-type]

    assert build_bahn_leg(board_name="A", alight_name="B", when=_WHEN, http_get=boom) == (
        DbNavigatorConfig.SUCHE_URL,
        DbNavigatorConfig.FALLBACK_LABEL,
    )


def test_build_bahn_leg_degrades_on_missing_eva():
    # A station that doesn't resolve to an EVA → fallback (can't be journey-queried).
    stub: DbGetter = lambda *, url, body, timeout: [{"id": "A=1@O=x@", "name": "x"}]  # noqa: E731
    assert build_bahn_leg(board_name="A", alight_name="B", when=_WHEN, http_get=stub) == (
        DbNavigatorConfig.SUCHE_URL,
        DbNavigatorConfig.FALLBACK_LABEL,
    )


def test_build_bahn_leg_degrades_on_no_connection():
    # Both stops resolve but no connection exists → fallback.
    def stub(*, url, body, timeout):
        if body is None:
            return _orte_hit(_MUNICH_ID, "München Hbf") if "nchen" in url else _orte_hit(_BERLIN_ID, "Berlin Hbf")
        return {"verbindungen": []}

    assert build_bahn_leg(board_name="Berlin Hbf", alight_name="München Hbf", when=_WHEN, http_get=stub) == (
        DbNavigatorConfig.SUCHE_URL,
        DbNavigatorConfig.FALLBACK_LABEL,
    )


# --- live end-to-end (opt-in; hits the real bahn.de API) ---------------------

_LIVE_ROUTES = [("Freudenstadt Hbf", "Horb"), ("Berlin Hbf", "Potsdam Hbf"), ("Karlsruhe Hbf", "Pforzheim Hbf")]


@pytest.mark.skipif(
    os.environ.get("BIKE_ROUTER_LIVE_DB") != "1",
    reason="live bahn.de API test — set BIKE_ROUTER_LIVE_DB=1 to run (network)",
)
@pytest.mark.parametrize(("board", "alight"), _LIVE_ROUTES, ids=[f"{b}->{a}" for b, a in _LIVE_ROUTES])
def test_build_bahn_leg_live(board: str, alight: str):
    # Against the REAL API each route MUST resolve first try to a working deep link + a real "dep…→arr…"
    # label, NEVER the bare fallback — the getter spaces its calls so Akamai doesn't 403 the burst.
    now = datetime.datetime.now().replace(second=0, microsecond=0)
    url, label = build_bahn_leg(board_name=board, alight_name=alight, when=now, http_get=default_db_get)
    assert url.startswith(f"{DbNavigatorConfig.SUCHE_URL}#") and "soid=" in url  # a real deep link, not the bare URL
    assert "→" in label and label != DbNavigatorConfig.FALLBACK_LABEL  # a real connection label, not the fallback
