"""Deutsche Bahn (bahn.de) timetable deep-link builder: station names → a working "look up the train" URL.

Resolves each stop via the ``orte`` API and labels the ride via the journey API; any network failure
degrades to a bare bahn.de search link. The train-leg analogue of gmaps.build_transit_url.
"""

import datetime
import gzip
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

from bike_router.core.constants import DbNavigatorConfig

logger = logging.getLogger(__name__)

# The orte id embeds the station's EVA number as "@L=<digits>@"; only an EVA-bearing hit is journey-queryable.
_EVA = re.compile(r"@L=(\d+)@")


class DbGetter(Protocol):
    """Callable seam for a DB API call returning parsed JSON: GET when ``body`` is None, else POST."""

    def __call__(self, *, url: str, body: dict[str, object] | None, timeout: float) -> object: ...


def default_db_get(*, url: str, body: dict[str, object] | None, timeout: float) -> object:
    """Real DB API call returning parsed JSON (the production DbGetter): GET if ``body`` is None, else POST.

    Uses urllib, NOT requests: bahn.de's Akamai layer fingerprints and 403-blocks the requests client
    (OPS_BLOCKED) while letting urllib through. Sends a browser User-Agent and gunzips a gzip reply.

    Args:
        url: The DB endpoint.
        body: JSON body to POST, or None for a GET.
        timeout: Per-request timeout in seconds.
    """
    headers = {"Accept": "application/json", "User-Agent": DbNavigatorConfig.USER_AGENT, "Accept-Encoding": "gzip"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    time.sleep(DbNavigatorConfig.REQUEST_SPACING_S)  # space calls so Akamai doesn't 403 a burst
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return json.loads(raw)


def resolve_halt(*, name: str, http_get: DbGetter) -> tuple[str, str, str] | None:
    """Resolve a station name to (station_id, display_name, eva) via the DB ``orte`` API, or None.

    Returns None if there is no hit or the top hit carries no EVA (``@L=``) — some hits are addresses,
    not stations. The id is passed through VERBATIM (mangling it, e.g. stripping ``@p=``, breaks the link).

    Args:
        name: A station name, e.g. "Freudenstadt Hbf".
        http_get: Injectable DB getter (shared seam) for offline tests.
    """
    params = {"suchbegriff": name, "typ": "ALL", "limit": str(DbNavigatorConfig.ORTE_LIMIT)}
    query = "&".join(f"{key}={urllib.parse.quote(value)}" for key, value in params.items())
    hits: Any = http_get(url=f"{DbNavigatorConfig.ORTE_URL}?{query}", body=None, timeout=DbNavigatorConfig.TIMEOUT_S)
    if not hits:
        return None
    top = hits[0]  # a well-formed orte reply is a non-empty list of station dicts
    eva = _EVA.search(top["id"])
    return (top["id"], top["name"], eva.group(1)) if eva else None


def _leg_products(*, connection: dict[str, object]) -> list[tuple[str, str]]:
    """(mittelText, produktGattung) for each PUBLICTRANSPORT leg of a connection.

    Foot-path/transfer segments have no ``verkehrsmittel.typ == PUBLICTRANSPORT`` and are skipped —
    only vehicle legs carry a product to check/display.
    """
    legs: list[tuple[str, str]] = []
    segments: Any = connection["verbindungsAbschnitte"]
    for segment in segments:
        vehicle = segment.get("verkehrsmittel", {})
        if vehicle.get("typ") != "PUBLICTRANSPORT":
            continue
        legs.append((vehicle.get("mittelText", "?"), vehicle["produktGattung"]))
    return legs


def verify_connection(
    *, board_eva: str, alight_eva: str, when: datetime.datetime, http_get: DbGetter
) -> dict[str, object] | None:
    """The first connection board→alight at ``when`` via the journey API, or None if none exist.

    The request restricts products to regional/S/U/Tram (matching the link's ``vm`` filter), so the first
    result is the one whose trains the deep link will show; used only to LABEL the leg.

    Args:
        board_eva: Boarding station EVA number.
        alight_eva: Alighting station EVA number.
        when: Departure datetime the search starts from.
        http_get: Injectable DB getter (shared seam) for offline tests.
    """
    body: dict[str, object] = {
        "reservierungsKontingenteVorhanden": False,
        "schnelleVerbindungen": True,
        "sitzplatzOnly": False,
        "abfahrtsHalt": f"A=1@L={board_eva}@",
        "ankunftsHalt": f"A=1@L={alight_eva}@",
        "produktgattungen": list(DbNavigatorConfig.ALLOWED_PRODUCTS),
        "anfrageZeitpunkt": when.strftime("%Y-%m-%dT%H:%M:00"),
        "ankunftSuche": "ABFAHRT",
        "klasse": "KLASSE_2",
        "reisende": [
            {
                "typ": "ERWACHSENER",
                "anzahl": 1,
                "alter": [],
                "ermaessigungen": [{"art": "KEINE_ERMAESSIGUNG", "klasse": "KLASSENLOS"}],
            }
        ],
    }
    reply: Any = http_get(url=DbNavigatorConfig.JOURNEY_URL, body=body, timeout=DbNavigatorConfig.TIMEOUT_S)
    connections = reply["verbindungen"]
    return connections[0] if connections else None


def build_link(*, board: tuple[str, str, str], alight: tuple[str, str, str], when: datetime.datetime) -> str:
    """A bahn.de timetable deep link for board→alight departing at ``when``, regional modes only — PURE.

    Mirrors the hand-verified working format: ``vm`` selects the transport modes, ``r`` encodes one adult
    2nd-class traveller, ``hza=D`` the departure mode. Each ``(id, name, eva)`` triple comes from resolve_halt.

    Args:
        board: The boarding stop's resolved (station_id, name, eva) triple.
        alight: The alighting stop's resolved (station_id, name, eva) triple.
        when: Departure datetime.
    """
    params = {
        "sts": "true",
        "so": board[1],
        "zo": alight[1],
        "kl": DbNavigatorConfig.KLASSE,
        "r": DbNavigatorConfig.TRAVELLER,
        "soid": board[0],
        "zoid": alight[0],
        "sot": "ST",
        "zot": "ST",
        "hd": when.strftime("%Y-%m-%dT%H:%M:00"),
        "hza": "D",
        "hz": "[]",
        "ar": "false",
        "s": "false",
        "d": "false",
        "vm": DbNavigatorConfig.VM_CODES,
        "fm": "true",
        "bp": "false",
    }
    fragment = "&".join(f"{key}={urllib.parse.quote(str(value))}" for key, value in params.items())
    return f"{DbNavigatorConfig.SUCHE_URL}#{fragment}"


def build_label(*, connection: dict[str, object]) -> str:
    """A "HH:MM → HH:MM · RB33 › RE4" label: dep/arr time + trains, elided to "first › … › last" past 2 legs."""
    segments: Any = connection["verbindungsAbschnitte"]
    dep = segments[0]["abfahrt"]["sollzeit"][11:16]
    arr = segments[-1]["ankunft"]["sollzeit"][11:16]
    names = [mittel for mittel, _ in _leg_products(connection=connection)]
    shown = names if len(names) <= 2 else [names[0], "…", names[-1]]
    return f"{dep} → {arr} · {' › '.join(shown)}"


def build_bahn_leg(
    *, board_name: str, alight_name: str, when: datetime.datetime, http_get: DbGetter
) -> tuple[str, str]:
    """(bahn_url, label) for a train ride at ``when``; degrades to a bare search link on any failure.

    Resolves both stops, verifies a connection for the label, and builds the deep link; any network error
    (a 403 ⊂ URLError) or missing station/connection falls back to (SUCHE_URL, "bahn.de").

    Args:
        board_name: Boarding station name.
        alight_name: Alighting station name.
        when: Departure datetime (from the route's own timing).
        http_get: Injectable DB getter (shared seam) for offline tests.
    """
    fallback = (DbNavigatorConfig.SUCHE_URL, DbNavigatorConfig.FALLBACK_LABEL)
    try:
        board = resolve_halt(name=board_name, http_get=http_get)
        alight = resolve_halt(name=alight_name, http_get=http_get)
        if board is None or alight is None:
            return fallback
        connection = verify_connection(board_eva=board[2], alight_eva=alight[2], when=when, http_get=http_get)
        if connection is None:
            return fallback
        return build_link(board=board, alight=alight, when=when), build_label(connection=connection)
    except (urllib.error.URLError, OSError) as exc:  # unreachable / blocked (403) / timeout — degrade to bare link
        logger.info(f"DB link for {board_name!r} → {alight_name!r} degraded to bare search ({exc})")
        return fallback
