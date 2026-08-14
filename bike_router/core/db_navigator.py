"""Deutsche Bahn (bahn.de) timetable deep-link builder: station name + coords → a working search URL.

Fully OFFLINE: the link embeds each stop as a HAFAS coordinate location id (``A=2@O=name@X=lon@Y=lat@``)
that bahn.de resolves in the user's browser on click — no server-side API call (Akamai 403s datacenter IPs).
"""

import datetime
import urllib.parse

from bike_router.core.constants import DbNavigatorConfig


def coord_location_id(*, name: str, lat: float, lon: float) -> str:
    """A HAFAS coordinate location id ``A=2@O=<name>@X=<lon·1e6>@Y=<lat·1e6>@`` — resolves client-side.

    X is longitude and Y latitude, both in micro-degrees (integer); no server lookup needed.

    Args:
        name: Display name for the stop.
        lat: Latitude in degrees.
        lon: Longitude in degrees.
    """
    return f"A=2@O={name}@X={round(lon * 1_000_000)}@Y={round(lat * 1_000_000)}@"


def build_bahn_url(
    *,
    board_name: str,
    board_lat: float,
    board_lon: float,
    alight_name: str,
    alight_lat: float,
    alight_lon: float,
    depart: datetime.datetime,
) -> str:
    """A bahn.de deep link for a train ride — PURE, no network (coordinate ids resolve in the browser).

    Mirrors the hand-verified format: coordinate ``soid``/``zoid`` with ``sot=ADR``, the ``vm`` regional
    filter, one 2nd-class traveller, and our estimated boarding time as the search anchor.

    Args:
        board_name: Boarding station name.
        board_lat: Boarding latitude.
        board_lon: Boarding longitude.
        alight_name: Alighting station name.
        alight_lat: Alighting latitude.
        alight_lon: Alighting longitude.
        depart: Estimated boarding datetime (from the route's own timing) — the search's start time.
    """
    params = {
        "sts": "true",
        "so": board_name,
        "zo": alight_name,
        "kl": DbNavigatorConfig.KLASSE,
        "r": DbNavigatorConfig.TRAVELLER,
        "soid": coord_location_id(name=board_name, lat=board_lat, lon=board_lon),
        "zoid": coord_location_id(name=alight_name, lat=alight_lat, lon=alight_lon),
        "sot": "ADR",
        "zot": "ADR",
        "hd": depart.strftime("%Y-%m-%dT%H:%M:00"),
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
