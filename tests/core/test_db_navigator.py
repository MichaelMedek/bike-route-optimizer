"""Unit tests for the offline bahn.de deep-link builder (db_navigator).

The builder is PURE (no network): it embeds coordinate location ids that resolve in the user's browser
on click, so the app never calls the bahn API (Akamai 403s datacenter IPs) — nothing to mock here.
"""

import datetime

from bike_router.core.constants import DbNavigatorConfig
from bike_router.core.db_navigator import build_bahn_url, coord_location_id

_DEPART = datetime.datetime(2026, 8, 20, 9, 15, 0)


def test_coord_location_id():
    # A=2 coordinate id: name verbatim, X=lon·1e6, Y=lat·1e6, rounded to whole micro-degrees.
    assert coord_location_id(name="Gaggenau", lat=48.8009, lon=8.3216) == "A=2@O=Gaggenau@X=8321600@Y=48800900@"


def test_build_bahn_url():
    # PURE builder: coordinate soid/zoid (sot=ADR), the vm regional filter, and our estimated dep time.
    url = build_bahn_url(
        board_name="Gaggenau",
        board_lat=48.8009,
        board_lon=8.3216,
        alight_name="Baiersbronn",
        alight_lat=48.5036,
        alight_lon=8.3720,
        depart=_DEPART,
    )
    assert url.startswith(f"{DbNavigatorConfig.SUCHE_URL}#")
    assert "sot=ADR" in url and "zot=ADR" in url  # coordinate stops, resolved client-side
    assert "soid=" in url and "zoid=" in url and "hd=2026-08-20T09%3A15%3A00" in url
    assert f"vm={DbNavigatorConfig.VM_CODES.replace(',', '%2C')}" in url or "vm=03" in url
