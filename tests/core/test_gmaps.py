"""Google Maps URL builder tests."""

import pytest

from bike_router.core.constants import GmapsConfig
from bike_router.core.gmaps import _fmt, build_gmaps_url


def _waypoints() -> list[tuple[float, float]]:
    """Exactly N_WAYPOINTS (lat, lon) points — the count build_gmaps_url requires."""
    return [(48.0 + i * 0.1, 8.0 + i * 0.1) for i in range(GmapsConfig.N_WAYPOINTS)]


def test_fmt():
    # The one coord formatter: "lat,lon" at COORD_PRECISION (6) decimals.
    assert _fmt(point=(48.5, 8.25)) == "48.500000,8.250000"


def test_build_gmaps_url():
    # Full URL: api=1 + bicycling + origin/destination; interior points pipe-joined (N-2 waypoints
    # → N-3 encoded pipes). A 2-point leg emits origin+destination, no waypoints; <2 points fails loud.
    url = build_gmaps_url(waypoints_latlon=_waypoints())
    assert url.startswith("https://www.google.com/maps/dir/?api=1")
    assert "travelmode=bicycling" in url and "origin=" in url and "destination=" in url and "waypoints=" in url
    assert url.count("%7C") == GmapsConfig.N_WAYPOINTS - 3
    two = build_gmaps_url(waypoints_latlon=[(48.0, 8.0), (49.0, 9.0)])
    assert "origin=" in two and "destination=" in two and "waypoints=" not in two
    with pytest.raises(AssertionError):
        build_gmaps_url(waypoints_latlon=[(48.0, 8.0)])
