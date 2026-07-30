"""Google Maps URL builder tests."""

import pytest

from bike_router.core.constants import GmapsConfig
from bike_router.core.gmaps import build_gmaps_url


def _waypoints() -> list[tuple[float, float]]:
    """Exactly N_WAYPOINTS (lat, lon) points — the count build_gmaps_url requires."""
    return [(48.0 + i * 0.1, 8.0 + i * 0.1) for i in range(GmapsConfig.N_WAYPOINTS)]


def test_url_structure():
    url = build_gmaps_url(waypoints_latlon=_waypoints())
    assert url.startswith("https://www.google.com/maps/dir/?api=1")
    assert "travelmode=bicycling" in url
    assert "origin=" in url and "destination=" in url
    assert "waypoints=" in url
    # origin + destination are separate params, so the pipe-joined intermediates number
    # N_WAYPOINTS - 2, giving N_WAYPOINTS - 3 separators (URL-encoded pipe %7C).
    assert url.count("%7C") == GmapsConfig.N_WAYPOINTS - 3


def test_two_points_is_valid_origin_destination_only():
    # A thinned short leg may reach build_gmaps_url with just origin + destination.
    url = build_gmaps_url(waypoints_latlon=[(48.0, 8.0), (49.0, 9.0)])
    assert url.startswith("https://www.google.com/maps/dir/?api=1")
    assert "origin=" in url and "destination=" in url
    assert "waypoints=" not in url  # no interior points → no waypoints param


def test_too_few_points_raises():
    with pytest.raises(AssertionError):
        build_gmaps_url(waypoints_latlon=[(48.0, 8.0)])  # need origin + destination
