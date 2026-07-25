"""Google Maps URL builder tests."""

import pytest

from bike_router.gmaps import build_gmaps_url


def _ten_points() -> list[tuple[float, float]]:
    return [(48.0 + i * 0.1, 8.0 + i * 0.1) for i in range(10)]


def test_url_structure():
    url = build_gmaps_url(waypoints_latlon=_ten_points())
    assert url.startswith("https://www.google.com/maps/dir/?api=1")
    assert "travelmode=bicycling" in url
    assert "origin=" in url and "destination=" in url
    assert "waypoints=" in url
    # 8 intermediate waypoints → 7 separators (URL-encoded pipe %7C)
    assert url.count("%7C") == 7


def test_wrong_count_raises():
    with pytest.raises(ValueError):
        build_gmaps_url(waypoints_latlon=[(48.0, 8.0), (49.0, 9.0)])
