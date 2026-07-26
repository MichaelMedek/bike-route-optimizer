"""Google Maps directions-URL builder.

Uses the official Maps URLs `api=1` scheme: origin + up to 9 intermediate waypoints
+ destination, travelmode=bicycling. With N=10 Visvalingam-selected points that is
8 intermediate waypoints, within the api=1 limit.
"""

from urllib.parse import urlencode

from bike_router.constants import GmapsConfig, GraphConfig


def _fmt(point: tuple[float, float]) -> str:
    latitude, longitude = point
    assert -90.0 <= latitude <= 90.0, "latitude out of range"
    assert -180.0 <= longitude <= 180.0, "longitude out of range"
    return f"{latitude:.{GraphConfig.COORD_PRECISION}f},{longitude:.{GraphConfig.COORD_PRECISION}f}"


def build_gmaps_url(waypoints_latlon: list[tuple[float, float]]) -> str:
    """Build a bicycling directions URL from exactly N (lat, lon) points."""
    expected = GmapsConfig.N_WAYPOINTS
    if len(waypoints_latlon) != expected:
        raise ValueError(f"expected exactly {expected} waypoints, got {len(waypoints_latlon)}")
    assert expected >= 2, "need origin + destination at minimum"

    origin = _fmt(point=waypoints_latlon[0])
    destination = _fmt(point=waypoints_latlon[-1])
    intermediate = [_fmt(point=point) for point in waypoints_latlon[1:-1]]
    assert len(intermediate) <= GmapsConfig.MAX_INTERMEDIATE_WAYPOINTS, "too many intermediate waypoints for Maps api=1"

    params = {
        "origin": origin,
        "destination": destination,
        "travelmode": GmapsConfig.TRAVEL_MODE,
    }
    if intermediate:
        params["waypoints"] = "|".join(intermediate)

    # urlencode handles URL-escaping (the "|" becomes %7C, commas %2C).
    url = f"{GmapsConfig.BASE_URL}&{urlencode(params)}"
    assert url.startswith("https://"), "Maps URL must be https"
    return url
