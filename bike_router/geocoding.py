"""Geocoding via OpenStreetMap Nominatim (through geopy).

Resolves a place string to (lat, lon). A single RateLimiter-wrapped callable is
built once (make_geocode_fn) and reused for origin + destination so the 1 req/s
Nominatim policy is honoured across both lookups.
"""

import logging
from collections.abc import Callable

from geopy.exc import GeocoderServiceError
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from geopy.location import Location

from bike_router.constants import NominatimConfig

logger = logging.getLogger(__name__)

GeocodeFn = Callable[[str], Location | None]


class GeocodeError(RuntimeError):
    """Raised when a place string cannot be resolved to coordinates."""


def make_geocode_fn() -> GeocodeFn:
    """Build the rate-limited Nominatim geocode callable (1 req/s)."""
    geolocator = Nominatim(user_agent=NominatimConfig.USER_AGENT)
    fn: GeocodeFn = RateLimiter(geolocator.geocode, min_delay_seconds=NominatimConfig.RATE_LIMIT_S)
    return fn


def geocode(place: str, geocode_fn: GeocodeFn) -> tuple[float, float]:
    """Resolve ``place`` to (lat, lon) via the given geocode callable.

    Args:
        place: A human place string, e.g. "Freudenstadt, Germany".
        geocode_fn: Rate-limited geocode callable from make_geocode_fn; reused
            across origin + destination so the rate limiter spans both.

    Raises:
        GeocodeError: if the place is not found or the service errors.
    """
    try:
        location = geocode_fn(place)
    except GeocoderServiceError as exc:  # network / service failure
        raise GeocodeError(f"Geocoding service error for {place!r}: {exc}") from exc
    if location is None:
        raise GeocodeError(f"Could not geocode {place!r} — no matching location found.")
    return float(location.latitude), float(location.longitude)
