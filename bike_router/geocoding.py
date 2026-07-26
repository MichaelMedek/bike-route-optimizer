"""Geocoding via OpenStreetMap Nominatim (through geopy).

Resolves a place string to (lat, lon). One RateLimiter-wrapped callable
(make_geocode_fn) is reused for origin + destination so the 1 req/s policy holds.
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
    """Base for any failure to resolve a place string to coordinates."""


class GeocodeConnectionError(GeocodeError):
    """The geocoding service was unreachable (no internet / DNS / service down)."""


class GeocodeNotFoundError(GeocodeError):
    """The service answered but no location matched the place string."""


def make_geocode_fn() -> GeocodeFn:
    """Build the rate-limited Nominatim geocode callable (1 req/s).

    ``swallow_exceptions=False`` + ``max_retries=0`` so a network failure raises
    a GeocoderServiceError immediately instead of retry-logging a wall of
    tracebacks and returning None (which masks the outage as "not found").
    """
    geolocator = Nominatim(user_agent=NominatimConfig.USER_AGENT)
    fn: GeocodeFn = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=NominatimConfig.RATE_LIMIT_S,
        max_retries=0,
        swallow_exceptions=False,
    )
    return fn


def geocode(place: str, geocode_fn: GeocodeFn) -> tuple[float, float]:
    """Resolve ``place`` to (lat, lon) via the given geocode callable.

    Args:
        place: A human place string, e.g. "Freudenstadt, Germany".
        geocode_fn: Rate-limited geocode callable from make_geocode_fn; reused
            across origin + destination so the rate limiter spans both.

    Raises:
        GeocodeConnectionError: if the service is unreachable (no internet).
        GeocodeNotFoundError: if the service answered but nothing matched.
    """
    try:
        location = geocode_fn(place)
    except GeocoderServiceError as exc:  # unreachable service / no connection
        raise GeocodeConnectionError(
            f"Could not reach the geocoding service for {place!r} — check your internet connection."
        ) from exc
    if location is None:
        raise GeocodeNotFoundError(f"Could not geocode {place!r} — no matching location found.")
    return float(location.latitude), float(location.longitude)


def geocode_endpoint(place: str, label: str, geocode_fn: GeocodeFn) -> tuple[float, float]:
    """Geocode one named endpoint, re-raising the same error type with the field name.

    Wraps ``geocode`` so start/destination lookups fail loud with the field name in
    the message, preserving the connection-vs-not-found distinction. Blank input is
    rejected without a lookup.

    Args:
        place: The place string to resolve.
        label: Human field name for the error (e.g. "Start", "Destination").
        geocode_fn: Rate-limited geocode callable from make_geocode_fn.
    """
    if not place.strip():
        raise GeocodeNotFoundError(f"{label} is empty")
    try:
        return geocode(place=place, geocode_fn=geocode_fn)
    except GeocodeConnectionError as exc:
        raise GeocodeConnectionError(f"{label} ({place!r}): {exc}") from exc
    except GeocodeNotFoundError as exc:
        raise GeocodeNotFoundError(f"{label} ({place!r}): could not find this place.") from exc
