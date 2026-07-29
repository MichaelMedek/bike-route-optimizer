"""Geocoding: Nominatim (one-shot resolve) + Photon (search-as-you-type).

``geocode``/``geocode_endpoint`` resolve a place string to (lat, lon) via Nominatim
(one deliberate lookup — policy-fine). ``photon_autocomplete`` powers the web
typeahead via Photon, which OSM built for that (Nominatim forbids client autocomplete).
"""

import logging
from collections.abc import Callable
from typing import Protocol

import requests
from geopy.exc import GeocoderServiceError
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from geopy.location import Location

from bike_router.constants import NominatimConfig, PhotonConfig
from bike_router.errors import GeocodeConnectionError, GeocodeNotFoundError

logger = logging.getLogger(__name__)

GeocodeFn = Callable[[str], Location | None]
# Injectable HTTP seam (url, params, timeout) → parsed JSON — lets tests run offline.
HttpParams = dict[str, str | float]

# Nominatim's usage policy MANDATES caching ("Results must be cached … clients sending
# repeatedly the same query may be blocked"). Place→(lat,lon) is immutable.
_GEOCODE_CACHE: dict[str, tuple[float, float]] = {}


class HttpGetter(Protocol):
    """Callable seam for an HTTP GET returning parsed JSON (keyword-callable)."""

    def __call__(self, *, url: str, params: HttpParams, timeout: float) -> object: ...


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
    if place in _GEOCODE_CACHE:  # policy-mandated: never re-query an identical string
        return _GEOCODE_CACHE[place]
    try:
        location = geocode_fn(place)
    except GeocoderServiceError as exc:  # unreachable service / no connection
        raise GeocodeConnectionError(
            f"Could not reach the geocoding service for {place!r} — check your internet connection."
        ) from exc
    if location is None:
        raise GeocodeNotFoundError(f"Could not geocode {place!r} — no matching location found.")
    result = (float(location.latitude), float(location.longitude))
    _GEOCODE_CACHE[place] = result
    return result


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


def _default_http_get(*, url: str, params: HttpParams, timeout: float) -> object:
    """Real HTTP GET returning parsed JSON (the production HttpGetter).

    Sends the project User-Agent — Photon 403s the default ``python-requests`` UA.
    """
    response = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": NominatimConfig.USER_AGENT})
    response.raise_for_status()
    return response.json()


def photon_label(properties: dict[str, object]) -> str:
    """Human label "Name, City, State" from a Photon feature's properties.

    Blank parts are skipped and a part equal to the name is not repeated, so a city
    whose name IS the settlement doesn't render "Freudenstadt, Freudenstadt".
    """
    name = str(properties.get("name") or "").strip()
    parts = [name]
    for key in ("city", "state"):
        value = str(properties.get(key) or "").strip()
        if value and value not in parts:
            parts.append(value)
    return ", ".join(part for part in parts if part)


def photon_autocomplete(
    *,
    term: str,
    bbox: tuple[float, float, float, float],
    limit: int = PhotonConfig.LIMIT,
    http_get: HttpGetter = _default_http_get,
) -> list[str]:
    """Search-as-you-type place labels biased to ``bbox`` (for the suggestion helper).

    Returns display labels only ("Name, City, State"); clicking one just fills the text
    box (still freely editable), and the box text is what gets geocoded on submit — so
    suggestions are a pure convenience, never required.

    A blank term returns [] without a request. ANY network/timeout/parse error also
    returns [] — a per-keystroke typeahead must never crash on a weak connection.

    Args:
        term: The partial text the user has typed.
        bbox: Coverage box (west, south, east, north) to bias + limit suggestions.
        limit: Max suggestions to request.
        http_get: Injectable HTTP getter (url, params, timeout) → parsed JSON.
    """
    if not term.strip():
        return []
    west, south, east, north = bbox
    params: HttpParams = {
        "q": term,
        "limit": limit,
        "lang": PhotonConfig.LANG,
        "osm_tag": PhotonConfig.PLACE_OSM_TAG,
        "bbox": f"{west},{south},{east},{north}",
        "lon": (west + east) / 2.0,  # centre bias so nearer places rank first
        "lat": (south + north) / 2.0,
    }
    try:
        payload = http_get(url=PhotonConfig.BASE_URL, params=params, timeout=PhotonConfig.TIMEOUT_S)
    except requests.RequestException as exc:  # ONLY the genuine network/HTTP failure
        logger.info(f"Photon autocomplete failed for {term!r} — offering no suggestions ({exc})")
        return []
    features = payload["features"]  # type: ignore[index]  # a well-formed Photon reply always has it
    return [photon_label(properties=feature["properties"]) for feature in features]
