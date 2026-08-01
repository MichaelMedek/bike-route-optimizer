"""Geocoding: Nominatim (one-shot resolve) + Photon (search-as-you-type).

``geocode``/``geocode_endpoint`` resolve a place string to (lat, lon) via Nominatim; ``photon_autocomplete``
powers the web typeahead via Photon (Nominatim forbids client autocomplete).
"""

import logging
from collections.abc import Callable
from typing import Protocol

import requests
from geopy.exc import GeocoderServiceError
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from geopy.location import Location

from bike_router.core.constants import NominatimConfig, PhotonConfig
from bike_router.core.errors import GeocodeConnectionError, GeocodeNotFoundError

logger = logging.getLogger(__name__)

GeocodeFn = Callable[[str], Location | None]
# Injectable HTTP seam (url, params, timeout) → parsed JSON — lets tests run offline.
# A tuple value becomes a repeated query param (e.g. osm_tag=place:city&osm_tag=place:town).
HttpParams = dict[str, str | float | tuple[str, ...]]

# Nominatim's usage policy MANDATES caching ("Results must be cached … clients sending
# repeatedly the same query may be blocked"). Place→(lat,lon) is immutable.
_GEOCODE_CACHE: dict[str, tuple[float, float]] = {}


class HttpGetter(Protocol):
    """Callable seam for an HTTP GET returning parsed JSON (keyword-callable)."""

    def __call__(self, *, url: str, params: HttpParams, timeout: float) -> object: ...


def make_geocode_fn() -> GeocodeFn:
    """Build the rate-limited Nominatim geocode callable (1 req/s). ``swallow_exceptions=False``
    + ``max_retries=0`` so a network failure raises GeocoderServiceError immediately instead of
    retry-logging a wall of tracebacks and returning None (which masks the outage as "not found").
    """
    geolocator = Nominatim(user_agent=NominatimConfig.USER_AGENT)
    fn: GeocodeFn = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=NominatimConfig.RATE_LIMIT_S,
        max_retries=0,
        swallow_exceptions=False,
    )
    return fn


def _parse_latlon(place: str) -> tuple[float, float] | None:
    """A ``"lat, lon"`` literal → (lat, lon), or None if it isn't one (fall through to Nominatim).

    Lets the GPS button feed raw coordinates through the SAME text box as place names — the box
    text stays the single input. Requires exactly two comma-separated floats in valid WGS84 range.
    """
    parts = place.split(",")
    if len(parts) != 2:
        return None
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return (lat, lon) if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0 else None


def geocode(place: str, geocode_fn: GeocodeFn) -> tuple[float, float]:
    """Resolve ``place`` to (lat, lon) via the given geocode callable.

    A ``"lat, lon"`` literal (from the GPS button) resolves directly; any other string is geocoded.

    Args:
        place: A human place string, e.g. "Freudenstadt, Germany", or a "lat, lon" literal.
        geocode_fn: Rate-limited geocode callable from make_geocode_fn; reused
            across origin + destination so the rate limiter spans both.

    Raises:
        GeocodeConnectionError: if the service is unreachable (no internet).
        GeocodeNotFoundError: if the service answered but nothing matched.
    """
    if place in _GEOCODE_CACHE:  # policy-mandated: never re-query an identical string
        return _GEOCODE_CACHE[place]
    coords = _parse_latlon(place=place)
    if coords is not None:  # raw GPS coordinates — no lookup needed
        _GEOCODE_CACHE[place] = coords
        return coords
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
    """Geocode one named endpoint, re-raising the same error type with the field name so
    start/destination lookups fail loud (connection-vs-not-found preserved). Blank input is
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
    limit: int,
    osm_tag: str,
    http_get: HttpGetter,
) -> list[str]:
    """Search-as-you-type place labels ("Name, City, State") biased to ``bbox``.

    A blank term or ANY network/timeout/parse error returns [] — a per-keystroke typeahead
    must never crash on a weak connection (suggestions are convenience, never required).

    Args:
        term: The partial text the user has typed.
        bbox: Coverage box (west, south, east, north) to bias + limit suggestions.
        limit: Max suggestions to request.
        osm_tag: Photon osm_tag filter — settlements by default, ``railway:station`` for Bahnhöfe.
        http_get: Injectable HTTP getter (url, params, timeout) → parsed JSON.
    """
    if not term.strip():
        return []
    west, south, east, north = bbox
    params: HttpParams = {
        "q": term,
        "limit": limit,
        "lang": PhotonConfig.LANG,
        "osm_tag": osm_tag,
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


def as_bahnhof(*, name: str) -> str:
    """Append " Bahnhof" to a place name unless it already ends in it — the ONE station-label rule.

    The bare OSM name (just "Sauldorf") geocodes to the town centre; "<name> Bahnhof" hits the
    platform. Shared by the autocomplete Bahnhof pick and the clickable top-station markers.
    """
    trimmed = name.strip()
    return trimmed if trimmed.lower().endswith("bahnhof") else f"{trimmed} Bahnhof"


def bahnhof_suggestion(
    *,
    term: str,
    bbox: tuple[float, float, float, float],
    http_get: HttpGetter,
) -> str | None:
    """The "<place> Bahnhof" label for the typed term IF a railway station matches, else None.

    Station OSM ``name``s often lack "Bahnhof" (just "Sauldorf"), rendering identical to the town;
    the station-first query's name is concatenated to "<name> Bahnhof" and shown FIRST as a red button.

    Args:
        term: The partial text the user has typed.
        bbox: Coverage box (west, south, east, north) to bias the station query.
        http_get: Injectable HTTP getter.
    """
    stations = photon_autocomplete(
        term=term, bbox=bbox, limit=1, osm_tag=PhotonConfig.STATION_OSM_TAG, http_get=http_get
    )
    if not stations:
        return None
    name = stations[0].split(",")[0].strip()  # station label's own name, before the ", State" part
    if not name:
        return None
    return as_bahnhof(name=name)


def autocomplete_with_stations(
    *,
    term: str,
    bbox: tuple[float, float, float, float],
    limit: int,
    http_get: HttpGetter,
) -> tuple[str | None, list[str]]:
    """(bahnhof_label, place_labels): a red-button "<place> Bahnhof" pick (if a station matches) plus
    ordinary settlement suggestions. Splitting them lets the UI render the Bahnhof pick FIRST and red.

    Args:
        term: The partial text the user has typed.
        bbox: Coverage box (west, south, east, north) to bias + limit suggestions.
        limit: Max settlement suggestions to request.
        http_get: Injectable HTTP getter.
    """
    bahnhof = bahnhof_suggestion(term=term, bbox=bbox, http_get=http_get)
    places = photon_autocomplete(
        term=term, bbox=bbox, limit=limit, osm_tag=PhotonConfig.PLACE_OSM_TAG, http_get=http_get
    )
    return bahnhof, [p for p in places if p != bahnhof]  # drop a place duplicating the Bahnhof pick


def nearest_place_name(*, lat: float, lon: float, http_get: HttpGetter) -> str | None:
    """Nearest settlement NAME to (lat, lon) via Photon reverse geocoding — the place ``name``
    (e.g. "Baiersbronn"), never a full address. Any network/parse error or empty result returns
    None (caller drops the label) so a weak connection never crashes. Names the gmaps waypoints.
    """
    params: HttpParams = {
        "lat": lat,
        "lon": lon,
        "lang": PhotonConfig.LANG,
        "osm_tag": PhotonConfig.REVERSE_PLACE_TAGS,  # only real settlements → a town/village, never a postcode
        "radius": PhotonConfig.REVERSE_RADIUS_KM,  # search within this many km so a remote waypoint still names one
        "limit": 1,
    }
    reverse_url = PhotonConfig.BASE_URL.removesuffix("/api") + "/reverse"
    try:
        payload = http_get(url=reverse_url, params=params, timeout=PhotonConfig.TIMEOUT_S)
    except requests.RequestException as exc:
        logger.info(f"Photon reverse lookup failed at ({lat:.4f}, {lon:.4f}) — no name ({exc})")
        return None
    features = payload["features"]  # type: ignore[index]
    if not features:
        return None
    name = str(features[0]["properties"].get("name") or "").strip()
    return name or None
