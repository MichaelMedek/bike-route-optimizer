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
    """A ``"lat, lon"`` (optionally ``"lat, lon (Name)"``) literal → (lat, lon), else None.

    Lets the GPS button, top-station markers, and the Bahnhof pick feed EXACT coordinates through the
    same box as names: coords always win and the ``(Name)`` label is ignored, so a pick never re-geocodes.
    """
    coord_text = place.split("(", 1)[0]  # drop a trailing "(Name)" annotation, if any
    parts = coord_text.split(",")
    if len(parts) != 2:
        return None
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return (lat, lon) if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0 else None


def latlon_box_value(*, lat: float, lon: float, name: str | None) -> str:
    """A place-box value carrying EXACT coordinates + an optional readable ``(Name)`` — the ONE
    format ``_parse_latlon`` reads back. Coordinates always win on geocode, so a marker/station pick
    resolves to its true point and never depends on re-geocoding a fuzzy name (Zürich, Schalkstetten).
    """
    coords = f"{lat:.5f}, {lon:.5f}"
    return f"{coords} ({name.strip()})" if name and name.strip() else coords


def box_display_label(value: str) -> str:
    """The human label to SHOW for a box value: the ``(Name)`` inside a coords literal, else the value.

    So a coords-carrying pick reads as "Zürich Bahnhof" in the UI while the box still holds the exact
    "lat, lon (Name)" that geocodes to the true point.
    """
    if _parse_latlon(place=value) is not None and "(" in value and value.rstrip().endswith(")"):
        return value[value.index("(") + 1 : value.rstrip().rindex(")")].strip()
    return value


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


def default_http_get(*, url: str, params: HttpParams, timeout: float) -> object:
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


def _photon_query(*, url: str, params: HttpParams, http_get: HttpGetter) -> list[dict[str, object]]:
    """GET a Photon endpoint and return its GeoJSON ``features`` (empty on ANY network/parse error).

    The ONE Photon call site — the search + reverse helpers share it, so error tolerance and the
    features extraction live in a single place. A per-keystroke/per-marker lookup must never crash.
    """
    try:
        payload = http_get(url=url, params=params, timeout=PhotonConfig.TIMEOUT_S)
    except requests.RequestException as exc:
        logger.info(f"Photon request to {url} failed — no results ({exc})")
        return []
    return list(payload["features"])  # type: ignore[index]  # a well-formed Photon reply always has it


def _photon_features(
    *,
    term: str,
    bbox: tuple[float, float, float, float],
    limit: int,
    osm_tag: str,
    http_get: HttpGetter,
) -> list[dict[str, object]]:
    """Raw Photon GeoJSON features for a typed term, biased to ``bbox`` (empty on blank/any error).

    ``photon_autocomplete`` maps these to labels; the station pick reads their geometry for exact coords.

    Args:
        term: The partial text the user has typed.
        bbox: Coverage box (west, south, east, north) to bias + limit results.
        limit: Max features to request.
        osm_tag: Photon osm_tag filter — settlements, or ``railway:station`` for Bahnhöfe.
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
    return _photon_query(url=PhotonConfig.BASE_URL, params=params, http_get=http_get)


def photon_autocomplete(
    *,
    term: str,
    bbox: tuple[float, float, float, float],
    limit: int,
    osm_tag: str,
    http_get: HttpGetter,
) -> list[str]:
    """Search-as-you-type place labels ("Name, City, State") biased to ``bbox`` (blank/error → [])."""
    features = _photon_features(term=term, bbox=bbox, limit=limit, osm_tag=osm_tag, http_get=http_get)
    return [photon_label(properties=_feature_properties(feature)) for feature in features]


def _feature_lonlat(feature: dict[str, object]) -> tuple[float, float] | None:
    """(lon, lat) from a Photon feature's Point geometry, or None if malformed."""
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        return None
    coords = geometry.get("coordinates")
    if not isinstance(coords, list) or len(coords) != 2:
        return None
    return float(coords[0]), float(coords[1])


def _feature_properties(feature: dict[str, object]) -> dict[str, object]:
    """A Photon feature's ``properties`` dict (empty if absent/malformed) — the ONE typed accessor."""
    properties = feature.get("properties")
    return properties if isinstance(properties, dict) else {}


def _feature_name(feature: dict[str, object]) -> str:
    """The stripped ``name`` from a Photon feature's properties, or "" if absent/malformed."""
    return str(_feature_properties(feature).get("name") or "").strip()


def as_bahnhof(*, name: str) -> str:
    """Append " Bahnhof" to a place name unless it already ends in it — the ONE station-label rule.

    Station OSM ``name``s often lack "Bahnhof" (just "Sauldorf"), rendering identical to the town, so
    the readable label spells out the platform. Shared by the autocomplete pick and top-station markers.
    """
    trimmed = name.strip()
    return trimmed if trimmed.lower().endswith("bahnhof") else f"{trimmed} Bahnhof"


def bahnhof_suggestion(
    *,
    term: str,
    bbox: tuple[float, float, float, float],
    http_get: HttpGetter,
) -> str | None:
    """A station-box value ``"lat, lon (Name Bahnhof)"`` for the typed term IF a station matches, else None.

    Uses the matched station's OWN Photon coordinates (not a name re-geocode), so picking it snaps to
    the exact platform even where "<name> Bahnhof" doesn't geocode (e.g. Zürich, Schalkstetten).

    Args:
        term: The partial text the user has typed.
        bbox: Coverage box (west, south, east, north) to bias the station query.
        http_get: Injectable HTTP getter.
    """
    features = _photon_features(term=term, bbox=bbox, limit=1, osm_tag=PhotonConfig.STATION_OSM_TAG, http_get=http_get)
    if not features:
        return None
    lonlat = _feature_lonlat(features[0])
    name = _feature_name(features[0])
    if lonlat is None or not name:
        return None
    lon, lat = lonlat
    return latlon_box_value(lat=lat, lon=lon, name=as_bahnhof(name=name))


def autocomplete_with_stations(
    *,
    term: str,
    bbox: tuple[float, float, float, float],
    limit: int,
    http_get: HttpGetter,
) -> tuple[str | None, list[str]]:
    """(bahnhof_box_value, place_labels): a red-button station pick ``"lat, lon (Name Bahnhof)"`` (if a
    station matches) plus ordinary settlement suggestions, so the UI renders the Bahnhof pick FIRST + red.

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
    return bahnhof, places


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
    features = _photon_query(url=reverse_url, params=params, http_get=http_get)
    return (_feature_name(features[0]) or None) if features else None
