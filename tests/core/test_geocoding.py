"""geocoding tests — Nominatim resolve + Photon typeahead/reverse, fully mocked (zero network).

One test_<fn> per production symbol (exact-name mirror) and TestHttpGetter for the Protocol.
Each folds every scenario for its target; the geopy geocoder and the injectable HTTP getter are
mocked so nothing touches the network.
"""

from unittest.mock import MagicMock

import pytest
import requests
from geopy.exc import GeocoderServiceError, GeocoderUnavailable

from bike_router.core.constants import NominatimConfig
from bike_router.core.errors import BikeRouterError, GeocodeConnectionError, GeocodeNotFoundError
from bike_router.core.geocoding import (
    _GEOCODE_CACHE,
    HttpGetter,
    _default_http_get,
    geocode,
    geocode_endpoint,
    make_geocode_fn,
    nearest_place_name,
    photon_autocomplete,
    photon_label,
)

_BBOX = (8.30, 48.40, 8.80, 48.95)  # (west, south, east, north)


@pytest.fixture(autouse=True)
def _clear_geocode_cache():
    """Reset the module-level geocode cache so per-test call counts are deterministic."""
    _GEOCODE_CACHE.clear()
    yield
    _GEOCODE_CACHE.clear()


def _located(lat: float, lon: float) -> MagicMock:
    """A geopy Location stand-in exposing .latitude/.longitude."""
    loc = MagicMock()
    loc.latitude, loc.longitude = lat, lon
    return loc


def _photon_feature(name: str, lon: float, lat: float, **props: object) -> dict[str, object]:
    """A minimal Photon GeoJSON feature (coordinates are [lon, lat])."""
    return {"geometry": {"coordinates": [lon, lat]}, "properties": {"name": name, **props}}


# --- HTTP seam ---------------------------------------------------------------


class TestHttpGetter:
    def test_protocol_is_satisfied_by_a_keyword_callable(self):
        # The seam is a keyword-callable (url, params, timeout) → JSON; a plain fn implements it.
        def getter(*, url: str, params: dict, timeout: float) -> object:
            return {"ok": url}

        fn: HttpGetter = getter  # a conforming callable type-checks as the Protocol
        assert fn(url="https://x", params={}, timeout=1.0) == {"ok": "https://x"}


def test_default_http_get(monkeypatch):
    # Sends the project User-Agent (Photon 403s the default requests UA), raises on HTTP error,
    # and returns parsed JSON. requests.get is patched so nothing hits the network.
    response = MagicMock()
    response.json.return_value = {"features": []}
    fake_get = MagicMock(return_value=response)
    monkeypatch.setattr(requests, "get", fake_get)
    payload = _default_http_get(url="https://photon/api", params={"q": "x"}, timeout=2.0)
    assert payload == {"features": []}
    response.raise_for_status.assert_called_once()
    assert fake_get.call_args.kwargs["headers"]["User-Agent"] == NominatimConfig.USER_AGENT


# --- Nominatim resolve -------------------------------------------------------


def test_make_geocode_fn():
    # Returns a callable wrapping Nominatim (rate-limited); building it touches no network.
    fn = make_geocode_fn()
    assert callable(fn)


def test_geocode():
    # A found location → (lat, lon); an identical repeat is served from cache (Nominatim policy);
    # None → GeocodeNotFoundError; a service error → GeocodeConnectionError (distinct types).
    fake = MagicMock(return_value=_located(48.4633, 8.4116))
    assert geocode(place="Freudenstadt, Germany", geocode_fn=fake) == (48.4633, 8.4116)
    fake.assert_called_once_with("Freudenstadt, Germany")

    cached = MagicMock(return_value=_located(48.0, 8.0))
    first = geocode(place="Horb", geocode_fn=cached)
    second = geocode(place="Horb", geocode_fn=cached)
    assert first == second == (48.0, 8.0)
    cached.assert_called_once()  # second served from cache, no network

    not_found = MagicMock(return_value=None)
    with pytest.raises(GeocodeNotFoundError) as nf:
        geocode(place="Nowhere at all, Atlantis", geocode_fn=not_found)
    assert isinstance(nf.value, BikeRouterError) and not isinstance(nf.value, GeocodeConnectionError)

    unreachable = MagicMock(side_effect=GeocoderServiceError("boom"))
    with pytest.raises(GeocodeConnectionError, match="internet connection"):
        geocode(place="Somewhere", geocode_fn=unreachable)


def test_geocode_endpoint():
    # Delegates to geocode with the field name in the message; blank input is rejected without a
    # lookup; the connection-vs-not-found distinction is preserved and field-named.
    ok = MagicMock(return_value=_located(48.46, 8.41))
    assert geocode_endpoint(place="Freudenstadt", label="Start", geocode_fn=ok) == (48.46, 8.41)

    blank = MagicMock()
    with pytest.raises(GeocodeNotFoundError, match="Start is empty"):
        geocode_endpoint(place="   ", label="Start", geocode_fn=blank)
    blank.assert_not_called()  # blank input must not trigger a lookup

    missing = MagicMock(return_value=None)
    with pytest.raises(GeocodeNotFoundError, match=r"Destination \('xyz'\)"):
        geocode_endpoint(place="xyz", label="Destination", geocode_fn=missing)

    down = MagicMock(side_effect=GeocoderUnavailable("no dns"))
    with pytest.raises(GeocodeConnectionError, match=r"Start \('Baiersbronn'\)") as conn:
        geocode_endpoint(place="Baiersbronn", label="Start", geocode_fn=down)
    assert isinstance(conn.value, BikeRouterError) and not isinstance(conn.value, GeocodeNotFoundError)


# --- Photon typeahead / labels / reverse -------------------------------------


def test_photon_label():
    # "Name, City, State"; a part equal to the name is not repeated; blank parts skipped.
    assert photon_label(properties={"name": "Baiersbronn", "city": "Baiersbronn", "state": "BW"}) == "Baiersbronn, BW"
    assert photon_label(properties={"name": "Pforzheim", "state": "BW"}) == "Pforzheim, BW"
    assert photon_label(properties={"name": "Nowhere"}) == "Nowhere"  # no trailing commas


def test_photon_autocomplete():
    # Maps features → labels in order; a blank term returns [] without a request; no results / a
    # network error → []; and the request carries the bbox + centre-bias params.
    payload = {
        "features": [
            _photon_feature(
                name="Freudenstadt", lon=8.4116, lat=48.4633, city="Freudenstadt", state="Baden-Württemberg"
            ),
            _photon_feature(name="Pforzheim", lon=8.6947, lat=48.8922, state="Baden-Württemberg"),
        ]
    }
    assert photon_autocomplete(term="Freud", bbox=_BBOX, http_get=MagicMock(return_value=payload)) == [
        "Freudenstadt, Baden-Württemberg",
        "Pforzheim, Baden-Württemberg",
    ]

    blank_get = MagicMock()
    assert photon_autocomplete(term="   ", bbox=_BBOX, http_get=blank_get) == []
    blank_get.assert_not_called()  # a blank term must never hit the network

    assert photon_autocomplete(term="zzzz", bbox=_BBOX, http_get=MagicMock(return_value={"features": []})) == []
    boom = MagicMock(side_effect=requests.RequestException("timeout"))
    assert photon_autocomplete(term="Freud", bbox=_BBOX, http_get=boom) == []  # never crashes on a weak link

    params_get = MagicMock(return_value={"features": []})
    photon_autocomplete(term="Freud", bbox=_BBOX, limit=7, http_get=params_get)
    params = params_get.call_args.kwargs["params"]
    assert params["bbox"] == "8.3,48.4,8.8,48.95"
    assert params["osm_tag"] == "place" and params["lang"] == "de" and params["limit"] == 7
    assert params["lon"] == pytest.approx((8.30 + 8.80) / 2) and params["lat"] == pytest.approx((48.40 + 48.95) / 2)


def test_nearest_place_name():
    # Returns ONLY the settlement name via the /reverse endpoint; no feature / blank name / a
    # network error → None (the caller drops that marker's label).
    payload = {"features": [_photon_feature(name="Baiersbronn", lon=8.37, lat=48.5, city="Freudenstadt")]}
    ok = MagicMock(return_value=payload)
    assert nearest_place_name(lat=48.5, lon=8.37, http_get=ok) == "Baiersbronn"
    assert ok.call_args.kwargs["url"].endswith("/reverse")  # reverse endpoint, not /api

    assert nearest_place_name(lat=0.0, lon=0.0, http_get=MagicMock(return_value={"features": []})) is None
    blank = MagicMock(return_value={"features": [_photon_feature(name="", lon=0.0, lat=0.0)]})
    assert nearest_place_name(lat=0.0, lon=0.0, http_get=blank) is None
    boom = MagicMock(side_effect=requests.RequestException("timeout"))
    assert nearest_place_name(lat=0.0, lon=0.0, http_get=boom) is None
