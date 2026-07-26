"""Geocoding tests — fully mocked geopy, zero network (spec §6)."""

from unittest.mock import MagicMock

import pytest

from bike_router.errors import BikeRouterError, GeocodeConnectionError, GeocodeNotFoundError
from bike_router.geocoding import (
    geocode,
    geocode_endpoint,
    photon_autocomplete,
    photon_label,
)


def test_geocode_returns_latlon():
    """A found location returns (lat, lon)."""
    loc = MagicMock()
    loc.latitude = 48.4633
    loc.longitude = 8.4116
    fake_geocode_fn = MagicMock(return_value=loc)

    lat, lon = geocode(place="Freudenstadt, Germany", geocode_fn=fake_geocode_fn)

    assert (lat, lon) == (48.4633, 8.4116)
    fake_geocode_fn.assert_called_once_with("Freudenstadt, Germany")


def test_geocode_not_found_is_bikerouter_error():
    """None result (no match) is a GeocodeNotFoundError, distinct from a connection error."""
    fake_geocode_fn = MagicMock(return_value=None)
    with pytest.raises(GeocodeNotFoundError) as excinfo:
        geocode(place="Nowhere at all, Atlantis", geocode_fn=fake_geocode_fn)
    assert isinstance(excinfo.value, BikeRouterError)
    assert not isinstance(excinfo.value, GeocodeConnectionError)


def test_geocode_service_error_raises_connection_error():
    """A geopy service error (unreachable) is a GeocodeConnectionError, distinct from not-found."""
    from geopy.exc import GeocoderServiceError

    fake_geocode_fn = MagicMock(side_effect=GeocoderServiceError("boom"))
    with pytest.raises(GeocodeConnectionError, match="internet connection"):
        geocode(place="Somewhere", geocode_fn=fake_geocode_fn)


def test_make_geocode_fn_builds_rate_limited_callable():
    """make_geocode_fn returns a callable wrapping Nominatim (no network here)."""
    from bike_router.geocoding import make_geocode_fn

    fn = make_geocode_fn()
    assert callable(fn)


def test_geocode_endpoint_returns_latlon():
    loc = MagicMock()
    loc.latitude, loc.longitude = 48.46, 8.41
    fake = MagicMock(return_value=loc)
    assert geocode_endpoint(place="Freudenstadt", label="Start", geocode_fn=fake) == (48.46, 8.41)


def test_geocode_endpoint_blank_raises_field_named():
    fake = MagicMock()
    with pytest.raises(GeocodeNotFoundError, match="Start is empty"):
        geocode_endpoint(place="   ", label="Start", geocode_fn=fake)
    fake.assert_not_called()  # blank input must not trigger a lookup


def test_geocode_endpoint_not_found_raises_field_named():
    fake = MagicMock(return_value=None)
    with pytest.raises(GeocodeNotFoundError, match=r"Destination \('xyz'\)"):
        geocode_endpoint(place="xyz", label="Destination", geocode_fn=fake)


def test_geocode_endpoint_connection_error_is_distinct_type():
    """No-internet at an endpoint is a GeocodeConnectionError (distinct from not-found)."""
    from geopy.exc import GeocoderUnavailable

    fake = MagicMock(side_effect=GeocoderUnavailable("no dns"))
    with pytest.raises(GeocodeConnectionError, match=r"Start \('Baiersbronn'\)") as excinfo:
        geocode_endpoint(place="Baiersbronn", label="Start", geocode_fn=fake)
    assert isinstance(excinfo.value, BikeRouterError)  # the base the CLI/app catch
    assert not isinstance(excinfo.value, GeocodeNotFoundError)  # NOT the not-found type


# --- Photon search-as-you-type (zero network: inject the HTTP getter) ---------

_BBOX = (8.30, 48.40, 8.80, 48.95)  # (west, south, east, north)


def _photon_feature(name: str, lon: float, lat: float, **props: object) -> dict[str, object]:
    """A minimal Photon GeoJSON feature (coordinates are [lon, lat])."""
    return {"geometry": {"coordinates": [lon, lat]}, "properties": {"name": name, **props}}


def test_photon_autocomplete_maps_features_to_labels():
    """Two features → their display labels in order (labels only; box text is geocoded on submit)."""
    payload = {
        "features": [
            _photon_feature("Freudenstadt", 8.4116, 48.4633, city="Freudenstadt", state="Baden-Württemberg"),
            _photon_feature("Pforzheim", 8.6947, 48.8922, state="Baden-Württemberg"),
        ]
    }
    fake_get = MagicMock(return_value=payload)
    out = photon_autocomplete(term="Freud", bbox=_BBOX, http_get=fake_get)
    assert out == [
        "Freudenstadt, Baden-Württemberg",
        "Pforzheim, Baden-Württemberg",
    ]


def test_photon_autocomplete_blank_term_returns_empty_without_request():
    fake_get = MagicMock()
    assert photon_autocomplete(term="   ", bbox=_BBOX, http_get=fake_get) == []
    fake_get.assert_not_called()  # a blank term must never hit the network


def test_photon_autocomplete_no_results_returns_empty():
    fake_get = MagicMock(return_value={"features": []})
    assert photon_autocomplete(term="zzzz", bbox=_BBOX, http_get=fake_get) == []


def test_photon_autocomplete_network_error_returns_empty():
    """A weak/absent connection yields no suggestions — the typeahead never crashes."""
    import requests

    fake_get = MagicMock(side_effect=requests.RequestException("timeout"))
    assert photon_autocomplete(term="Freud", bbox=_BBOX, http_get=fake_get) == []


def test_photon_autocomplete_passes_bbox_and_center_params():
    fake_get = MagicMock(return_value={"features": []})
    photon_autocomplete(term="Freud", bbox=_BBOX, limit=7, http_get=fake_get)
    params = fake_get.call_args.kwargs["params"]
    assert params["bbox"] == "8.3,48.4,8.8,48.95"
    assert params["osm_tag"] == "place"
    assert params["lang"] == "de"
    assert params["limit"] == 7
    assert params["lon"] == pytest.approx((8.30 + 8.80) / 2)
    assert params["lat"] == pytest.approx((48.40 + 48.95) / 2)


def test_photon_label_formats_name_city_state():
    assert photon_label(properties={"name": "Baiersbronn", "city": "Baiersbronn", "state": "BW"}) == "Baiersbronn, BW"
    assert photon_label(properties={"name": "Pforzheim", "state": "BW"}) == "Pforzheim, BW"
    assert photon_label(properties={"name": "Nowhere"}) == "Nowhere"  # no trailing commas
