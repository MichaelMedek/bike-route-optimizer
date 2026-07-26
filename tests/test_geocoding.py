"""Geocoding tests — fully mocked geopy, zero network (spec §6)."""

from unittest.mock import MagicMock

import pytest

from bike_router.errors import BikeRouterError, GeocodeConnectionError, GeocodeNotFoundError
from bike_router.geocoding import geocode, geocode_endpoint


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
