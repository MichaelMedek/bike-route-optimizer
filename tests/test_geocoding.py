"""Geocoding tests — fully mocked geopy, zero network (spec §6)."""

from unittest.mock import MagicMock

import pytest

from bike_router.geocoding import GeocodeError, geocode, geocode_endpoint


def test_geocode_returns_latlon():
    """A found location returns (lat, lon)."""
    loc = MagicMock()
    loc.latitude = 48.4633
    loc.longitude = 8.4116
    fake_geocode_fn = MagicMock(return_value=loc)

    lat, lon = geocode(place="Freudenstadt, Germany", geocode_fn=fake_geocode_fn)

    assert (lat, lon) == (48.4633, 8.4116)
    fake_geocode_fn.assert_called_once_with("Freudenstadt, Germany")


def test_geocode_not_found_raises():
    """None result (no match) raises GeocodeError."""
    fake_geocode_fn = MagicMock(return_value=None)
    with pytest.raises(GeocodeError):
        geocode(place="Nowhere at all, Atlantis", geocode_fn=fake_geocode_fn)


def test_geocode_service_error_raises():
    """A geopy service error is wrapped as GeocodeError."""
    from geopy.exc import GeocoderServiceError

    fake_geocode_fn = MagicMock(side_effect=GeocoderServiceError("boom"))
    with pytest.raises(GeocodeError):
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
    with pytest.raises(GeocodeError, match="Start is empty"):
        geocode_endpoint(place="   ", label="Start", geocode_fn=fake)
    fake.assert_not_called()  # blank input must not trigger a lookup


def test_geocode_endpoint_not_found_raises_field_named():
    fake = MagicMock(return_value=None)
    with pytest.raises(GeocodeError, match=r"Destination \('xyz'\)"):
        geocode_endpoint(place="xyz", label="Destination", geocode_fn=fake)
