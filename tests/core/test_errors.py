"""errors tests — the exception hierarchy every handled failure derives from."""

import pytest

from bike_router.core.errors import (
    BikeRouterError,
    GeocodeConnectionError,
    GeocodeNotFoundError,
    NoRouteError,
    OutOfCoverageError,
    ParamOutOfRangeError,
    RouteTooLargeError,
    TripTooLongError,
    TripTooShortError,
)

# Every user-facing failure subclasses BikeRouterError (the ONE type callers catch), so the
# app/CLI can catch that base and nothing else. One parametrized check covers the whole tree.
_SUBCLASSES = [
    GeocodeConnectionError,
    GeocodeNotFoundError,
    TripTooShortError,
    TripTooLongError,
    OutOfCoverageError,
    NoRouteError,
    RouteTooLargeError,
    ParamOutOfRangeError,
]


class TestBikeRouterError:
    def test_is_the_catchable_base(self):
        # A bare Exception is NOT caught as BikeRouterError; the base itself is.
        assert issubclass(BikeRouterError, Exception)
        with pytest.raises(BikeRouterError, match="boom"):
            raise BikeRouterError("boom")

    @pytest.mark.parametrize("exc_type", _SUBCLASSES, ids=[c.__name__ for c in _SUBCLASSES])
    def test_every_subclass_is_a_bikerouter_error(self, exc_type):
        # Catching the base catches every handled failure — the contract the app relies on.
        with pytest.raises(BikeRouterError):
            raise exc_type("x")
        assert issubclass(exc_type, BikeRouterError)


def _raises_own_type(exc_type: type[BikeRouterError]) -> None:
    """A subclass raises as itself AND is catchable as the base (shared assertion)."""
    with pytest.raises(exc_type):
        raise exc_type("msg")
    assert issubclass(exc_type, BikeRouterError)


class TestGeocodeConnectionError:
    def test_raises_and_is_base(self):
        _raises_own_type(exc_type=GeocodeConnectionError)


class TestGeocodeNotFoundError:
    def test_raises_and_is_base(self):
        _raises_own_type(exc_type=GeocodeNotFoundError)


class TestTripTooShortError:
    def test_raises_and_is_base(self):
        _raises_own_type(exc_type=TripTooShortError)


class TestTripTooLongError:
    def test_raises_and_is_base(self):
        _raises_own_type(exc_type=TripTooLongError)


class TestOutOfCoverageError:
    def test_raises_and_is_base(self):
        _raises_own_type(exc_type=OutOfCoverageError)


class TestNoRouteError:
    def test_raises_and_is_base(self):
        _raises_own_type(exc_type=NoRouteError)


class TestRouteTooLargeError:
    def test_raises_and_is_base(self):
        _raises_own_type(exc_type=RouteTooLargeError)


class TestParamOutOfRangeError:
    def test_raises_and_is_base(self):
        _raises_own_type(exc_type=ParamOutOfRangeError)
