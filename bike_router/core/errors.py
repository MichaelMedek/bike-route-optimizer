"""Custom exceptions for every expected, user-facing failure.

One base (BikeRouterError) so callers catch exactly these; a bare builtin escaping means a real
bug, not a handled condition. Grouped: geocoding failures and route-planning failures.
"""


class BikeRouterError(Exception):
    """Base for every expected, user-facing failure. The ONLY type callers catch."""


class GeocodeConnectionError(BikeRouterError):
    """The geocoding service was unreachable (no internet / DNS / service down)."""


class GeocodeNotFoundError(BikeRouterError):
    """The service answered but no location matched the place string."""


class TripTooShortError(BikeRouterError):
    """Start and destination are closer than the minimum plannable distance."""


class TripTooLongError(BikeRouterError):
    """Start and destination are farther than the maximum plannable distance."""


class OutOfCoverageError(BikeRouterError):
    """An endpoint falls outside the prebuilt graph's bbox."""


class NoRouteError(BikeRouterError):
    """No path connects the two endpoints within the search corridor."""


class RouteTooLargeError(BikeRouterError):
    """The corridor would load more edges than the server's memory budget allows."""


class ParamOutOfRangeError(BikeRouterError):
    """A routing preference is outside the allowed [0, MAX_EXTRA_KM] range."""
