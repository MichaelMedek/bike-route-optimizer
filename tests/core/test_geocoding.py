"""geocoding tests — Nominatim resolve + Photon typeahead/reverse, fully mocked (zero network).

One test_<fn> per production symbol (exact-name mirror) and TestHttpGetter for the Protocol.
Each folds every scenario for its target; the geopy geocoder and the injectable HTTP getter are
mocked so nothing touches the network.
"""

from unittest.mock import MagicMock

import pytest
import requests
from geopy.exc import GeocoderServiceError, GeocoderUnavailable

from bike_router.core.constants import NominatimConfig, PhotonConfig
from bike_router.core.errors import BikeRouterError, GeocodeConnectionError, GeocodeNotFoundError
from bike_router.core.geocoding import (
    _GEOCODE_CACHE,
    HttpGetter,
    _feature_lonlat,
    _feature_name,
    _feature_properties,
    _parse_latlon,
    _photon_features,
    _photon_query,
    as_bahnhof,
    autocomplete_with_stations,
    bahnhof_suggestion,
    box_display_label,
    default_http_get,
    geocode,
    geocode_endpoint,
    latlon_box_value,
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
    payload = default_http_get(url="https://photon/api", params={"q": "x"}, timeout=2.0)
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

    # A "lat, lon" literal (from the GPS button) resolves DIRECTLY — no Nominatim call.
    gps = MagicMock()
    assert geocode(place="48.4633, 8.4116", geocode_fn=gps) == (48.4633, 8.4116)
    gps.assert_not_called()  # coordinates never hit the network


def test_parse_latlon():
    # Two in-range comma floats → tuple (spaces optional); anything else → None (falls to Nominatim).
    assert _parse_latlon(place="48.4633, 8.4116") == (48.4633, 8.4116)
    assert _parse_latlon(place="48.46,8.41") == (48.46, 8.41)  # no space
    assert _parse_latlon(place="47.5, 9.5 (Zürich Bahnhof)") == (47.5, 9.5)  # trailing (Name) ignored — coords win
    assert _parse_latlon(place="Freudenstadt, Germany") is None  # a place, not coords
    assert _parse_latlon(place="200, 8") is None  # lat out of range
    assert _parse_latlon(place="48, 8, 9") is None  # wrong part count
    assert _parse_latlon(place="Freudenstadt") is None  # no comma


def test_latlon_box_value():
    # The ONE "lat, lon (Name)" box format _parse_latlon reads back; name optional (explicit) + stripped.
    assert latlon_box_value(lat=47.5, lon=9.5, name="Zürich Bahnhof") == "47.50000, 9.50000 (Zürich Bahnhof)"
    assert latlon_box_value(lat=47.5, lon=9.5, name=None) == "47.50000, 9.50000"  # no name → bare coords
    assert latlon_box_value(lat=47.5, lon=9.5, name="  ") == "47.50000, 9.50000"  # blank name dropped
    assert _parse_latlon(place=latlon_box_value(lat=47.5, lon=9.5, name="X")) == (47.5, 9.5)  # round-trips to coords


def test_marker_pick_geocodes_to_exact_coords_not_name():
    # REGRESSION (bug 6/5): a valid marker/station pick must resolve to its EXACT clicked point via a
    # single fill→geocode chain — never re-geocoding the (unresolvable) name (Schalkstetten, Zürich).
    box_value = latlon_box_value(lat=47.98765, lon=9.12345, name="Schalkstetten Bahnhof")
    never = MagicMock(side_effect=AssertionError("must NOT hit the network — coords resolve directly"))
    assert geocode(place=box_value, geocode_fn=never) == (47.98765, 9.12345)  # coords win, name ignored, no lookup


def test_zurich_station_pick_resolves_via_coords():
    # REGRESSION (bug 5): "Zürich" often has no geocodable "<name> Bahnhof", so the station pick must
    # carry the matched station's OWN coords — picking it then resolves EXACTLY there, no name lookup.
    station = MagicMock(return_value={"features": [_photon_feature(name="Zürich HB", lon=8.54, lat=47.378)]})
    pick = bahnhof_suggestion(term="Zürich", bbox=_BBOX, http_get=station)
    assert pick == "47.37800, 8.54000 (Zürich HB Bahnhof)"
    never = MagicMock(side_effect=AssertionError("station pick must resolve by coords, not re-geocode the name"))
    assert geocode(place=pick, geocode_fn=never) == (47.378, 8.54)  # the exact station point


def test_box_display_label():
    # Shows the (Name) inside a coords literal, else the value verbatim (a plain place name).
    assert box_display_label("47.50000, 9.50000 (Zürich Bahnhof)") == "Zürich Bahnhof"
    assert box_display_label("Freudenstadt, Germany") == "Freudenstadt, Germany"  # not a coords literal → as-is
    assert box_display_label("47.50000, 9.50000") == "47.50000, 9.50000"  # coords, no name → the value


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
    assert photon_autocomplete(
        term="Freud",
        bbox=_BBOX,
        limit=PhotonConfig.LIMIT,
        osm_tag=PhotonConfig.PLACE_OSM_TAG,
        http_get=MagicMock(return_value=payload),
    ) == [
        "Freudenstadt, Baden-Württemberg",
        "Pforzheim, Baden-Württemberg",
    ]

    blank_get = MagicMock()
    assert (
        photon_autocomplete(
            term="   ", bbox=_BBOX, limit=PhotonConfig.LIMIT, osm_tag=PhotonConfig.PLACE_OSM_TAG, http_get=blank_get
        )
        == []
    )
    blank_get.assert_not_called()  # a blank term must never hit the network

    assert (
        photon_autocomplete(
            term="zzzz",
            bbox=_BBOX,
            limit=PhotonConfig.LIMIT,
            osm_tag=PhotonConfig.PLACE_OSM_TAG,
            http_get=MagicMock(return_value={"features": []}),
        )
        == []
    )
    boom = MagicMock(side_effect=requests.RequestException("timeout"))
    # never crashes on a weak link
    assert (
        photon_autocomplete(
            term="Freud", bbox=_BBOX, limit=PhotonConfig.LIMIT, osm_tag=PhotonConfig.PLACE_OSM_TAG, http_get=boom
        )
        == []
    )

    params_get = MagicMock(return_value={"features": []})
    photon_autocomplete(term="Freud", bbox=_BBOX, limit=7, osm_tag=PhotonConfig.PLACE_OSM_TAG, http_get=params_get)
    params = params_get.call_args.kwargs["params"]
    assert params["bbox"] == "8.3,48.4,8.8,48.95"
    assert params["osm_tag"] == "place" and params["lang"] == "de" and params["limit"] == 7
    assert params["lon"] == pytest.approx((8.30 + 8.80) / 2) and params["lat"] == pytest.approx((48.40 + 48.95) / 2)


def test_photon_autocomplete_station_tag():
    # The osm_tag override reaches Photon so a station-only query can be issued.
    get = MagicMock(return_value={"features": []})
    photon_autocomplete(
        term="Langenargen", bbox=_BBOX, limit=PhotonConfig.LIMIT, osm_tag=PhotonConfig.STATION_OSM_TAG, http_get=get
    )
    assert get.call_args.kwargs["params"]["osm_tag"] == PhotonConfig.STATION_OSM_TAG


def test_as_bahnhof():
    # Appends " Bahnhof" so a bare station name geocodes to the platform, not the town centre;
    # a name already ending in "Bahnhof" (any case) is not doubled; surrounding space is trimmed.
    assert as_bahnhof(name="Sauldorf") == "Sauldorf Bahnhof"
    assert as_bahnhof(name="  Langenargen  ") == "Langenargen Bahnhof"
    assert as_bahnhof(name="Zürich Flughafen Bahnhof") == "Zürich Flughafen Bahnhof"
    assert as_bahnhof(name="Konstanz BAHNHOF") == "Konstanz BAHNHOF"


def test_photon_query():
    # The ONE Photon call site: returns the payload's features; ANY network/HTTP error → [] (never crashes).
    payload = {"features": [_photon_feature(name="X", lon=9.0, lat=48.0)]}
    got = _photon_query(url="https://photon/reverse", params={"lat": 48.0}, http_get=MagicMock(return_value=payload))
    assert len(got) == 1 and got[0]["properties"]["name"] == "X"
    boom = MagicMock(side_effect=requests.RequestException("timeout"))
    assert _photon_query(url="https://photon/reverse", params={"lat": 48.0}, http_get=boom) == []


def test_photon_features():
    # The ONE raw-feature query: returns the payload's features; blank term / any error → [].
    payload = {"features": [_photon_feature(name="Langenargen", lon=9.55, lat=47.6)]}
    got = _photon_features(
        term="Lang", bbox=_BBOX, limit=1, osm_tag=PhotonConfig.STATION_OSM_TAG, http_get=MagicMock(return_value=payload)
    )
    assert len(got) == 1 and got[0]["properties"]["name"] == "Langenargen"
    blank = MagicMock()
    assert _photon_features(term="  ", bbox=_BBOX, limit=1, osm_tag="x", http_get=blank) == []
    blank.assert_not_called()
    boom = MagicMock(side_effect=requests.RequestException("timeout"))
    assert _photon_features(term="Lang", bbox=_BBOX, limit=1, osm_tag="x", http_get=boom) == []


def test_feature_lonlat():
    # (lon, lat) from a Point geometry; malformed/absent geometry → None.
    assert _feature_lonlat(_photon_feature(name="X", lon=9.55, lat=47.6)) == (9.55, 47.6)
    assert _feature_lonlat({"properties": {"name": "X"}}) is None  # no geometry
    assert _feature_lonlat({"geometry": {"coordinates": [9.55]}}) is None  # not a pair


def test_feature_properties():
    # The ONE typed properties accessor: the dict when present, {} when absent/malformed.
    assert _feature_properties(_photon_feature(name="X", lon=9.0, lat=48.0)) == {"name": "X"}
    assert _feature_properties({"geometry": {}}) == {}  # no properties
    assert _feature_properties({"properties": "nope"}) == {}  # not a dict


def test_feature_name():
    # The stripped feature name, or "" when absent/malformed.
    assert _feature_name(_photon_feature(name="  Langenargen ", lon=9.0, lat=48.0)) == "Langenargen"
    assert _feature_name({"properties": {}}) == ""  # no name
    assert _feature_name({}) == ""  # no properties at all


def test_bahnhof_suggestion():
    # A station match → a "lat, lon (Name Bahnhof)" box value using the station's OWN coordinates
    # (station-first OSM query); no station → None; a name already ending in "Bahnhof" isn't doubled.
    def station(*, url, params, timeout):  # noqa: ANN001, ANN202
        assert params["osm_tag"] == PhotonConfig.STATION_OSM_TAG  # station-first query
        return {"features": [_photon_feature(name="Langenargen", lon=9.55, lat=47.6)]}

    assert (
        bahnhof_suggestion(term="Langenargen", bbox=_BBOX, http_get=station)
        == "47.60000, 9.55000 (Langenargen Bahnhof)"
    )

    none = MagicMock(return_value={"features": []})
    assert bahnhof_suggestion(term="zzz", bbox=_BBOX, http_get=none) is None

    already = MagicMock(
        return_value={"features": [_photon_feature(name="Zürich Flughafen Bahnhof", lon=8.5, lat=47.5)]}
    )
    assert (
        bahnhof_suggestion(term="Zürich", bbox=_BBOX, http_get=already)
        == "47.50000, 8.50000 (Zürich Flughafen Bahnhof)"
    )


def test_autocomplete_with_stations():
    # (bahnhof_box_value, place_labels): the red-button station pick (coords + Bahnhof name) leads; the
    # ordinary settlement suggestions follow. A station match → a "lat, lon (Name Bahnhof)" pick.
    def by_tag(*, url, params, timeout):  # noqa: ANN001, ANN202  # station vs place mock
        return {"features": [_photon_feature(name="Langenargen", lon=9.55, lat=47.6)]}

    bahnhof, places = autocomplete_with_stations(term="Langenargen", bbox=_BBOX, limit=7, http_get=by_tag)
    assert bahnhof == "47.60000, 9.55000 (Langenargen Bahnhof)"  # coords-carrying red-button pick
    assert places  # settlement suggestions still offered

    # No station → no Bahnhof pick; just the place suggestions.
    place_only = MagicMock(
        side_effect=lambda *, url, params, timeout: (
            {"features": []}
            if params["osm_tag"] == PhotonConfig.STATION_OSM_TAG
            else {"features": [_photon_feature(name="Xdorf", lon=9.0, lat=48.0)]}
        )
    )
    bahnhof, places = autocomplete_with_stations(term="Xdorf", bbox=_BBOX, limit=7, http_get=place_only)
    assert bahnhof is None and places == ["Xdorf"]


def test_nearest_place_name():
    # Returns ONLY the settlement name via the /reverse endpoint; no feature / blank name / a
    # network error → None (the caller drops that marker's label).
    payload = {"features": [_photon_feature(name="Baiersbronn", lon=8.37, lat=48.5, city="Freudenstadt")]}
    ok = MagicMock(return_value=payload)
    assert nearest_place_name(lat=48.5, lon=8.37, http_get=ok) == "Baiersbronn"
    assert ok.call_args.kwargs["url"].endswith("/reverse")  # reverse endpoint, not /api
    params = ok.call_args.kwargs["params"]
    assert params["radius"] == PhotonConfig.REVERSE_RADIUS_KM  # generous radius so a remote waypoint still names one
    # restricted to real settlement types (repeated osm_tag) → a town/village, never a postcode/farm
    assert params["osm_tag"] == PhotonConfig.REVERSE_PLACE_TAGS and params["limit"] == 1

    assert nearest_place_name(lat=0.0, lon=0.0, http_get=MagicMock(return_value={"features": []})) is None
    blank = MagicMock(return_value={"features": [_photon_feature(name="", lon=0.0, lat=0.0)]})
    assert nearest_place_name(lat=0.0, lon=0.0, http_get=blank) is None
    boom = MagicMock(side_effect=requests.RequestException("timeout"))
    assert nearest_place_name(lat=0.0, lon=0.0, http_get=boom) is None
