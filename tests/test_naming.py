"""Output-naming tests — slugs, capped places, tuning suffix, place-stamped paths."""

import pytest

from bike_router.constants import OutputConfig
from bike_router.naming import params_suffix, route_basename, route_output_paths, short_place, slugify
from tests.conftest import DEFAULT_PARAMS, zero_params


def test_slugify_basic():
    assert slugify(text="Freudenstadt, Germany") == "freudenstadt_germany"
    assert slugify(text="  Bad Wildbad / Enz  ") == "bad_wildbad_enz"


def test_slugify_empty_raises():
    with pytest.raises(ValueError):
        slugify(text="!!! ,,,")


def test_short_place_caps_to_leading_groups():
    # First group kept when it's >3 chars; else the first two groups.
    assert short_place(text="Pforzheim, Baden-Württemberg") == "pforzheim"
    assert short_place(text="Bad Wildbad, Baden-Württemberg") == "bad_wildbad"  # "bad" ≤ 3 → keep two
    assert short_place(text="Horb am Neckar") == "horb"


def test_params_suffix_orders_and_formats_knobs():
    # Five knobs in PARAM_SPECS order; each value rounded to 3 dp with dot→dash.
    assert params_suffix(params=zero_params(extra_km_per_uphill_100m=12.0)) == (
        "uphill-12-000_unpaved-0-000_main-0-000_rail-0-000_boarding-0-000"
    )


def test_route_basename_has_both_places_and_tuning():
    stem = route_basename(origin="Pforzheim, Germany", destination="Bad Wildbad, Germany", params=DEFAULT_PARAMS)
    assert stem.startswith("pforzheim__to__bad_wildbad__")
    assert stem.count("__to__") == 1
    assert params_suffix(params=DEFAULT_PARAMS) in stem


def test_route_output_paths_under_output_dir():
    gpx, png = route_output_paths(origin="Freudenstadt", destination="Pforzheim", params=DEFAULT_PARAMS)
    assert gpx.parent == OutputConfig.OUTPUT_DIR
    assert gpx.suffix == ".gpx" and png.suffix == ".png"
    assert gpx.stem == png.stem == route_basename(origin="Freudenstadt", destination="Pforzheim", params=DEFAULT_PARAMS)


def test_route_output_paths_differ_by_tuning():
    # Two runs that differ only by a slider get distinct filenames (nothing overwritten).
    a = route_output_paths(origin="A town", destination="B town", params=zero_params(extra_km_per_uphill_100m=5.0))[0]
    b = route_output_paths(origin="A town", destination="B town", params=zero_params(extra_km_per_uphill_100m=9.0))[0]
    assert a.name != b.name
