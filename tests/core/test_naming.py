"""Output-naming tests — slugs, capped places, tuning suffix, place-stamped paths."""

import pytest

from bike_router.core.constants import OutputConfig
from bike_router.core.naming import (
    _param_token,
    params_suffix,
    route_basename,
    route_output_paths,
    short_place,
    slugify,
)
from tests.conftest import DEFAULT_PARAMS, zero_params


def test_slugify():
    # Lowercase ASCII slug; non-alphanumerics collapse to single underscores; all-punctuation raises.
    assert slugify(text="Freudenstadt, Germany") == "freudenstadt_germany"
    assert slugify(text="  Bad Wildbad / Enz  ") == "bad_wildbad_enz"
    with pytest.raises(ValueError):
        slugify(text="!!! ,,,")


def test_short_place():
    # First group kept when it's >3 chars; else the first two groups (drops admin qualifiers).
    assert short_place(text="Pforzheim, Baden-Württemberg") == "pforzheim"
    assert short_place(text="Bad Wildbad, Baden-Württemberg") == "bad_wildbad"  # "bad" ≤ 3 → keep two
    assert short_place(text="Horb am Neckar") == "horb"


def test_param_token():
    # A tuning value → 3-dp string with dot→dash (filesystem-safe).
    assert _param_token(value=12.0) == "12-000"
    assert _param_token(value=1.5) == "1-500"


def test_params_suffix():
    # Five knobs in PARAM_SPECS order, each _param_token-formatted.
    assert params_suffix(params=zero_params(extra_km_per_uphill_100m=12.0)) == (
        "uphill-12-000_unpaved-0-000_main-0-000_rail-0-000_boarding-0-000"
    )


def test_route_basename():
    stem = route_basename(origin="Pforzheim, Germany", destination="Bad Wildbad, Germany", params=DEFAULT_PARAMS)
    assert stem.startswith("pforzheim__to__bad_wildbad__") and stem.count("__to__") == 1
    assert params_suffix(params=DEFAULT_PARAMS) in stem


def test_route_output_paths():
    # (gpx, png) share the place+tuning stem under OUTPUT_DIR; a slider change yields a new name.
    gpx, png = route_output_paths(origin="Freudenstadt", destination="Pforzheim", params=DEFAULT_PARAMS)
    assert gpx.parent == OutputConfig.OUTPUT_DIR and gpx.suffix == ".gpx" and png.suffix == ".png"
    assert gpx.stem == png.stem == route_basename(origin="Freudenstadt", destination="Pforzheim", params=DEFAULT_PARAMS)
    a = route_output_paths(origin="A town", destination="B town", params=zero_params(extra_km_per_uphill_100m=5.0))[0]
    b = route_output_paths(origin="A town", destination="B town", params=zero_params(extra_km_per_uphill_100m=9.0))[0]
    assert a.name != b.name
