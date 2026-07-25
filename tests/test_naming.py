"""Output-naming tests — slugs + place-stamped paths."""

import pytest

from bike_router.constants import OutputConfig
from bike_router.naming import route_basename, route_output_paths, slugify


def test_slugify_basic():
    assert slugify(text="Freudenstadt, Germany") == "freudenstadt_germany"
    assert slugify(text="  Bad Wildbad / Enz  ") == "bad_wildbad_enz"


def test_slugify_empty_raises():
    with pytest.raises(ValueError):
        slugify(text="!!! ,,,")


def test_route_basename_contains_both_places():
    stem = route_basename(origin="Freudenstadt, Germany", destination="Pforzheim, Germany")
    assert stem == "freudenstadt_germany__to__pforzheim_germany"


def test_route_output_paths_under_output_dir():
    gpx, png = route_output_paths(origin="Freudenstadt", destination="Pforzheim")
    assert gpx.parent == OutputConfig.OUTPUT_DIR
    assert gpx.name == "freudenstadt__to__pforzheim.gpx"
    assert png.name == "freudenstadt__to__pforzheim.png"
