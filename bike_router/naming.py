"""Output-path naming: filenames carry both places + the tuning knobs, for recognisability.

e.g. origin "Pforzheim, Baden-Württemberg" + dest "Bad Wildbad, …" with default params →
output/pforzheim__to__bad_wildbad__uphill-12-000_unpaved-1-000_main-1-000_rail-1-000_boarding-20-000.gpx
"""

import re
from pathlib import Path

from bike_router.constants import PARAM_SPECS, OutputConfig, RoutingParams

# Each PARAM_SPECS field → short filename token (order = the abbreviation order in the stem).
_PARAM_ABBREV = {
    "extra_km_per_uphill_100m": "uphill",
    "extra_km_per_unpaved_km": "unpaved",
    "extra_km_per_main_road_km": "main",
    "extra_km_per_rail_km": "rail",
    "extra_km_per_boarding": "boarding",
}
assert set(_PARAM_ABBREV) == {spec.field for spec in PARAM_SPECS}, "abbrev keys must match PARAM_SPECS fields"


def slugify(text: str) -> str:
    """Lowercase ASCII slug: non-alphanumerics collapse to single underscores.

    Args:
        text: Free-form place string.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    if not slug:
        raise ValueError(f"cannot slugify {text!r} — no alphanumeric characters")
    assert "__" not in slug, "single-underscore collapse invariant violated"
    return slug


def short_place(text: str) -> str:
    """Cap a place slug to its leading group(s): first group if >3 chars, else the first two.

    Drops the trailing admin qualifiers OSM appends, e.g. ``pforzheim_baden_w_rttemberg`` →
    ``pforzheim`` and ``bad_wildbad_baden_w_rttemberg`` → ``bad_wildbad``.
    """
    groups = slugify(text=text).split("_")
    kept = groups[:1] if len(groups[0]) > 3 else groups[:2]
    return "_".join(kept)


def _param_token(value: float) -> str:
    """Tuning value → ``12-000`` (rounded to 3 decimals, dot→dash for a filesystem-safe stem)."""
    return f"{value:.3f}".replace(".", "-")


def params_suffix(params: RoutingParams) -> str:
    """``uphill-12-000_unpaved-1-000_…`` — the 5 knobs in PARAM_SPECS order.

    Two runs that differ only by tuning get distinct filenames (nothing is overwritten).
    """
    return "_".join(f"{abbrev}-{_param_token(value=getattr(params, field))}" for field, abbrev in _PARAM_ABBREV.items())


def route_basename(origin: str, destination: str, params: RoutingParams) -> str:
    """`<origin>__to__<destination>__<tuning>` stem from both capped places + the knobs."""
    stem = f"{short_place(text=origin)}__to__{short_place(text=destination)}__{params_suffix(params=params)}"
    assert stem.count("__to__") == 1, "basename must contain exactly one '__to__' separator"
    return stem


def route_output_paths(origin: str, destination: str, params: RoutingParams) -> tuple[Path, Path]:
    """(gpx_path, png_path) under OUTPUT_DIR, stamped with both places + the tuning knobs."""
    stem = route_basename(origin=origin, destination=destination, params=params)
    gpx_path = OutputConfig.OUTPUT_DIR / f"{stem}.gpx"
    png_path = OutputConfig.OUTPUT_DIR / f"{stem}.png"
    assert gpx_path.suffix == ".gpx" and png_path.suffix == ".png", "output suffixes must match type"
    return gpx_path, png_path
