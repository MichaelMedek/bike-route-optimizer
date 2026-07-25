"""Output-path naming: filenames carry the start+end places AND the route
profile postfix, so all four route variants are clearly recognisable.

e.g. origin "Freudenstadt, Germany" + dest "Pforzheim" + profile "flattest" →
output/freudenstadt_germany__to__pforzheim__flattest.gpx
"""

import re
from pathlib import Path

from bike_router.constants import OutputConfig, RouteProfile


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


def route_basename(origin: str, destination: str, profile: RouteProfile) -> str:
    """`<origin>__to__<destination>__<profile>` stem from both place slugs."""
    stem = f"{slugify(text=origin)}__to__{slugify(text=destination)}__{profile.postfix}"
    assert stem.count("__to__") == 1, "basename must contain exactly one '__to__' separator"
    return stem


def route_output_paths(origin: str, destination: str, profile: RouteProfile) -> tuple[Path, Path]:
    """(gpx_path, png_path) under OUTPUT_DIR, stamped with places + profile."""
    stem = route_basename(origin=origin, destination=destination, profile=profile)
    gpx_path = OutputConfig.OUTPUT_DIR / f"{stem}.gpx"
    png_path = OutputConfig.OUTPUT_DIR / f"{stem}.png"
    assert gpx_path.suffix == ".gpx" and png_path.suffix == ".png", "output suffixes must match type"
    return gpx_path, png_path
