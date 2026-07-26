"""Output-path naming: filenames carry start+end places for recognisability.

e.g. origin "Freudenstadt, Germany" + dest "Pforzheim" →
output/freudenstadt_germany__to__pforzheim.gpx
"""

import re
from pathlib import Path

from bike_router.constants import OutputConfig


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


def route_basename(origin: str, destination: str) -> str:
    """`<origin>__to__<destination>` stem from both place slugs."""
    stem = f"{slugify(text=origin)}__to__{slugify(text=destination)}"
    assert stem.count("__to__") == 1, "basename must contain exactly one '__to__' separator"
    return stem


def route_output_paths(origin: str, destination: str) -> tuple[Path, Path]:
    """(gpx_path, png_path) under OUTPUT_DIR, stamped with both places."""
    stem = route_basename(origin=origin, destination=destination)
    gpx_path = OutputConfig.OUTPUT_DIR / f"{stem}.gpx"
    png_path = OutputConfig.OUTPUT_DIR / f"{stem}.png"
    assert gpx_path.suffix == ".gpx" and png_path.suffix == ".png", "output suffixes must match type"
    return gpx_path, png_path
