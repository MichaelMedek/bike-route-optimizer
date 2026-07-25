"""Eco- & surface-optimized bicycle route planner — CLI entrypoint.

Usage:
    python main.py "Freudenstadt, Germany" "Pforzheim, Germany"

Computes four routes (flattest / shortest / smoothest / balanced), each written
to its own place+profile-stamped GPX and debug PNG. The DEM is downloaded from
Hugging Face on first run; override with --dem path/to/dem.tif.

Thin CLI shell only: argument parsing + calling imported package functions. All
pipeline logic lives in bike_router.pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

from bike_router.constants import DEMConfig
from bike_router.elevation import ensure_dem
from bike_router.geocoding import GeocodeError
from bike_router.pipeline import plan_routes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eco- & surface-optimized bicycle route planner.")
    parser.add_argument("origin", help='Start place, e.g. "Freudenstadt, Germany"')
    parser.add_argument("destination", help='Destination place, e.g. "Pforzheim, Germany"')
    parser.add_argument(
        "--dem",
        type=Path,
        default=DEMConfig.EURODEM_PATH,
        help="Path to the DEM GeoTIFF (default: auto-downloaded from Hugging Face on first run).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        dem_path = ensure_dem(dem_path=arguments.dem)
    except (FileNotFoundError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2

    try:
        results = plan_routes(origin=arguments.origin, destination=arguments.destination, dem_path=dem_path)
    except GeocodeError as error:
        print(str(error), file=sys.stderr)
        return 1

    for result in results:
        stats = result.stats
        print(f"\n=== {result.profile.name} ===")
        print(
            f"  {stats.distance_km:.1f} km · {stats.duration_min:.0f} min · "
            f"+{stats.ascent_m:.0f} m / -{stats.descent_m:.0f} m"
        )
        print(f"  GPX: {result.gpx_path}")
        print(f"  Heatmap: {result.png_path}")
        print(f"  Google Maps: {result.gmaps_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
