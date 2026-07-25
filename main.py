"""Eco- & surface-optimized bicycle route planner — CLI entrypoint.

Usage:
    python main.py "Freudenstadt, Germany" "Pforzheim, Germany"
    python main.py "A" "B" --extra_km_per_uphill_100m 100   # force a flat route

Computes one route tuned by three intuitive "extra km" preferences, written to a
place-stamped GPX + debug PNG. The DEM is downloaded from Hugging Face on first
run; override with --dem path/to/dem.tif.

Thin CLI shell only: argument parsing + calling imported package functions. All
pipeline logic lives in bike_router.pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

from bike_router.constants import PARAM_SPECS, DEMConfig, RoutingParams
from bike_router.elevation import ensure_dem
from bike_router.geocoding import GeocodeError
from bike_router.pipeline import plan_route
from bike_router.progress import tqdm_progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eco- & surface-optimized bicycle route planner.")
    parser.add_argument("origin", help='Start place, e.g. "Freudenstadt, Germany"')
    parser.add_argument("destination", help='Destination place, e.g. "Pforzheim, Germany"')
    # One CLI flag per routing knob, straight from the shared PARAM_SPECS.
    for spec in PARAM_SPECS:
        parser.add_argument(f"--{spec.field}", type=float, default=spec.default, help=spec.help)
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

    # RoutingParams validates ranges in __post_init__ and raises ValueError loudly.
    try:
        params = RoutingParams(**{spec.field: getattr(arguments, spec.field) for spec in PARAM_SPECS})
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    try:
        dem_path = ensure_dem(dem_path=arguments.dem)
    except (FileNotFoundError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2

    try:
        result = plan_route(
            origin=arguments.origin,
            destination=arguments.destination,
            dem_path=dem_path,
            params=params,
            progress=tqdm_progress(desc="Simplifying graph"),
        )
    except GeocodeError as error:
        print(str(error), file=sys.stderr)
        return 1

    track = result.track
    print(
        f"\nRoute: {track.distance_km:.1f} km · {track.duration_min:.0f} min · "
        f"+{track.ascent_m:.0f} m / -{track.descent_m:.0f} m"
    )
    print(f"GPX: {result.gpx_path}")
    print(f"Heatmap: {result.png_path}")
    print(f"Google Maps: {result.gmaps_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
