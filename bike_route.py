"""Eco- & surface-optimized bicycle route planner — CLI entrypoint.

Usage:
    python bike_route.py "Freudenstadt, Germany" "Pforzheim, Germany"
    python bike_route.py "A" "B" --extra_km_per_uphill_100m 100   # force a flat route

Computes one route tuned by five "extra km" preferences, written to a place-stamped
GPX + debug PNG. The prebuilt DACH bike+rail graph (downloaded from Hugging Face on
first run) bakes in elevation, so NO DEM is used at inference.

Thin CLI shell only: arg parsing + calling package functions; pipeline logic lives
in bike_router.pipeline.
"""

import argparse
import logging

from bike_router.composition import format_composition
from bike_router.constants import PARAM_SPECS, RoutingParams
from bike_router.graph_store import download_graph_from_hf
from bike_router.pipeline import plan_route
from bike_router.progress import tqdm_progress
from bike_router.simplify import format_bike_legs, format_rail_legs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eco- & surface-optimized bicycle route planner.")
    parser.add_argument("origin", help='Start place, e.g. "Freudenstadt, Germany"')
    parser.add_argument("destination", help='Destination place, e.g. "Pforzheim, Germany"')
    # One CLI flag per routing knob, straight from the shared PARAM_SPECS.
    for spec in PARAM_SPECS:
        parser.add_argument(f"--{spec.field}", type=float, default=spec.default, help=spec.help)
    parser.add_argument("-v", "--verbose", action="store_true")
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Progress belongs ONLY to the one-time graph download (tqdm here, st.progress in the app).
    download_graph_from_hf(progress=tqdm_progress(desc="Downloading graph"))
    # Expected failures raise a BikeRouterError — its class name + message already explain.
    params = RoutingParams(**{spec.field: getattr(arguments, spec.field) for spec in PARAM_SPECS})
    result = plan_route(
        origin=arguments.origin,
        destination=arguments.destination,
        params=params,
    )

    track = result.track
    print(f"\nTotal (bike + train): {track.total.oneline}")
    print(f"Bike only:            {track.bike.oneline}")
    print(format_composition(comp=result.composition))
    print(f"GPX: {result.gpx_path}")
    print(f"Heatmap: {result.png_path}")
    # Train rides: boarding + alighting station per ride, to search in a railway app.
    if result.rail_legs:
        print("Trains to catch:")
        for line in format_rail_legs(rail_legs=result.rail_legs):
            print(f"  {line}")
    # One bicycling link per pedalled leg, each labelled by its real endpoints (origin/
    # destination at the ends, station names where a train ride abuts the leg).
    print("Google Maps (one bike leg per train ride):")
    for label, leg in zip(format_bike_legs(bike_legs=result.bike_legs), result.bike_legs, strict=True):
        print(f"  {label}: {leg.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
