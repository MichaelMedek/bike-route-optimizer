"""Eco- & surface-optimized bicycle route planner — CLI entrypoint.

Usage:
    python main.py "Freudenstadt, Germany" "Pforzheim, Germany"
    python main.py "A" "B" --extra_km_per_uphill_100m 100   # force a flat route

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

    params = RoutingParams(**{spec.field: getattr(arguments, spec.field) for spec in PARAM_SPECS})
    download_graph_from_hf()  # prebuilt DACH bike+rail graph (once, cached)
    result = plan_route(
        origin=arguments.origin,
        destination=arguments.destination,
        params=params,
        progress=tqdm_progress(desc="Routing"),
    )

    track = result.track
    print(
        f"\nRoute: {track.distance_km:.1f} km · {track.duration_min:.0f} min · "
        f"+{track.ascent_m:.0f} m / -{track.descent_m:.0f} m"
    )
    print(format_composition(comp=result.composition))
    print(f"GPX: {result.gpx_path}")
    print(f"Heatmap: {result.png_path}")
    print(f"Google Maps: {result.gmaps_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
