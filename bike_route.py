"""Eco- & surface-optimized bicycle route planner — CLI entrypoint.

Usage:
    python bike_route.py "Freudenstadt, Germany" "Pforzheim, Germany"
    python bike_route.py "A" "B" --extra_km_per_uphill_100m 100   # force a flat route

Computes one route tuned by five "extra km" preferences, written to a place-stamped
GPX + debug PNG. The prebuilt DACH bike+rail graph (downloaded from Hugging Face on
first run) bakes in elevation, so NO DEM is used at inference.

Thin CLI shell only: arg parsing + calling package functions; pipeline logic lives
in bike_router.core.pipeline.
"""

import argparse
import logging

from bike_router.core.constants import PARAM_SPECS, RoutingParams
from bike_router.core.graph_store import download_graph_from_hf
from bike_router.core.pipeline import format_cli_report, plan_route


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

    # snapshot_download prints its own "Fetching N files" progress to the terminal.
    download_graph_from_hf()
    # Expected failures raise a BikeRouterError — its class name + message already explain.
    params = RoutingParams(**{spec.field: getattr(arguments, spec.field) for spec in PARAM_SPECS})
    result = plan_route(origin=arguments.origin, destination=arguments.destination, params=params)
    print(format_cli_report(result=result))  # the whole stdout block is assembled + tested in core
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
