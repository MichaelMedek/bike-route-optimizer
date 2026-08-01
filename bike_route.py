"""Eco- & surface-optimized bicycle route planner — CLI entrypoint.

Usage:
    python bike_route.py "Freudenstadt, Germany" "Pforzheim, Germany"
    python bike_route.py "A" "B" --extra_km_per_uphill_100m 100   # force a flat route

Entry script: argument parsing lives HERE; the graph download,
planning, and report assembly are the tested bike_router.core.pipeline.run_route.
"""

import argparse
import logging

from bike_router.core.constants import LOG_FORMAT, PARAM_SPECS, GraphConfig, RoutingParams
from bike_router.core.pipeline import run_route


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eco- & surface-optimized bicycle route planner.")
    parser.add_argument("origin", help='Start place, e.g. "Freudenstadt, Germany"')
    parser.add_argument("destination", help='Destination place, e.g. "Pforzheim, Germany"')
    for spec in PARAM_SPECS:  # one --flag per routing knob, straight from the shared specs
        parser.add_argument(f"--{spec.field}", type=float, default=spec.default, help=spec.help)
    parser.add_argument("-v", "--verbose", action="store_true")
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format=LOG_FORMAT,
    )
    # Expected failures raise a BikeRouterError — its class name + message already explain.
    params = RoutingParams(**{spec.field: getattr(arguments, spec.field) for spec in PARAM_SPECS})
    print(
        run_route(
            origin=arguments.origin, destination=arguments.destination, params=params, graph_dir=GraphConfig.GRAPH_DIR
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
