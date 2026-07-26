"""Default-parameter mode-choice tests: does the tuned rider bike or train each route?

Two layers:
  * a synthetic hill-vs-rail graph pinning the core contract (uphill→train, downhill→bike,
    no-rail→bike) deterministically and offline;
  * the GIVEN real-city routes as a FULL end-to-end run — real shipped dataset AND real OSM
    geocoding,(skipped only if the dataset isn't downloaded). Both
    directions: net-uphill takes the TRAIN, net-downhill / near-flat stays on the BIKE, and a
    pair with no train line bikes both ways.
"""

import networkx as nx
import pytest

from bike_router.constants import GraphConfig, Mode, NodeType
from bike_router.pipeline import plan_route
from bike_router.routing import shortest_route
from tests.conftest import DEFAULT_PARAMS, make_hill_vs_rail_graph


def _uses_train_path(graph: nx.MultiDiGraph, path: list[int]) -> bool:
    """True iff the route visits a rail-station node (boards a train)."""
    return any(graph.nodes[node]["node_type"] == NodeType.RAIL for node in path)


# (label, climb_m, rail_alternative, downhill, expect_train) — 10 synthetic scenarios.
_STEEP, _MILD = 150.0, 40.0
CASES = [
    ("steep uphill, train available → train", _STEEP, True, False, True),
    ("very steep uphill, train available → train", 250.0, True, False, True),
    ("steep uphill, NO train → bike", _STEEP, False, False, False),
    ("steep DOWNHILL, train available → bike", _STEEP, True, True, False),
    ("very steep downhill, train available → bike", 250.0, True, True, False),
    ("mild uphill, train available → bike", _MILD, True, False, False),
    ("mild uphill, NO train → bike", _MILD, False, False, False),
    ("mild downhill, train available → bike", _MILD, True, True, False),
    ("flat, train available → bike", 0.0, True, False, False),
    ("flat, NO train → bike", 0.0, False, False, False),
]


@pytest.mark.parametrize(("label", "climb_m", "rail", "downhill", "expect_train"), CASES, ids=[c[0] for c in CASES])
def test_default_params_pick_expected_mode(
    label: str,
    climb_m: float,
    rail: bool,  # noqa: FBT001
    downhill: bool,  # noqa: FBT001
    expect_train: bool,  # noqa: FBT001
) -> None:
    """With the DEFAULT params, the synthetic router bikes or trains as a sensible rider would."""
    graph = make_hill_vs_rail_graph(params=DEFAULT_PARAMS, climb_m=climb_m, rail_alternative=rail)
    source, target = (2, 1) if downhill else (1, 2)  # downhill = start at the high end
    path = shortest_route(graph=graph, source=source, target=target)
    assert _uses_train_path(graph=graph, path=path) is expect_train, f"{label}: got path {path}"


# --- Real routes: FULL e2e — real dataset + real OSM geocoding, nothing stubbed ------------

# GIVEN ground truth (origin, destination, expect_train) — real German towns, both directions.
_REAL_CASES = [
    ("Baiersbronn, Germany", "Freudenstadt, Germany", True),
    ("Freudenstadt, Germany", "Baiersbronn, Germany", False),
    ("Freudenstadt, Germany", "Pforzheim, Germany", False),
    ("Pforzheim, Germany", "Freudenstadt, Germany", True),
    ("Horb am Neckar, Germany", "Freudenstadt, Germany", True),
    ("Freudenstadt, Germany", "Horb am Neckar, Germany", False),
    ("Freudenstadt, Germany", "Nagold, Germany", False),
    ("Nagold, Germany", "Freudenstadt, Germany", True),
    ("Nagold, Germany", "Calw, Germany", False),
    ("Calw, Germany", "Nagold, Germany", False),
    ("Bad Wildbad, Germany", "Pforzheim, Germany", False),
    ("Pforzheim, Germany", "Bad Wildbad, Germany", True),
    ("Calw, Germany", "Pforzheim, Germany", False),
    ("Pforzheim, Germany", "Calw, Germany", False),
    ("Bad Wildbad, Germany", "Simmersfeld, Germany", False),
    ("Simmersfeld, Germany", "Bad Wildbad, Germany", False),
]


@pytest.mark.skipif(
    not (GraphConfig.GRAPH_DIR / GraphConfig.META_FILENAME).exists(),
    reason="real dataset not present in data/ (only the committed fixture is available)",
)
@pytest.mark.parametrize(
    ("origin", "destination", "expect_train"),
    _REAL_CASES,
    ids=[f"{o.split(',')[0]}->{d.split(',')[0]}" for o, d, _ in _REAL_CASES],
)
def test_default_params_real_route_mode(origin: str, destination: str, expect_train: bool) -> None:  # noqa: FBT001
    """FULL e2e: DEFAULT params, real dataset, real OSM geocoding — each route bikes or trains as given."""
    result = plan_route(origin=origin, destination=destination, params=DEFAULT_PARAMS)
    used_train = result.composition.by_mode_km.get(Mode.RAIL, 0.0) > 0.0
    assert used_train is expect_train
