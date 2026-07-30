"""STRICT import-boundary enforcement — the architectural firewall between the 3 layers.

Parses every module's imports (AST, no execution) and asserts:
  * core imports NOTHING from ui or preprocessing (it is the shared foundation).
  * preprocessing / ui import only from core (never each other).
  * core + ui import only third-party packages declared in requirements.txt (runtime);
    preprocessing may also use requirements-preprocessing.txt. So a stray build-only import
    (osmnx/pyrosm/geopandas/rasterio/networkx) into the runtime fails loud, keeping the
    deploy install lean and the layers honest.
"""

import ast
import pathlib
import sys

import pytest

from bike_router.core.constants import PROJECT_ROOT

_PKG = PROJECT_ROOT / "bike_router"
_LAYERS = ("core", "ui", "preprocessing")

# Which layers each layer is ALLOWED to import from (besides its own).
_ALLOWED = {
    "core": set(),  # core depends on neither sibling
    "ui": {"core"},  # UI reads core only
    "preprocessing": {"core"},  # build reads core only
}

# Import root → PyPI distribution name, only where they differ (else the root IS the name).
_IMPORT_TO_DIST = {
    "sklearn": "scikit-learn",
    "streamlit_deckgl": "streamlit-deckgl",
    "cv2": "opencv-python",
    "PIL": "pillow",
}


def _layer_modules() -> list[tuple[str, pathlib.Path]]:
    """(layer, path) for every .py file under bike_router/{core,ui,preprocessing}."""
    return [(layer, path) for layer in _LAYERS for path in (_PKG / layer).glob("*.py")]


def _import_roots(path: pathlib.Path) -> set[str]:
    """Top-level import names in a module (AST, nothing executed)."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
    return roots


def _imported_layers(path: pathlib.Path) -> set[str]:
    """The bike_router layers a module imports from."""
    layers: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        mods = []
        if isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module]
        elif isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        for mod in mods:
            parts = mod.split(".")
            if len(parts) >= 2 and parts[0] == "bike_router" and parts[1] in _LAYERS:
                layers.add(parts[1])
    return layers


def _declared_dists(requirements: pathlib.Path) -> set[str]:
    """PyPI distribution names declared in a requirements file (normalized, version stripped)."""
    dists: set[str] = set()
    for raw in requirements.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = line
        for sep in (">=", "==", "<=", "~=", ">", "<", "!="):
            name = name.split(sep, 1)[0]
        dists.add(name.strip().lower().replace("_", "-"))
    return dists


def _layered_dists(requirements: pathlib.Path) -> dict[str, str]:
    """Dist → layer tag (core|ui|dev) parsed from each line's ``# CORE:/UI:/DEV:`` comment.

    A runtime dep must be labelled so the test can check a module imports only its layer's deps
    (core deps are shared; ui deps are UI-only). Untagged non-option lines default to core.
    """
    layered: dict[str, str] = {}
    for raw in requirements.read_text().splitlines():
        code, _, comment = raw.partition("#")
        line = code.strip()
        if not line or line.startswith("-"):
            continue
        name = line
        for sep in (">=", "==", "<=", "~=", ">", "<", "!="):
            name = name.split(sep, 1)[0]
        dist = name.strip().lower().replace("_", "-")
        tag = comment.strip().split(":", 1)[0].strip().lower()
        layered[dist] = tag if tag in {"core", "ui", "dev"} else "core"
    return layered


_RUNTIME_LAYERED = _layered_dists(PROJECT_ROOT / "requirements.txt")
_CORE_DISTS = {d for d, t in _RUNTIME_LAYERED.items() if t == "core"}
_UI_DISTS = {d for d, t in _RUNTIME_LAYERED.items() if t == "ui"}
_PREPROCESSING_DISTS = _declared_dists(PROJECT_ROOT / "requirements-preprocessing.txt")
# What each MODULE layer may import: core deps are shared; ui deps are UI-only; preprocessing
# deps are build-only. (DEV-tagged deps — pytest etc. — are never imported by shipped modules.)
_LAYER_DISTS = {
    "core": _CORE_DISTS,
    "ui": _CORE_DISTS | _UI_DISTS,
    "preprocessing": _CORE_DISTS | _PREPROCESSING_DISTS,
}


def _third_party_dists(path: pathlib.Path) -> set[str]:
    """Third-party PyPI distributions a module imports (stdlib + first-party filtered out).

    Every explicit import counts — even one transitively available (e.g. numpy via pandas)
    must be declared in its own right, so the requirements file lists what the code truly needs.
    """
    dists: set[str] = set()
    for root in _import_roots(path):
        if root in sys.stdlib_module_names or root == "bike_router":
            continue
        dists.add(_IMPORT_TO_DIST.get(root, root).lower().replace("_", "-"))
    return dists


@pytest.mark.parametrize(("layer", "path"), _layer_modules(), ids=[f"{lay}/{p.name}" for lay, p in _layer_modules()])
def test_module_respects_layer_boundaries(layer: str, path: pathlib.Path) -> None:
    """Every module imports only from its own layer + the layers _ALLOWED lists — else fail loud."""
    permitted = _ALLOWED[layer] | {layer}
    illegal = _imported_layers(path) - permitted
    assert not illegal, f"{layer}/{path.name} illegally imports from {sorted(illegal)} (allowed: {sorted(permitted)})"


def test_core_never_imports_ui_or_preprocessing() -> None:
    """The headline invariant, stated once over the WHOLE core layer (not just per-file)."""
    forbidden = {"ui", "preprocessing"}
    offenders = {
        path.name: sorted(_imported_layers(path) & forbidden)
        for layer, path in _layer_modules()
        if layer == "core" and (_imported_layers(path) & forbidden)
    }
    assert not offenders, f"core modules importing ui/preprocessing: {offenders}"


@pytest.mark.parametrize(("layer", "path"), _layer_modules(), ids=[f"{lay}/{p.name}" for lay, p in _layer_modules()])
def test_every_import_is_a_declared_requirement(layer: str, path: pathlib.Path) -> None:
    """Every third-party import in a module is declared — and tagged — for that module's layer.

    core modules may import only CORE-tagged deps; ui modules CORE|UI; preprocessing CORE|build.
    EVERY explicit import must be listed in its own right (even numpy, though it ships with
    pandas), and a runtime module importing an undeclared/build-only lib (networkx/osmnx/…)
    fails loud — so the shipped install can never silently need something unlisted.
    """
    undeclared = _third_party_dists(path) - _LAYER_DISTS[layer]
    assert not undeclared, (
        f"{layer}/{path.name} imports {sorted(undeclared)} — not declared/allowed for the {layer} layer. "
        f"Add each to requirements{'' if layer != 'preprocessing' else '-preprocessing'}.txt "
        f"(with the correct # {layer.upper()}: tag)."
    )


# --- test-tree layout: test files themselves live under tests/{core,ui,preprocessing}/ --------

_TESTS_DIR = pathlib.Path(__file__).resolve().parent
# Files allowed to sit at tests/ root (shared, not layer-specific).
_ROOT_ALLOWED = {"conftest.py", "test_import_boundaries.py"}


def _misplaced_test_files() -> list[str]:
    """Any test_*.py at tests/ root that isn't a shared file — it must live in a layer folder."""
    return sorted(p.name for p in _TESTS_DIR.glob("test_*.py") if p.name not in _ROOT_ALLOWED)


def test_test_files_are_organized_into_layer_folders() -> None:
    """Tests mirror the code: every test_*.py lives under tests/{core,ui,preprocessing}/.

    Only conftest.py + this boundary test may sit at tests/ root. A stray test at the top
    level fails loud, so the test tree stays as strictly layered as the package it exercises.
    """
    misplaced = _misplaced_test_files()
    assert not misplaced, f"test files must live in tests/(core|ui|preprocessing)/, found at root: {misplaced}"
    for layer in _LAYERS:
        assert (_TESTS_DIR / layer).is_dir(), f"missing tests/{layer}/ folder"


@pytest.mark.parametrize(
    ("layer", "path"),
    [(lay, p) for lay in _LAYERS for p in (_TESTS_DIR / lay).glob("test_*.py")],
    ids=[f"{lay}/{p.name}" for lay in _LAYERS for p in (_TESTS_DIR / lay).glob("test_*.py")],
)
def test_test_file_only_imports_its_own_layer(layer: str, path: pathlib.Path) -> None:
    """A test under tests/<layer>/ imports bike_router only from layers <layer> is allowed to use.

    So a core test never reaches into preprocessing/ui, and a ui test never into preprocessing —
    the same firewall the package obeys, now proven for the tests that guard it.
    """
    permitted = _ALLOWED[layer] | {layer}
    illegal = _imported_layers(path) - permitted
    assert not illegal, (
        f"tests/{layer}/{path.name} imports bike_router.{sorted(illegal)} (allowed: {sorted(permitted)})"
    )
