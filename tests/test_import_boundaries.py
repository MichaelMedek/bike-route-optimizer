"""STRICT import-boundary enforcement — the architectural firewall between the 3 layers.

Parses every module's imports (AST, no execution) and the two requirements files, then asserts:
  * core imports NOTHING from ui or preprocessing (it is the shared foundation).
  * preprocessing / ui import only from core (never each other).
  * every third-party import is a declared requirement, tagged for the importing layer
    (core deps shared; ui deps UI-only; build deps preprocessing-only).
  * test files mirror the layout under tests/{core,ui,preprocessing}/ and obey the same rule.
  * every requirement line carries a ``# <LAYER>: reason`` justification.
"""

import ast
import pathlib
import sys

import pytest

from bike_router.core.constants import PROJECT_ROOT

_PKG = PROJECT_ROOT / "bike_router"
_TESTS_DIR = PROJECT_ROOT / "tests"
_LAYERS = ("core", "ui", "preprocessing")
_TAGS = {"core", "ui", "dev", "build"}
# Which layers each layer may import bike_router from (besides itself).
_ALLOWED = {"core": set(), "ui": {"core"}, "preprocessing": {"core"}}
# Which requirement TAGS each layer's third-party imports may draw from.
_LAYER_TAGS = {"core": {"core"}, "ui": {"core", "ui"}, "preprocessing": {"core", "build"}}
# Import root → PyPI distribution name, only where they differ.
_IMPORT_TO_DIST = {
    "sklearn": "scikit-learn",
    "streamlit_deckgl": "streamlit-deckgl",
    "cv2": "opencv-python",
    "PIL": "pillow",
}


def _norm(name: str) -> str:
    """PyPI distribution name → normalized comparison form (lowercase, dashes)."""
    return name.strip().lower().replace("_", "-")


# --- ONE requirements parser: (dist, tag, reason) per real line across BOTH files -------------


def _parse_requirements(requirements: pathlib.Path) -> list[tuple[str, str, str]]:
    """(dist, tag, reason) for each real requirement line (skips blanks + option lines -r/-e).

    tag is the lowercased word before ':' in the inline comment (core|ui|dev|build); reason is
    the text after it. An untagged line yields tag="" so the annotation test can flag it.
    """
    rows: list[tuple[str, str, str]] = []
    for raw in requirements.read_text().splitlines():
        code, _, comment = raw.partition("#")
        line = code.strip()
        if not line or line.startswith("-"):
            continue
        for sep in (">=", "==", "<=", "~=", ">", "<", "!="):
            line = line.split(sep, 1)[0]
        tag, _, reason = comment.strip().partition(":")
        rows.append((_norm(line), tag.strip().lower(), reason.strip()))
    return rows


_RUNTIME_REQS = _parse_requirements(PROJECT_ROOT / "requirements.txt")
_BUILD_REQS = _parse_requirements(PROJECT_ROOT / "requirements-preprocessing.txt")
# dist sets by requirement tag — what each MODULE layer's imports may draw from.
_DISTS_BY_TAG = {tag: {d for d, t, _ in _RUNTIME_REQS + _BUILD_REQS if t == tag} for tag in _TAGS}
_LAYER_DISTS = {layer: set().union(*(_DISTS_BY_TAG[t] for t in tags)) for layer, tags in _LAYER_TAGS.items()}


# --- ONE import extractor: every imported module string, via AST (nothing executed) -----------


def _imported_modules(path: pathlib.Path) -> list[str]:
    """Every module string imported by a file (absolute imports only; relative skipped)."""
    mods: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(alias.name for alias in node.names)
    return mods


def _imported_layers(path: pathlib.Path) -> set[str]:
    """The bike_router layers (core|ui|preprocessing) a file imports from."""
    return {
        parts[1]
        for mod in _imported_modules(path)
        if len(parts := mod.split(".")) >= 2 and parts[0] == "bike_router" and parts[1] in _LAYERS
    }


def _third_party_dists(path: pathlib.Path) -> set[str]:
    """Third-party PyPI distributions a file imports (stdlib + first-party filtered out).

    Every explicit import counts — even one transitively available (e.g. numpy via pandas) —
    so the requirements file must list what the code truly imports.
    """
    roots = {mod.split(".")[0] for mod in _imported_modules(path)}
    return {_norm(_IMPORT_TO_DIST.get(r, r)) for r in roots if r not in sys.stdlib_module_names and r != "bike_router"}


def _package_modules() -> list[tuple[str, pathlib.Path]]:
    """(layer, path) for every .py under bike_router/{core,ui,preprocessing}."""
    return [(layer, path) for layer in _LAYERS for path in (_PKG / layer).glob("*.py")]


def _test_modules() -> list[tuple[str, pathlib.Path]]:
    """(layer, path) for every test_*.py under tests/{core,ui,preprocessing}."""
    return [(layer, path) for layer in _LAYERS for path in (_TESTS_DIR / layer).glob("test_*.py")]


_PKG_IDS = [f"{lay}/{p.name}" for lay, p in _package_modules()]
_TEST_IDS = [f"{lay}/{p.name}" for lay, p in _test_modules()]


# --- the assertions ---------------------------------------------------------------------------


@pytest.mark.parametrize(("layer", "path"), _package_modules(), ids=_PKG_IDS)
def test_module_respects_layer_boundaries(layer: str, path: pathlib.Path) -> None:
    """Every module imports bike_router only from its own layer + the layers _ALLOWED lists."""
    illegal = _imported_layers(path) - (_ALLOWED[layer] | {layer})
    assert not illegal, f"{layer}/{path.name} illegally imports bike_router.{sorted(illegal)}"


@pytest.mark.parametrize(("layer", "path"), _package_modules(), ids=_PKG_IDS)
def test_every_import_is_a_declared_requirement(layer: str, path: pathlib.Path) -> None:
    """Every third-party import is a requirement tagged for this module's layer (else fail loud)."""
    undeclared = _third_party_dists(path) - _LAYER_DISTS[layer]
    assert not undeclared, (
        f"{layer}/{path.name} imports {sorted(undeclared)} — not declared/tagged for the {layer} layer."
    )


def test_test_files_are_organized_into_layer_folders() -> None:
    """Only the two cross-layer meta-tests may sit at tests/ root; every other test lives in a layer folder."""
    _ROOT_META = {"test_import_boundaries.py", "test_unit_test_coverage.py"}
    stray = sorted(p.name for p in _TESTS_DIR.glob("test_*.py") if p.name not in _ROOT_META)
    assert not stray, f"test files must live in tests/(core|ui|preprocessing)/, found at root: {stray}"
    assert all((_TESTS_DIR / layer).is_dir() for layer in _LAYERS), "missing a tests/<layer>/ folder"


@pytest.mark.parametrize(("layer", "path"), _test_modules(), ids=_TEST_IDS)
def test_test_file_only_imports_its_own_layer(layer: str, path: pathlib.Path) -> None:
    """A test under tests/<layer>/ imports bike_router only from layers <layer> may use."""
    illegal = _imported_layers(path) - (_ALLOWED[layer] | {layer})
    assert not illegal, f"tests/{layer}/{path.name} imports bike_router.{sorted(illegal)}"


@pytest.mark.parametrize("requirements", ["requirements.txt", "requirements-preprocessing.txt"])
def test_every_requirement_has_a_layer_annotation(requirements: str) -> None:
    """Every dependency line carries a ``# <LAYER>: reason`` justification (LAYER ∈ core|ui|dev|build)."""
    unannotated = [
        dist
        for dist, tag, reason in _parse_requirements(PROJECT_ROOT / requirements)
        if tag not in _TAGS or len(reason) < 3
    ]
    assert not unannotated, f"{requirements}: deps lacking a '# <LAYER>: reason' annotation: {unannotated}"
