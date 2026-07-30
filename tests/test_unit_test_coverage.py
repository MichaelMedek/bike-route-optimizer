"""STRICT 1:1 test↔production mapping — every symbol is tested, every test file is anchored.

Parses each module under bike_router/{core,ui,preprocessing} (AST, no execution) and asserts,
for the mirror test file tests/<layer>/test_<module>.py:
  * every module-level function ``foo`` / ``_foo`` has a test function ``test_foo``.
  * every module-level class ``Foo`` has a test class ``TestFoo``.
  * (reverse) every test file maps 1:1 to a production module in the same layer — NO stray
    tests/<layer>/test_<x>.py without a bike_router/<layer>/<x>.py behind it.
Extra unit AND integration tests may pile into a mapped file, but there is no home for a test
that isn't anchored to one production module — forcing strict 1:1 file adherence.
"""

import ast
import pathlib

import pytest
from docstring_parser import DocstringStyle, parse

from bike_router.core.constants import PROJECT_ROOT

_PKG = PROJECT_ROOT / "bike_router"
_TESTS_DIR = PROJECT_ROOT / "tests"
_LAYERS = ("core", "ui", "preprocessing")
_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _module_defs(path: pathlib.Path, node_types: tuple[type, ...]) -> list[str]:
    """Names of module-level defs of the given AST node types (nested defs ignored)."""
    return [n.name for n in ast.parse(path.read_text()).body if isinstance(n, node_types)]


def _production_modules() -> list[tuple[str, pathlib.Path]]:
    """(layer, path) for every real module under bike_router/{core,ui,preprocessing} (no __init__)."""
    return [
        (layer, path) for layer in _LAYERS for path in sorted((_PKG / layer).glob("*.py")) if path.name != "__init__.py"
    ]


def _expected_test_file(layer: str, path: pathlib.Path) -> pathlib.Path:
    """The one test file a module's symbols must be tested in: tests/<layer>/test_<module>.py."""
    return _TESTS_DIR / layer / f"test_{path.name}"


def _test_symbols(test_file: pathlib.Path, node_types: tuple[type, ...]) -> set[str]:
    """Module-level def names in a test file, or empty set if the file is absent."""
    if not test_file.exists():
        return set()
    return set(_module_defs(test_file, node_types))


# One parametrize entry per production symbol → itemized pass/fail, not one lumped assert.
_FUNCTIONS = [(lay, p, name) for lay, p in _production_modules() for name in _module_defs(p, _FUNC_TYPES)]
_CLASSES = [(lay, p, name) for lay, p in _production_modules() for name in _module_defs(p, (ast.ClassDef,))]
_FN_IDS = [f"{lay}/{p.stem}::{name}" for lay, p, name in _FUNCTIONS]
_CLS_IDS = [f"{lay}/{p.stem}::{name}" for lay, p, name in _CLASSES]


def _test_files() -> list[tuple[str, pathlib.Path]]:
    """(layer, path) for every test_*.py under tests/{core,ui,preprocessing}."""
    return [(layer, path) for layer in _LAYERS for path in sorted((_TESTS_DIR / layer).glob("test_*.py"))]


_TEST_FILE_IDS = [f"{lay}/{p.name}" for lay, p in _test_files()]

_DEF = (*_FUNC_TYPES, ast.ClassDef)  # every documentable production symbol
# (layer, module_path, node) for every production function/class → one docstring check each.
_SYMBOL_NODES = [
    (lay, p, node)
    for lay, p in _production_modules()
    for node in ast.parse(p.read_text()).body
    if isinstance(node, _DEF)
]
_SYMBOL_IDS = [f"{lay}/{p.stem}::{node.name}" for lay, p, node in _SYMBOL_NODES]


@pytest.mark.parametrize(("layer", "path", "func"), _FUNCTIONS, ids=_FN_IDS)
def test_every_production_function_has_a_unit_test(layer: str, path: pathlib.Path, func: str) -> None:
    """Free-floating ``func``/``_func`` must have a ``test_func`` in tests/<layer>/test_<module>.py."""
    test_file = _expected_test_file(layer=layer, path=path)
    expected = f"test_{func.lstrip('_')}"
    have = _test_symbols(test_file=test_file, node_types=_FUNC_TYPES)
    assert expected in have, (
        f"{layer}/{path.name}::{func}() has no dedicated unit test — "
        f"expected a function '{expected}' in tests/{layer}/{test_file.name}"
    )


@pytest.mark.parametrize(("layer", "path", "cls"), _CLASSES, ids=_CLS_IDS)
def test_every_production_class_has_a_unit_test(layer: str, path: pathlib.Path, cls: str) -> None:
    """Class ``Foo`` must have a ``TestFoo`` test class in tests/<layer>/test_<module>.py."""
    test_file = _expected_test_file(layer=layer, path=path)
    expected = f"Test{cls}"
    have = _test_symbols(test_file=test_file, node_types=(ast.ClassDef,))
    assert expected in have, (
        f"{layer}/{path.name}::{cls} has no dedicated test class — "
        f"expected a class '{expected}' in tests/{layer}/{test_file.name}"
    )


@pytest.mark.parametrize(("layer", "path"), _test_files(), ids=_TEST_FILE_IDS)
def test_test_file_maps_to_a_production_module(layer: str, path: pathlib.Path) -> None:
    """No stray tests: tests/<layer>/test_<x>.py requires bike_router/<layer>/<x>.py behind it."""
    module = _PKG / layer / path.name[len("test_") :]
    assert module.exists(), (
        f"tests/{layer}/{path.name} maps to no production module — "
        f"expected {module.relative_to(PROJECT_ROOT)}. Fold its cases into the matching "
        f"test_<module>.py; integration tests must live in a mapped file too."
    )


def test_tests_outnumber_production_symbols() -> None:
    """Total test funcs+classes ≥ 1.4× production funcs+classes — a real testing surplus, not parity."""
    prod = sum(len(_module_defs(p, _FUNC_TYPES + (ast.ClassDef,))) for _, p in _production_modules())
    tests = sum(len(_module_defs(p, _FUNC_TYPES + (ast.ClassDef,))) for _, p in _test_files())
    assert tests >= prod * 1.4, (
        f"only {tests} test funcs+classes for {prod} production ones (ratio {tests / prod:.2f}); "
        f"need ≥{prod * 1.4:.0f} (1.4×)"
    )


@pytest.mark.parametrize(("layer", "path", "node"), _SYMBOL_NODES, ids=_SYMBOL_IDS)
def test_docstring_is_terse_with_valid_google_args(layer: str, path: pathlib.Path, node: ast.AST) -> None:
    """Every production symbol: 1–3 description lines, plus (optional) a valid Google-style Args block.

    Uses docstring_parser: description = short + long, so Args/Returns/
    Raises sections are naturally excluded from the line budget; an ``Args:`` header that yields no
    parsed params is malformed and fails.
    """
    raw = ast.get_docstring(node, clean=True)
    assert raw, f"{layer}/{path.name}::{node.name} has no docstring (need 1–3 description lines)"
    parsed = parse(raw, style=DocstringStyle.GOOGLE)
    desc = "\n".join(part for part in (parsed.short_description, parsed.long_description) if part)
    n_lines = len(desc.splitlines())
    assert 1 <= n_lines <= 3, (
        f"{layer}/{path.name}::{node.name} docstring has {n_lines} description lines; max 3 "
        f"(Args/Returns/Raises don't count)"
    )
    assert not ("Args:" in raw and not parsed.params), (
        f"{layer}/{path.name}::{node.name} has an 'Args:' header but no valid Google-style params under it"
    )
