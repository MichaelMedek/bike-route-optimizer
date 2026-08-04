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
import re

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

# (layer, module_path, node) for EVERY function anywhere in a production module — module-level,
# nested, and methods — so the no-default-values rule below is checked without exception.
_ALL_FUNCTION_NODES = [
    (lay, p, node)
    for lay, p in _production_modules()
    for node in ast.walk(ast.parse(p.read_text()))
    if isinstance(node, _FUNC_TYPES)
]
_ALL_FUNCTION_IDS = [f"{lay}/{p.stem}::{node.name}" for lay, p, node in _ALL_FUNCTION_NODES]

# --- non-package .py discovery (ONE exclusion rule) -------------------------------------------
# tests/ + scripts/ are EXEMPT from the root-file rules below: their code legitimately DEFERS
# build-only deps (networkx/osmnx/graph_writer) so the runtime-only CI job (requirements.txt only)
# still collects; setup.py is packaging (defines nothing). .venv/.claude are not project code.
_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
_EXEMPT_DIRS = (_TESTS_DIR, _SCRIPTS_DIR)


def _repo_py_files(*, include_package: bool) -> list[pathlib.Path]:
    """Project .py files under the ONE exclusion rule; ``include_package`` keeps/drops bike_router/."""
    return [
        p
        for p in sorted(PROJECT_ROOT.rglob("*.py"))
        if not any(x in p.parts for x in (".venv", ".claude"))
        and not any(d in p.parents for d in _EXEMPT_DIRS)
        and p.name != "setup.py"
        and (include_package or _PKG not in p.parents)
    ]


def _function_nodes(paths: list[pathlib.Path]) -> list[tuple[pathlib.Path, ast.AST]]:
    """(path, node) for EVERY function (module-level, nested, method) across the given files."""
    return [(p, node) for p in paths for node in ast.walk(ast.parse(p.read_text())) if isinstance(node, _FUNC_TYPES)]


# The package + root entry scripts (tests/ + scripts/ exempt) — the ONE file list the
# no-in-function-imports and no-private-cross-import rules both scan; computed once here.
_PRODUCTION_PY = _repo_py_files(include_package=True)
_PRODUCTION_IDS = [str(p.relative_to(PROJECT_ROOT)) for p in _PRODUCTION_PY]
_REPO_FUNCTION_NODES = _function_nodes(_PRODUCTION_PY)
_REPO_FUNCTION_IDS = [f"{p.relative_to(PROJECT_ROOT)}::{node.name}" for p, node in _REPO_FUNCTION_NODES]

# entry-only-main rule: EVERY non-package root .py (dynamic, so a NEW script can't smuggle logic
# past the package's strict meta-tests) may define ONLY main().
_ENTRY_SCRIPTS = _repo_py_files(include_package=False)
_ENTRY_IDS = [str(p.relative_to(PROJECT_ROOT)) for p in _ENTRY_SCRIPTS]


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


@pytest.mark.parametrize(
    ("layer", "path"), _production_modules(), ids=[f"{lay}/{p.stem}" for lay, p in _production_modules()]
)
def test_module_docstring_is_terse(layer: str, path: pathlib.Path) -> None:
    """Every production module has a module docstring of 1–3 description lines (terse, like symbols)."""
    raw = ast.get_docstring(ast.parse(path.read_text()), clean=True)
    assert raw, f"{layer}/{path.name} has no module docstring (need 1–3 description lines)"
    parsed = parse(raw, style=DocstringStyle.GOOGLE)
    desc = "\n".join(part for part in (parsed.short_description, parsed.long_description) if part)
    n_lines = len(desc.splitlines())
    assert 1 <= n_lines <= 3, f"{layer}/{path.name} module docstring has {n_lines} description lines; max 3"


@pytest.mark.parametrize(("layer", "path", "node"), _ALL_FUNCTION_NODES, ids=_ALL_FUNCTION_IDS)
def test_no_default_argument_values(layer: str, path: pathlib.Path, node: ast.AST) -> None:
    """No function in core/ui/preprocessing may define ANY default argument value — WITHOUT EXCEPTION.

    Defaults belong ONLY in the entry scripts (bike_route.py / app_webmap.py); library code must force
    every caller to pass every argument explicitly, so a value is never silently assumed. Covers
    positional, keyword-only, *args, and **kwargs defaults for module-level fns, nested fns, and methods.
    """
    args = node.args
    offenders = args.defaults + [d for d in args.kw_defaults if d is not None]
    assert not offenders, (
        f"{layer}/{path.name}::{node.name} defines {len(offenders)} default argument value(s); "
        f"non-entry-script functions must define NONE — pass every argument explicitly at the call site"
    )


@pytest.mark.parametrize(("path", "node"), _REPO_FUNCTION_NODES, ids=_REPO_FUNCTION_IDS)
def test_no_imports_inside_functions(path: pathlib.Path, node: ast.AST) -> None:
    """No function in the package + root entry scripts may import inside its body — imports are module-level.

    tests/ and scripts/ are EXEMPT: their code legitimately DEFERS build-only deps (networkx/osmnx) so
    the runtime-only CI job (requirements.txt only) still collects. A deferred/local import elsewhere
    hides a dependency, delays import errors to call time, and usually papers over a fixable cycle.
    """
    inner = [
        f"{sub.lineno}:{sub.module if isinstance(sub, ast.ImportFrom) else ','.join(a.name for a in sub.names)}"
        for sub in ast.walk(node)
        if sub is not node and isinstance(sub, ast.Import | ast.ImportFrom)
    ]
    assert not inner, (
        f"{path.relative_to(PROJECT_ROOT)}::{node.name} imports inside the function body ({'; '.join(inner)}); "
        f"hoist every import to module level (fix the circular import structurally if that's why)"
    )


@pytest.mark.parametrize("path", _ENTRY_SCRIPTS, ids=_ENTRY_IDS)
def test_entry_script_defines_only_main(path: pathlib.Path) -> None:
    """EVERY .py outside the package + tests/ (entry scripts, CLIs, any new one) defines ONLY ``main``.

    They are thin shells: imports + one main() that wires bike_router.core (the shared logic). Any
    other top-level function or class means real logic leaked outside the package — where the strict
    per-symbol test/docstring/no-default gates don't reach — so move it into bike_router.core.
    """
    defined = [n.name for n in ast.parse(path.read_text()).body if isinstance(n, _DEF)]
    assert defined == ["main"], (
        f"{path.relative_to(PROJECT_ROOT)} defines {defined}; a non-package script may define ONLY 'main' — "
        f"move every other function/class into bike_router.core and import it"
    )


def _comment_block_runs(path: pathlib.Path) -> list[tuple[int, int]]:
    """(start_line, run_length) for each block of CONSECUTIVE standalone ``#`` comment lines.

    Only whole-line comments count (a trailing ``x = 1  # note`` is not a comment line); a blank or
    code line breaks the run. Used to cap comment blocks at 3 lines, like docstrings.
    """
    runs: list[tuple[int, int]] = []
    start = length = 0
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        if raw.lstrip().startswith("#"):
            start, length = (lineno, 1) if length == 0 else (start, length + 1)
        elif length:
            runs.append((start, length))
            length = 0
    if length:
        runs.append((start, length))
    return runs


@pytest.mark.parametrize(
    ("layer", "path"), _production_modules(), ids=[f"{lay}/{p.stem}" for lay, p in _production_modules()]
)
def test_comment_blocks_are_at_most_3_lines(layer: str, path: pathlib.Path) -> None:
    """No run of consecutive ``#`` comment lines exceeds 3 — comments stay terse, like docstrings."""
    too_long = [(start, length) for start, length in _comment_block_runs(path=path) if length > 3]
    assert not too_long, (
        f"{layer}/{path.name} has comment block(s) over 3 lines at "
        f"{', '.join(f'line {s} ({n} lines)' for s, n in too_long)} — tighten to ≤3"
    )


# Shipped production code the duplicate-string rule scans: the package + the root entry mains.
_DUP_SCAN_PY = sorted(_PKG.rglob("*.py")) + _ENTRY_SCRIPTS
# A numeric-format mini-language spec (``.2f``, ``,.0f``, ``d``): alignment/sign/width/precision
# glyphs then one optional presentation type. Exempt — it holds no domain concept and a real value
# like ``rail`` can't be spelled this way (its letters aren't all format-type chars).
_FORMAT_SPEC = re.compile(r"[<>=^+\- #0-9.,_]*[bcdeEfFgGnosxX%]?\Z")
# Same spelling, DIFFERENT domain in each site — the AST can't tell them apart, so one shared constant
# would wrongly couple unrelated things (or is a language idiom). NOT drift-prone; intentionally exempt.
#   __main__: the `if __name__ ==` idiom.  lat/lon/name/bbox: external API param AND DataFrame column.
#   color/width/path: deck.gl prop-dict keys.  origin/destination: CLI arg names AND gmaps params.
# Same-spelling-different-domain identifiers, each justified (≤5 words) — NOT a shared-constant miss.
_COINCIDENTAL_REASONS = {
    "__main__": "Python entry-point guard",
    "lat": "latitude arg name everywhere",
    "lon": "longitude arg name everywhere",
    "name": "generic name field/column/datum",
    "bbox": "coverage-box arg name everywhere",
    "color": "deck datum + plot key",
    "width": "deck datum + ribbon key",
    "path": "filesystem + deck path key",
    "origin": "endpoint arg + Maps param",
    "destination": "endpoint arg + Maps param",
    "y": "OSMnx lat attr + axis",
    "position": "deck.gl marker datum key",
    "geometry": "GeoJSON field vs OSMnx attr",
}
_COINCIDENTAL = frozenset(_COINCIDENTAL_REASONS)


def _is_domain_string(value: str) -> bool:
    """True if ``value`` must be a shared constant, not glue / a coincidental cross-domain identifier.

    Flags a letter-bearing value EXCEPT: format specs (``.2f``), whitespace-edged display/log fragments
    (`` m``, ``Wrote ``), and _COINCIDENTAL same-spelling-different-domain identifiers.
    """
    if value != value.strip() or value in _COINCIDENTAL:
        return False
    return bool(re.search(r"[A-Za-z]", value)) and not _FORMAT_SPEC.fullmatch(value)


def _string_literals(path: pathlib.Path) -> list[str]:
    """Free-standing str/bytes literal VALUES: NOT docstrings, subscript keys, or annotations.

    Excludes subscript keys (``df["osmid"]`` — schema access) and annotation strings (``"np.ndarray"``
    forward refs). Bytes are decoded so ``b"rail"`` can't dodge the str scan; f-string literal parts +
    implicit-concat are covered because ast.walk sees every Constant (merged or nested).
    """
    tree = ast.parse(path.read_text())
    excluded: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Module, ast.ClassDef, *_FUNC_TYPES))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            excluded.add(id(node.body[0].value))  # docstring
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            excluded.add(id(node.slice))  # df["col"] / d["key"] — structural access
        anns = [*([node.returns] if isinstance(node, _FUNC_TYPES) else [])]
        anns += [node.annotation] if isinstance(node, ast.AnnAssign | ast.arg) and node.annotation else []
        for ann in anns:
            excluded.update(id(sub) for sub in ast.walk(ann) if isinstance(sub, ast.Constant))
    values: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in excluded:
            continue
        if isinstance(node.value, str):
            values.append(node.value)
        elif isinstance(node.value, bytes):  # b"rail" must not dodge the str scan
            values.append(node.value.decode("utf-8", "replace"))
    return values


def test_no_duplicate_string_literals_across_production_code() -> None:
    """A domain string literal repeated across shipped code (package + entry mains) must be a constant.

    Flags any duplicated letter-bearing value except numeric-format specs (``.2f``); no length/word
    threshold to game. Evasions closed: bytes decoded, f-string parts + implicit-concat walked, keys/
    annotations/docstrings excluded. Single-char keys like ``y`` are caught.
    """
    where: dict[str, set[str]] = {}
    for path in _DUP_SCAN_PY:
        for value in _string_literals(path=path):
            if _is_domain_string(value=value):
                where.setdefault(value, set()).add(str(path.relative_to(PROJECT_ROOT)))
    dupes = {value: files for value, files in where.items() if len(files) > 1}
    assert not dupes, (
        "domain string literals repeated across files — extract each into ONE shared constant:\n"
        + "\n".join(f"  {value!r} in {sorted(files)}" for value, files in sorted(dupes.items()))
    )


@pytest.mark.parametrize("path", _PRODUCTION_PY, ids=_PRODUCTION_IDS)
def test_no_private_name_imported_across_modules(path: pathlib.Path) -> None:
    """Production code never imports a ``_``-prefixed name from another module (tests may).

    A leading underscore means module-private; importing it elsewhere breaks that boundary — the
    symbol is either truly private (used in-file) or a real API and should be renamed public. Dunders
    (``__future__`` etc.) are exempt. tests/ + scripts/ are excluded from _PRODUCTION_PY.
    """
    offenders = [
        f"{node.module}.{alias.name}"
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name.startswith("_") and not alias.name.startswith("__")
    ]
    assert not offenders, (
        f"{path.relative_to(PROJECT_ROOT)} imports module-private name(s) {offenders} — a leading-_ "
        f"symbol must stay in its module (use it in-file) or be renamed public if it's really shared"
    )
