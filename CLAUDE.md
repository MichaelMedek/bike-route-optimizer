# CLAUDE.md

Guidance for AI agents working in this repo. These are **non-negotiable** engineering principles — check every change against them. They exist because violations recur; treat them as review gates, not suggestions.

## Core principles

1. **Fail fast, no defensive fallbacks.** For internal invariants, use strict access (`d[key]`, direct attribute) and let it raise. Do NOT add `.get()`/`if x is None`/`try-except` that swallows or silently continues — that masks bugs the crash would have surfaced. Fallbacks are legitimate ONLY for genuine external/untrusted input (file/network data, env vars, user input). If you catch yourself writing a guard "just in case" around your own code's data, delete it and let it fail loud.

2. **Single source of truth — no drift.** Never maintain two pieces of state that must agree, or a hand-maintained list/set that must track an enum. Derive from the data, collapse to one field, or assert the relationship at import time.

3. **Fix the root cause, not the symptom.** When the same class of failure appears at one place, or even worse at multiple sites, stop patching each read/call site IMMEDIATELY! Find where the bad state is *produced* and fix it there. One correct emitter beats EVERY defensive readers. If needed refactor ALL callers across teh FULL codebase! "Too much work" is NEVER an excuse!

4. **No duplicated logic — extract and share.** If the same block appears in 2+ places, factor it into one function and call it. Duplicated logic drifts and gets fixed in only some copies. You are respnsible for all drifts, across the full code base, even if pre-existing. Dead code is STRICTLY forbidden and must be radically PRUNED out of all places across the code base!

5. **Prefer the existing pattern.** Before adding a new approach, find how the codebase already solves the same problem and follow it. Consistency over novelty.

## Style

- Terse, high-signal docstrings and comments — explain the *why*, not the obvious *what*. No restating the code in prose. MAX 2 or 3 liens of comments and docstring text blocks STRITCLY, but still Args in Google style.
- Explicit `if/elif/else ERROR` blocks over nested/clever ternaries when there's real branching logic.
- Match the surrounding code's naming, structure, and comment density.
- EVERY custom function must be called with complete explicit arguments by all callers, even if only ONE argeumt! `y=f(x)` MUST be strictly replaced by `y=f(x=x)`!

## Workflow

- Tests, `ruff`, and `mypy` must stay green; coverage gate must be met. Run the suite before declaring done.
- When you change behaviour, update the tests that encoded the old behaviour AND add a regression test for the specific bug — don't just make existing tests pass.
- After any complet batch of changes in code `pre-commit run --all-files` must be run, to let ruff reformat the code. No stupid manual import sortings and other bullshit needed.
- The meta tests `tests/test_import_boundaries.py` and `tests/test_unit_test_coverage.py` are ultra strict code and test quality guards and any code must strictly sattisfy them.
