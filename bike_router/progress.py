"""Progress reporting — a real per-item counter, not fake stage fractions.

The only honest place to show "so much of so much" is a loop that actually
iterates a known number of items: the degree-2 node contraction. ``ProgressFn``
receives (items_done, items_total); the CLI wires a tqdm bar, Streamlit can wire
``st.progress``. Callers that don't care pass nothing (the default no-op).
"""

from collections.abc import Callable

# Real progress sink: (items processed so far, total items).
ProgressFn = Callable[[int, int], None]


def null_progress(done: int, total: int) -> None:
    """Default no-op sink."""


def tqdm_progress(desc: str) -> ProgressFn:
    """A ProgressFn backed by a tqdm bar; ``total`` is set on the first call.

    Reflects genuine per-item work (nodes contracted), so the bar advances only as
    real iterations complete. tqdm is imported lazily to keep it optional.
    """
    from tqdm import tqdm

    bar = tqdm(desc=desc, unit="node")

    def report(done: int, total: int) -> None:
        assert 0 <= done <= total, "progress: done must be within [0, total]"
        if bar.total != total:
            bar.reset(total=total)
        bar.n = done
        bar.refresh()
        if done >= total:
            bar.close()

    return report
