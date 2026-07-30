"""Progress reporting — a real per-item counter, not fake stage fractions.

``ProgressFn`` receives (items_done, items_total) from the ONE place progress is
shown: the one-time prebuilt-graph download (files fetched). The CLI wires a tqdm
bar, Streamlit an ``st.progress`` bar; callers that don't care pass the no-op.
``log_rss`` logs process memory at key seams (deploy RAM ceiling ≈ 2.7 GB).
"""

import logging
import os
from collections.abc import Callable

import psutil

logger = logging.getLogger(__name__)

# Real progress sink: (items processed so far, total items).
ProgressFn = Callable[[int, int], None]


def null_progress(done: int, total: int) -> None:
    """Default no-op sink."""


def log_rss(*, label: str) -> float:
    """Log process resident memory (GB) at ``label`` and return it — deploy ceiling ≈ 2.7 GB."""
    rss_gb: float = psutil.Process(os.getpid()).memory_info().rss / 1e9
    logger.info(f"RSS {rss_gb:.2f} GB — {label}")
    return rss_gb


def tqdm_progress(desc: str) -> ProgressFn:
    """A ProgressFn backed by a tqdm bar; ``total`` is set on the first call.

    Reflects genuine per-item work (files downloaded), so the bar advances only as
    real items complete. tqdm is imported lazily to keep it optional.
    """
    from tqdm import tqdm

    bar = tqdm(desc=desc, unit="file")

    def report(done: int, total: int) -> None:
        assert 0 <= done <= total, "progress: done must be within [0, total]"
        if bar.total != total:
            bar.reset(total=total)
        bar.n = done
        bar.refresh()
        if done >= total:
            bar.close()

    return report
