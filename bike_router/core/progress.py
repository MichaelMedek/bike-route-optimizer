"""Progress + process-memory helpers for the pipeline.

``ProgressFn`` receives (items_done, items_total) from the ONE place progress is shown: the
one-time prebuilt-graph download (files landed on disk). ``log_rss`` logs process memory at
key seams (deploy RAM ceiling ≈ 2.7 GB).
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
