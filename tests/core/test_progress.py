"""Progress-reporter tests — no-op sink, RSS logger, tqdm-backed counter."""

import pytest

from bike_router.core.progress import log_rss, null_progress, tqdm_progress


def test_null_progress():
    # The default sink accepts (done, total) and returns nothing, without error.
    assert null_progress(done=3, total=10) is None


def test_log_rss():
    # Returns the process resident memory in GB as a positive float (used at pipeline seams).
    rss = log_rss(label="unit-test")
    assert isinstance(rss, float) and rss > 0.0


def test_tqdm_progress():
    # A real per-item counter: advances to total (closing the bar) and rejects done > total.
    report = tqdm_progress(desc="test")
    report(0, 100)
    report(50, 100)
    report(100, 100)  # reaching total closes the bar
    with pytest.raises(AssertionError):
        tqdm_progress(desc="test")(11, 10)  # done > total → fail loud
