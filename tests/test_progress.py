"""Progress-reporter tests — no-op sink + tqdm-backed counter."""

import pytest

from bike_router.progress import null_progress, tqdm_progress


def test_null_progress_is_noop():
    # accepts (done, total) and returns nothing, without error
    assert null_progress(done=3, total=10) is None


def test_tqdm_progress_advances_and_closes():
    report = tqdm_progress(desc="test")
    report(0, 100)
    report(50, 100)
    report(100, 100)  # reaching total closes the bar (no error on further calls avoided)


def test_tqdm_progress_rejects_out_of_range():
    report = tqdm_progress(desc="test")
    with pytest.raises(AssertionError):
        report(11, 10)  # done > total
