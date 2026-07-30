"""progress tests — the no-op progress sink + RSS memory logger used at pipeline seams."""

from bike_router.core.progress import log_rss, null_progress


def test_null_progress():
    # The default sink accepts (done, total) and returns nothing, without error.
    assert null_progress(done=3, total=10) is None


def test_log_rss():
    # Returns the process resident memory in GB as a positive float (used at pipeline seams).
    rss = log_rss(label="unit-test")
    assert isinstance(rss, float) and rss > 0.0
