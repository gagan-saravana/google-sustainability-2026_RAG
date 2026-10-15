"""Demo-limit tests: pure Python, injectable clock."""
from datetime import date

from src import limits
from src.limits import DemoLimiter


def _lim(per=3, cap=100, **kw):
    return DemoLimiter(per_browser=per, daily_cap=cap, **kw)


def _ask(lim, b):
    """One full successful question."""
    assert lim.reserve(b) is None
    lim.commit(b, "exchange")


def test_per_browser_limit_enforced_and_remaining_counts_down():
    lim = _lim()
    for expected_left in (2, 1, 0):
        _ask(lim, "a")
        assert lim.remaining("a") == expected_left
    assert lim.reserve("a") == limits.BROWSER_LIMIT


def test_browsers_are_independent():
    lim = _lim()
    for _ in range(3):
        _ask(lim, "a")
    assert lim.reserve("b") is None


def test_busy_blocks_double_submit_until_commit():
    lim = _lim()
    assert lim.reserve("a") is None
    assert lim.reserve("a") == limits.BUSY      # second tab / double click
    lim.commit("a", "x")
    assert lim.reserve("a") is None


def test_release_refunds_browser_and_daily_counters():
    lim = _lim(cap=1)
    assert lim.reserve("a") is None
    lim.release("a")
    assert lim.remaining("a") == 3 and lim.daily_remaining() == 1
    assert lim.session("a").history == []


def test_global_daily_cap_blocks_everyone_and_does_not_charge_the_browser():
    lim = _lim(cap=2)
    _ask(lim, "a")
    _ask(lim, "b")
    assert lim.reserve("c") == limits.DAILY_CAP
    assert lim.remaining("c") == 3


def test_daily_cap_resets_on_a_new_day():
    day = {"d": date(2026, 10, 13)}
    lim = _lim(cap=1, today=lambda: day["d"])
    _ask(lim, "a")
    assert lim.reserve("b") == limits.DAILY_CAP
    day["d"] = date(2026, 10, 14)
    assert lim.reserve("b") is None


def test_history_is_kept_in_order_per_browser():
    lim = _lim()
    for i in range(3):
        assert lim.reserve("a") is None
        lim.commit("a", f"ex{i}")
    assert lim.session("a").history == ["ex0", "ex1", "ex2"]


def test_session_table_is_bounded():
    lim = _lim(max_sessions=3)
    for i in range(10):
        lim.session(f"b{i}")
    assert len(lim._sessions) == 3
