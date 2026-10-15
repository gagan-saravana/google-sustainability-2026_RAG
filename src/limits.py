"""
limits.py: server-side demo limits. No accounts, no database, no files.

Why server-side? A counter kept only in the browser can be reset by anyone. Here
the browser merely carries an anonymous id (NiceGUI's signed session cookie); the
counters live in this process's memory and vanish on restart.

Two layers:
  * per browser: MAX_PROMPTS_PER_BROWSER questions (clear incognito -> new browser)
  * global daily cap: the layer that really protects the free Gemini quota,
    because incognito can always dodge the per-browser limit.

A failed model call is refunded (release), so users never lose a prompt to an API error.
Pure Python with an injectable clock, so it is unit-tested without NiceGUI.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable

# Reasons a question can be refused
BUSY = "busy"                    # this browser already has a question in flight
BROWSER_LIMIT = "browser_limit"  # this browser used all its questions
DAILY_CAP = "daily_cap"          # the whole demo is out of capacity for today


@dataclass
class Session:
    used: int = 0
    busy: bool = False
    history: list[Any] = field(default_factory=list)   # completed exchanges, oldest first


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


class DemoLimiter:
    def __init__(
        self,
        per_browser: int,
        daily_cap: int,
        max_sessions: int = 5000,
        today: Callable[[], date] = _utc_today,
    ) -> None:
        self.per_browser = per_browser
        self.daily_cap = daily_cap
        self.max_sessions = max_sessions
        self._today = today
        self._day = today()
        self._daily_used = 0
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    # -- bookkeeping ---------------------------------------------------------
    def _roll_day(self) -> None:
        if self._today() != self._day:
            self._day, self._daily_used = self._today(), 0

    def session(self, browser_id: str) -> Session:
        sess = self._sessions.get(browser_id)
        if sess is None:
            sess = self._sessions[browser_id] = Session()
            while len(self._sessions) > self.max_sessions:      # bound memory: forget the oldest
                self._sessions.popitem(last=False)
        else:
            self._sessions.move_to_end(browser_id)
        return sess

    def remaining(self, browser_id: str) -> int:
        return max(0, self.per_browser - self.session(browser_id).used)

    def daily_remaining(self) -> int:
        self._roll_day()
        return max(0, self.daily_cap - self._daily_used)

    # -- the question lifecycle: reserve -> (commit | release) -----------------------
    def reserve(self, browser_id: str) -> str | None:
        """Claim a slot BEFORE calling the model. Returns None if granted, else a reason code."""
        self._roll_day()
        sess = self.session(browser_id)
        if sess.busy:
            return BUSY
        if sess.used >= self.per_browser:
            return BROWSER_LIMIT
        if self._daily_used >= self.daily_cap:
            return DAILY_CAP
        sess.used += 1
        self._daily_used += 1
        sess.busy = True
        return None

    def commit(self, browser_id: str, exchange: Any) -> None:
        """The question was answered: keep it in this browser's history."""
        sess = self.session(browser_id)
        sess.history.append(exchange)
        sess.busy = False

    def release(self, browser_id: str) -> None:
        """The model call failed: refund the slot."""
        sess = self.session(browser_id)
        if sess.busy:
            sess.used = max(0, sess.used - 1)
            self._daily_used = max(0, self._daily_used - 1)
            sess.busy = False
