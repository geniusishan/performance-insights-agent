"""Usage limits for the public demo, so one visitor (or a script) can't exhaust the LLM quota.

Two layers:
  * per visitor session - a max number of questions per session, plus a short cooldown
  * whole app           - a sliding one-hour cap shared by every visitor; refreshing the page
                          to get a new session doesn't get around this one

Limits live in config/policy.yaml under `usage`. In-memory by design: resets when the app
restarts, which is the right trade-off for a demo. A production deployment would use a
shared store (e.g. Redis) and real user identity from SSO.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from pia.settings import policy


@dataclass
class Decision:
    allowed: bool
    message: str = ""


@dataclass
class SessionUsage:
    count: int = 0
    last: float = 0.0


@dataclass
class GlobalLimiter:
    per_hour: int
    _events: deque = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def try_acquire(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            while self._events and now - self._events[0] >= 3600:
                self._events.popleft()
            if len(self._events) >= self.per_hour:
                return False
            self._events.append(now)
            return True

    def remaining(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._lock:
            recent = sum(1 for t in self._events if now - t < 3600)
        return max(0, self.per_hour - recent)


def limits() -> dict:
    return {"per_session": 20, "cooldown_seconds": 4, "global_per_hour": 150} | (policy().get("usage") or {})


def check(session: SessionUsage, glob: GlobalLimiter, now: float | None = None) -> Decision:
    """Decide whether this visitor may spend one LLM request now. Records the use if allowed."""
    now = time.time() if now is None else now
    lim = limits()
    if session.count >= lim["per_session"]:
        return Decision(False, f"You've reached the demo limit of {lim['per_session']} questions per visit. "
                               "Thanks for trying it! The full code is on GitHub.")
    wait = lim["cooldown_seconds"] - (now - session.last)
    if session.last and wait > 0:
        return Decision(False, f"Please wait {wait:.0f} more second(s) before the next question.")
    if not glob.try_acquire(now):
        return Decision(False, "The demo is busy right now (hourly usage cap reached). Please try again later.")
    session.count += 1
    session.last = now
    return Decision(True)
