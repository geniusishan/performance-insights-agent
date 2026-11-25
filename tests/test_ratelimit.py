from pathlib import Path

from streamlit.testing.v1 import AppTest

from pia import audit, ratelimit
from pia.ratelimit import GlobalLimiter, SessionUsage, check

APP = str(Path(__file__).resolve().parents[1] / "app" / "chat.py")


def test_per_session_limit_and_cooldown():
    lim = ratelimit.limits()
    s, g = SessionUsage(), GlobalLimiter(per_hour=10_000)
    t = 1_000_000.0
    assert check(s, g, t).allowed
    assert not check(s, g, t + 1).allowed                      # inside cooldown
    for i in range(lim["per_session"] - 1):
        t += lim["cooldown_seconds"] + 1
        assert check(s, g, t).allowed
    t += lim["cooldown_seconds"] + 1
    d = check(s, g, t)
    assert not d.allowed and "limit" in d.message


def test_global_cap_applies_across_sessions_and_slides():
    g = GlobalLimiter(per_hour=3)
    t = 0.0
    for _ in range(3):
        assert check(SessionUsage(), g, t).allowed             # fresh session each time = page refresh
    assert not check(SessionUsage(), g, t + 10).allowed        # refreshing doesn't bypass the cap
    assert check(SessionUsage(), g, t + 3601).allowed          # window slides after an hour


def test_audit_tab_shows_only_own_session(monkeypatch):
    audit.record({"request_id": "x1", "session_id": "someone-else", "question": "SECRET OTHER VISITOR QUESTION",
                  "status": "answered", "seconds": 1.0})
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    rendered = " ".join(str(getattr(e, "value", "")) for e in at.main)
    for df in at.dataframe:
        rendered += df.value.to_string()
    assert "SECRET OTHER VISITOR QUESTION" not in rendered


def test_limit_blocks_question_in_app(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "dummy")
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["session_id"] = "t2"
    at.session_state["messages"] = []
    at.session_state["pending"] = "Which market leads?"
    at.session_state["usage"] = SessionUsage(count=ratelimit.limits()["per_session"])
    at.run()
    assert not at.exception
    assert any("demo limit" in w.value for w in at.warning)
    assert at.session_state["messages"] == []                  # nothing sent to the LLM
