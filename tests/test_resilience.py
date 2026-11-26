"""The app must degrade gracefully when the LLM provider misbehaves - never crash on users."""
import httpx
import openai
import pytest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from pia import db
from pia.agent import InsightsAgent
from pia.llm import FakeLLM, OpenAICompatibleLLM

TOP_SQL = "SELECT market_name, score FROM scorecard.market ORDER BY score DESC LIMIT 3"


def _boom(_user):
    raise openai.InternalServerError("upstream 500", response=httpx.Response(500, request=httpx.Request("POST", "http://x")), body=None)


def test_answer_step_failure_still_returns_verified_data():
    fake = FakeLLM({"route": {"intent": "data", "standalone_question": "q"}, "sql": {"sql": TOP_SQL}, "answer": _boom})
    r = InsightsAgent(fake, write_audit=False).ask("q")
    assert r.status == "answered"
    assert r.data is not None and len(r.data) == 3
    assert "llm_error_on_answer" in r.flags
    assert "verified result" in r.answer


def test_route_failure_returns_friendly_message_not_exception():
    fake = FakeLLM({"route": _boom})
    r = InsightsAgent(fake).ask("q")
    assert r.status == "failed"
    assert "try again" in r.answer and "InternalServerError" in r.answer


def test_llm_client_retries_without_json_mode(monkeypatch):
    llm = OpenAICompatibleLLM(provider="gemini", api_key="test-key", model="m")
    calls = []

    class Msg:
        content = '```json\n{"ok": true}\n```'

    class Resp:
        choices = [type("C", (), {"message": Msg()})()]
        usage = None

    def create(**kw):
        calls.append("response_format" in kw)
        if "response_format" in kw:
            _boom("")
        return Resp()

    monkeypatch.setattr(llm.client.chat.completions, "create", create)
    out = llm.complete_json("answer", "sys", "user")
    assert out.data == {"ok": True}
    assert calls == [True, False]


def test_chat_history_renders_without_debug_dump(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "chat.py"), default_timeout=60)
    at.session_state["session_id"] = "t"
    at.session_state["pending"] = None
    at.session_state["messages"] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "answer": "Hi there.", "sql": None, "data": None, "chart": None, "checks": [],
         "attempts": [], "checked": [], "unsupported": [], "assumptions": [], "follow_ups": [], "trace": [], "seconds": 0.1},
    ]
    at.run()
    assert not at.exception
    text = " ".join(str(getattr(e, "value", "")) for e in at.main)
    assert "DeltaGenerator" not in text
    assert not at.get("help")  # Streamlit magic must not render st.help of a return value


def test_eval_preflight_stops_on_broken_llm():
    from pia.evals.run_evals import preflight
    fake = FakeLLM({"ping": _boom})
    with pytest.raises(SystemExit, match="preflight failed"):
        preflight(fake)


def test_llm_settings_tolerate_pasted_whitespace_and_quotes(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", '  "abc123"\n')
    monkeypatch.setenv("LLM_MODEL", " 'gemini-x-flash'\n")
    monkeypatch.setenv("LLM_PROVIDER", "Gemini ")
    llm = OpenAICompatibleLLM()
    assert llm.client.api_key == "abc123"
    assert llm.model == "gemini-x-flash" and llm.provider == "gemini"
