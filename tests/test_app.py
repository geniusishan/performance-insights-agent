from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "chat.py")


def test_chat_app_renders_without_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 4
    next(b for b in at.button if b.label == "Generate template version").click().run()
    assert not at.exception
    assert any("Where we are" in m.value for m in at.markdown)
