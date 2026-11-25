"""Provider-agnostic LLM client.

Gemini, Groq, OpenAI and local Ollama all expose an OpenAI-compatible chat API, so one
client covers them all; switch with LLM_PROVIDER in .env. Free options: Gemini API
(Google AI Studio key) or Groq. Every call returns parsed JSON plus token usage and
latency for the audit log.

FakeLLM lets the whole agent graph be tested offline with scripted responses.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol

PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "openai": (None, "gpt-4o-mini"),
    "ollama": ("http://localhost:11434/v1", "qwen2.5-coder:7b"),
}


@dataclass
class LLMResponse:
    data: dict
    raw: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


class LLM(Protocol):
    name: str

    def complete_json(self, task: str, system: str, user: str) -> LLMResponse: ...


def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
    return json.loads(text)


class OpenAICompatibleLLM:
    def __init__(self, provider: str | None = None, model: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0):
        from openai import OpenAI

        self.provider = (provider or os.environ.get("LLM_PROVIDER", "gemini")).lower()
        if self.provider not in PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {sorted(PROVIDERS)}")
        base_url, default_model = PROVIDERS[self.provider]
        self.model = model or os.environ.get("LLM_MODEL") or default_model
        key = api_key or os.environ.get("LLM_API_KEY") or ("ollama" if self.provider == "ollama" else None)
        if not key:
            raise RuntimeError("LLM_API_KEY is not set. Copy .env.example to .env and add a free Gemini or Groq key.")
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=90, max_retries=3)
        self.temperature = temperature
        self.name = f"{self.provider}:{self.model}"

    def complete_json(self, task: str, system: str, user: str) -> LLMResponse:
        import openai

        t0 = time.perf_counter()
        kwargs = dict(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system + "\n\nRespond with a single JSON object only."},
                      {"role": "user", "content": user}],
        )
        try:
            resp = self.client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
        except (openai.InternalServerError, openai.BadRequestError):
            # Some providers' OpenAI-compatible endpoints intermittently fail in strict JSON mode;
            # the prompt already demands JSON and _parse_json tolerates fences, so retry without it.
            resp = self.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or "{}"
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            data=_parse_json(text), raw=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            seconds=round(time.perf_counter() - t0, 3),
        )


class FakeLLM:
    """Scripted LLM for tests: `script[task]` is a dict, a list (consumed in order) or a callable(user)->dict."""

    def __init__(self, script: dict[str, dict | list | Callable[[str], dict]]):
        self.script = {k: (list(v) if isinstance(v, list) else v) for k, v in script.items()}
        self.calls: list[tuple[str, str]] = []
        self.name = "fake"

    def complete_json(self, task: str, system: str, user: str) -> LLMResponse:
        self.calls.append((task, user))
        s = self.script[task]
        data = s.pop(0) if isinstance(s, list) else (s(user) if callable(s) else s)
        return LLMResponse(data=data, raw=json.dumps(data))


def default_llm() -> LLM:
    return OpenAICompatibleLLM()
