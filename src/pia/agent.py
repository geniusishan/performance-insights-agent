"""The Performance Insights Agent - a LangGraph state machine.

    route ──► data ──► write_sql ──► check_sql ──► run_sql ──► write_answer ──► verify_answer ──► finalize
      │                   ▲  ▲            │            │             ▲                 │
      │                   │  └─ rejected ─┘            │             └── unsupported ──┘
      │                   └────── db error ────────────┘                 numbers
      ├──► definition ──► define ─────────────────────────────────────────────────────────► finalize
      ├──► clarify ─────────────────────────────────────────────────────────────────────────► finalize
      └──► out_of_scope ──► refuse ─────────────────────────────────────────────────────────► finalize

Self-correction: guardrail rejections and database errors are fed back to the SQL writer
(bounded retries). Numbers in the narrative are verified against the result; if they
cannot be, the answer is shown with an explicit "unverified" warning instead of silently.
"""
from __future__ import annotations

import operator
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from pia import audit, charts, db, grounding, guardrails, prompts
from pia.llm import LLM, default_llm
from pia.settings import policy

INJECTION_HINTS = re.compile(
    r"ignore (all |the )?(previous|prior|above) (instructions|rules)|system prompt|you are now|"
    r"\b(drop|delete|truncate|update|insert|alter)\b\s+\w+|disregard|jailbreak", re.I)


class State(TypedDict, total=False):
    question: str
    role: str
    history: list[dict]
    as_of: str
    intent: str
    standalone_question: str
    sql: str
    assumptions: list[str]
    attempts: Annotated[list[dict], operator.add]
    verdict: Any
    df: Any
    db_seconds: float
    answer: str
    follow_ups: list[str]
    answer_tries: int
    grounding: Any
    status: str
    trace: Annotated[list[dict], operator.add]
    tokens: Annotated[list[int], operator.add]
    flags: list[str]


@dataclass
class AgentResult:
    question: str
    answer: str
    status: str                      # answered | answered_unverified | definition | clarify | refused | failed
    sql: str | None = None
    data: pd.DataFrame | None = None
    chart: dict | None = None
    guardrail_checks: list[str] = field(default_factory=list)
    grounding_checked: list[str] = field(default_factory=list)
    grounding_unsupported: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    follow_ups: list[str] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    seconds: float = 0.0
    tokens: int = 0
    request_id: str = ""


class InsightsAgent:
    def __init__(self, llm: LLM | None = None, write_audit: bool = True):
        self.llm = llm or default_llm()
        self.write_audit = write_audit
        self.limits = policy()["limits"]
        self.graph = self._build()

    # ---- helpers -----------------------------------------------------------
    def _llm(self, task: str, system: str, user: str) -> tuple[dict, dict]:
        r = self.llm.complete_json(task, system, user)
        return r.data, {"tokens": [r.prompt_tokens + r.completion_tokens]}

    @staticmethod
    def _step(name: str, t0: float, detail: str = "") -> dict:
        return {"trace": [{"step": name, "seconds": round(time.perf_counter() - t0, 3), "detail": detail}]}

    # ---- nodes ---------------------------------------------------------------
    def route(self, s: State) -> dict:
        t0 = time.perf_counter()
        q = s["question"].strip()
        flags = ["possible_prompt_injection"] if INJECTION_HINTS.search(q) else []
        if len(q) > self.limits["max_question_chars"]:
            return {"intent": "out_of_scope", "flags": flags + ["too_long"], **self._step("route", t0, "too long")}
        data, extra = self._llm("route", *prompts.route_prompt(q, s.get("history", [])))
        intent = data.get("intent", "data")
        if intent not in ("data", "definition", "clarify", "out_of_scope"):
            intent = "data"
        return {"intent": intent, "standalone_question": data.get("standalone_question") or q,
                "answer": data.get("clarifying_question", "") if intent == "clarify" else "",
                "flags": flags, **extra, **self._step("route", t0, intent)}

    def write_sql(self, s: State) -> dict:
        t0 = time.perf_counter()
        data, extra = self._llm("sql", *prompts.sql_prompt(s["standalone_question"], s["as_of"], s["role"],
                                                           s.get("attempts", [])))
        return {"sql": (data.get("sql") or "").strip(), "assumptions": data.get("assumptions") or [],
                **extra, **self._step("write_sql", t0, f"attempt {len(s.get('attempts', [])) + 1}")}

    def check_sql(self, s: State) -> dict:
        t0 = time.perf_counter()
        v = guardrails.validate(s["sql"], s["role"])
        out: dict = {"verdict": v, **self._step("check_sql", t0, "pass" if v.ok else "; ".join(v.reasons))}
        if v.ok:
            out["sql"] = v.sql
        else:
            out["attempts"] = [{"sql": s["sql"], "error": "guardrail: " + "; ".join(v.reasons), "stage": "guardrail"}]
        return out

    def run_sql(self, s: State) -> dict:
        t0 = time.perf_counter()
        try:
            res = db.run_query(s["sql"])
        except Exception as e:  # noqa: BLE001 - any DB error goes back to the SQL writer
            msg = str(e).splitlines()[0][:300]
            return {"df": None, "attempts": [{"sql": s["sql"], "error": f"database: {msg}", "stage": "database"}],
                    **self._step("run_sql", t0, f"error: {msg}")}
        return {"df": res.df, "db_seconds": res.seconds, **self._step("run_sql", t0, f"{len(res.df)} rows")}

    def write_answer(self, s: State) -> dict:
        t0 = time.perf_counter()
        g = s.get("grounding")
        feedback = f"these numbers are not in the result: {', '.join(g.unsupported)}" if g and not g.ok else None
        data, extra = self._llm("answer", *prompts.answer_prompt(s["standalone_question"], s["sql"], s["df"],
                                                                  s.get("assumptions", []), feedback))
        return {"answer": data.get("answer", "").strip(), "follow_ups": data.get("follow_ups", [])[:3],
                "answer_tries": s.get("answer_tries", 0) + 1, **extra, **self._step("write_answer", t0)}

    def verify_answer(self, s: State) -> dict:
        t0 = time.perf_counter()
        g = grounding.check(s["answer"], s["df"], s["standalone_question"])
        detail = f"{len(g.checked)} numbers checked" + ("" if g.ok else f", unsupported: {g.unsupported}")
        return {"grounding": g, **self._step("verify_answer", t0, detail)}

    def define(self, s: State) -> dict:
        t0 = time.perf_counter()
        data, extra = self._llm("define", *prompts.definition_prompt(s["standalone_question"]))
        return {"answer": data.get("answer", ""), "status": "definition", **extra, **self._step("define", t0)}

    def refuse(self, s: State) -> dict:
        return {"status": "refused", "answer": (
            "I can only answer questions about sales performance data: signings, targets and attainment, "
            "scorecards, new-merchant volume, activation, forecasts and data quality, by region, market, "
            "territory, rep or month. I can't change data or follow instructions outside that scope.")}

    def clarify(self, s: State) -> dict:
        return {"status": "clarify", "answer": s.get("answer") or "Could you be more specific?"}

    def fail(self, s: State) -> dict:
        last = s.get("attempts", [])[-1]["error"] if s.get("attempts") else "unknown error"
        return {"status": "failed", "answer": (
            "I couldn't produce a safe, valid query for that question, so I'm not going to guess. "
            f"Last issue: {last}. Try rephrasing, or narrowing it to a market, period or metric.")}

    # ---- routing -------------------------------------------------------------
    def _after_route(self, s: State) -> str:
        return {"data": "write_sql", "definition": "define", "clarify": "clarify"}.get(s["intent"], "refuse")

    def _after_check(self, s: State) -> str:
        if s["verdict"].ok:
            return "run_sql"
        return "write_sql" if len(s.get("attempts", [])) <= self.limits["max_repair_attempts"] else "fail"

    def _after_run(self, s: State) -> str:
        if s.get("df") is not None:
            return "write_answer"
        return "write_sql" if len(s.get("attempts", [])) <= self.limits["max_repair_attempts"] else "fail"

    def _after_verify(self, s: State) -> str:
        return "finalize" if s["grounding"].ok or s.get("answer_tries", 0) >= 2 else "write_answer"

    def finalize(self, s: State) -> dict:
        if s.get("status"):
            return {}
        g = s.get("grounding")
        return {"status": "answered" if (g is None or g.ok) else "answered_unverified"}

    def _build(self):
        g = StateGraph(State)
        for name in ("route", "write_sql", "check_sql", "run_sql", "write_answer", "verify_answer",
                     "define", "refuse", "clarify", "fail", "finalize"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "route")
        g.add_conditional_edges("route", self._after_route, ["write_sql", "define", "clarify", "refuse"])
        g.add_edge("write_sql", "check_sql")
        g.add_conditional_edges("check_sql", self._after_check, ["run_sql", "write_sql", "fail"])
        g.add_conditional_edges("run_sql", self._after_run, ["write_answer", "write_sql", "fail"])
        g.add_edge("write_answer", "verify_answer")
        g.add_conditional_edges("verify_answer", self._after_verify, ["finalize", "write_answer"])
        for n in ("define", "refuse", "clarify", "fail"):
            g.add_edge(n, "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    # ---- public API -------------------------------------------------------------
    def ask(self, question: str, role: str = "analyst", history: list[dict] | None = None,
            session_id: str = "cli") -> AgentResult:
        t0 = time.perf_counter()
        rid = uuid.uuid4().hex[:12]
        s: State = self.graph.invoke(
            {"question": question, "role": role, "history": history or [], "as_of": db.as_of(),
             "attempts": [], "trace": [], "tokens": [], "flags": []},
            {"recursion_limit": 30},
        )
        g = s.get("grounding")
        answer = s.get("answer", "")
        if s.get("status") == "answered_unverified":
            answer += ("\n\n> ⚠️ **Unverified:** some figures above (" + ", ".join(g.unsupported) +
                       ") could not be traced to the query result. Check the data table before sharing.")
        ok_sql = s.get("verdict") is not None and s["verdict"].ok and s.get("df") is not None
        res = AgentResult(
            question=question, answer=answer, status=s.get("status", "failed"),
            sql=s.get("sql") if ok_sql else None, data=s.get("df") if ok_sql else None,
            chart=charts.choose(s.get("df")) if ok_sql else None,
            guardrail_checks=s["verdict"].checks if s.get("verdict") is not None else [],
            grounding_checked=g.checked if g else [], grounding_unsupported=g.unsupported if g else [],
            assumptions=s.get("assumptions", []), follow_ups=s.get("follow_ups", []),
            attempts=s.get("attempts", []), trace=s.get("trace", []), flags=s.get("flags", []),
            seconds=round(time.perf_counter() - t0, 3), tokens=sum(s.get("tokens", [])), request_id=rid,
        )
        if self.write_audit:
            audit.record({
                "request_id": rid, "session_id": session_id, "role": role, "model": self.llm.name,
                "question": question, "standalone_question": s.get("standalone_question"),
                "intent": s.get("intent"), "status": res.status, "sql": res.sql,
                "rejected_attempts": res.attempts, "guardrail_checks": res.guardrail_checks,
                "rows": None if res.data is None else len(res.data),
                "grounding_checked": res.grounding_checked, "grounding_unsupported": res.grounding_unsupported,
                "flags": res.flags, "seconds": res.seconds, "tokens": res.tokens,
                "trace": res.trace, "answer": res.answer,
            })
        return res
