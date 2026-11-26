"""Performance Insights Agent - chat UI.

    streamlit run app/chat.py
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pia import audit, brief, db, ratelimit  # noqa: E402
from pia.settings import ROOT, policy  # noqa: E402

st.set_page_config(page_title="Performance Insights Agent", page_icon="🤖", layout="wide")

EXAMPLES = [
    "Why is Japan behind target this year?",
    "Which markets are behind target this year, and what's their biggest gap?",
    "Show monthly signings for India over the last 12 months",
    "Which 5 territories have the lowest weighted score?",
    "How does 90-day activation compare across regions?",
    "What's the signings forecast for Q1 2027 by market?",
]

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
    st.session_state.messages = []
    st.session_state.pending = None
if "usage" not in st.session_state:
    st.session_state.usage = ratelimit.SessionUsage()


@st.cache_resource(show_spinner=False)
def global_limiter() -> ratelimit.GlobalLimiter:
    """One limiter shared by every visitor of this app instance."""
    return ratelimit.GlobalLimiter(per_hour=ratelimit.limits()["global_per_hour"])


def allow_llm_call() -> bool:
    d = ratelimit.check(st.session_state.usage, global_limiter())
    if not d.allowed:
        st.warning(d.message)
    return d.allowed


@st.cache_resource(show_spinner=False)
def get_agent():
    from pia.agent import InsightsAgent
    return InsightsAgent()


@st.cache_resource(show_spinner="First launch: building the demo warehouse (about 10 seconds)…")
def ensure_warehouse() -> bool:
    """On a fresh deploy, build the Field Performance Hub warehouse if it isn't there."""
    from pia.settings import warehouse_path
    if not warehouse_path().exists():
        try:
            from fph import pipeline
        except ImportError:
            return False
        os.environ.setdefault("FPH_WAREHOUSE", str(warehouse_path()))
        warehouse_path().parent.mkdir(parents=True, exist_ok=True)
        pipeline.run(verbose=False)
    return True


ensure_warehouse()

# ---- sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("🤖 Insights Agent")
    try:
        as_of = db.as_of()
        st.success(f"Warehouse connected · data as of {pd.Timestamp(as_of):%b %Y}")
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        st.stop()
    key_ok = bool(os.environ.get("LLM_API_KEY")) or os.environ.get("LLM_PROVIDER") == "ollama"
    provider = os.environ.get("LLM_PROVIDER", "gemini")
    st.caption(f"LLM: **{provider}** {'✅' if key_ok else '⚠️ no API key (set LLM_API_KEY in .env or app secrets)'}")
    lim = ratelimit.limits()
    st.caption(f"Questions left this visit: {max(0, lim['per_session'] - st.session_state.usage.count)}"
               f" / {lim['per_session']}")
    role = st.selectbox("Your role", list(policy()["roles"]), help="Viewer cannot see rep names (restricted column).")
    st.markdown("**Try asking**")
    for ex in EXAMPLES:
        if st.button(ex, width="stretch", disabled=not key_ok):
            st.session_state.pending = ex
    if st.button("🧹 New conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Every query is checked by code-enforced guardrails, run read-only, "
               "and logged. Numbers in answers are verified against the data.")


def render_chart(df: pd.DataFrame, spec: dict) -> None:
    fn = px.line if spec["type"] == "line" else px.bar
    kw = {"x": spec["x"], "y": spec["y"]}
    if spec.get("color"):
        kw["color"] = spec["color"]
    d = df.sort_values(spec["y"], ascending=False) if spec["type"] == "bar" else df.sort_values(spec["x"])
    fig = fn(d, **kw)
    if spec.get("percent"):
        fig.update_layout(yaxis_tickformat=".0%")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), xaxis_title=None)
    st.plotly_chart(fig, width="stretch")


def render_result(m: dict, idx: int) -> None:
    st.markdown(m["answer"])
    df = pd.DataFrame(m["data"]) if m.get("data") is not None else None
    if df is not None and m.get("chart"):
        render_chart(df, m["chart"])
    if m.get("sql"):
        badge = "✅ numbers verified" if not m["unsupported"] else "⚠️ unverified numbers"
        with st.expander(f"🔎 How I got this · {badge} · {m['seconds']:.1f}s"):
            st.code(m["sql"], language="sql")
            st.markdown("**Guardrails:** " + " · ".join(f"✅ {c}" for c in m["checks"]))
            if m["attempts"]:
                st.markdown(f"**Self-corrected {len(m['attempts'])}×:** " +
                            "; ".join(a["error"] for a in m["attempts"]))
            if m["checked"]:
                st.markdown("**Verified figures:** " + ", ".join(m["checked"]))
            if m["assumptions"]:
                st.markdown("**Assumptions:** " + "; ".join(m["assumptions"]))
            st.dataframe(df, hide_index=True, width="stretch")
            st.download_button("Download CSV", df.to_csv(index=False), file_name=f"result_{idx}.csv",
                               key=f"dl{idx}")
            st.caption("Trace: " + " → ".join(f"{t['step']} ({t['seconds']}s)" for t in m["trace"]))
    for j, f in enumerate(m.get("follow_ups", [])):
        if st.button(f"↪ {f}", key=f"fu{idx}_{j}"):
            st.session_state.pending = f
            st.rerun()


tab_chat, tab_brief, tab_audit, tab_evals = st.tabs(["💬 Ask", "📰 Weekly brief", "🧾 Audit log", "🧪 Evals"])

with tab_chat:
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                render_result(m, i)
            else:
                st.markdown(m["content"])
    typed = st.chat_input("Ask about signings, targets, scorecards, forecasts…", disabled=not key_ok)
    question = typed or st.session_state.pending
    if question and not allow_llm_call():
        st.session_state.pending = None
        question = None
    if question:
        st.session_state.pending = None
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        history = [{"question": a["content"], "answer": b.get("answer", "")}
                   for a, b in zip(st.session_state.messages[::2], st.session_state.messages[1::2])]
        with st.chat_message("assistant"):
            with st.spinner("Thinking: routing → SQL → guardrails → query → verify…"):
                r = get_agent().ask(question, role=role, history=history, session_id=st.session_state.session_id)
            msg = {"role": "assistant", "answer": r.answer, "sql": r.sql,
                   "data": None if r.data is None else r.data.to_dict("list"), "chart": r.chart,
                   "checks": r.guardrail_checks, "attempts": r.attempts, "checked": r.grounding_checked,
                   "unsupported": r.grounding_unsupported, "assumptions": r.assumptions,
                   "follow_ups": r.follow_ups, "trace": r.trace, "seconds": r.seconds}
            render_result(msg, len(st.session_state.messages))
        st.session_state.messages.append(msg)

with tab_brief:
    st.markdown("Facts are computed with fixed SQL; the LLM only writes prose; every number is verified. "
                "If verification fails, a deterministic template is published instead.")
    c1, c2 = st.columns(2)
    if c1.button("Generate with LLM", disabled=not key_ok, type="primary") and allow_llm_call():
        st.session_state.brief = brief.generate()
    if c2.button("Generate template version"):
        f = brief.facts()
        st.session_state.brief = {"brief": brief.template_brief(f), "source": "template", "verified_numbers": [], "facts": f}
    if "brief" in st.session_state:
        b = st.session_state.brief
        st.markdown(b["brief"])
        st.caption(f"Source: {b['source']}" + (f" · verified: {', '.join(b['verified_numbers'])}" if b["verified_numbers"] else ""))
        st.download_button("Download .md", b["brief"], file_name="weekly_brief.md")
        with st.expander("FACTS pack"):
            st.json(b["facts"])

with tab_audit:
    st.caption("For privacy, you see only the questions from your own visit. Every visitor's questions are "
               "still logged on the server for governance.")
    log = audit.load(5000)
    if not log.empty and "session_id" in log.columns:
        log = log[log["session_id"] == st.session_state.session_id]
    if log.empty:
        st.info("No questions logged yet in this visit.")
    else:
        c = st.columns(4)
        c[0].metric("Questions", len(log))
        c[1].metric("Answered & verified", f"{(log['status'] == 'answered').mean():.0%}")
        c[2].metric("Refused / failed", int(log["status"].isin(["refused", "failed"]).sum()))
        c[3].metric("Median latency", f"{log['seconds'].median():.1f}s")
        st.dataframe(log.reindex(columns=["ts", "role", "question", "status", "rows", "seconds", "tokens", "flags", "sql"]),
                     hide_index=True, width="stretch")

with tab_evals:
    rep = ROOT / "evals" / "report.md"
    if rep.exists():
        st.markdown(rep.read_text())
    else:
        st.info("Run `python -m pia.evals.run_evals` to generate the evaluation report.")
