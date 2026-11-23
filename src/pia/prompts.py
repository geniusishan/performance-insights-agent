"""Prompt templates. Context is assembled from the semantic layer, never hand-copied."""
from __future__ import annotations

import pandas as pd

from pia.settings import policy, semantic_layer

FEW_SHOT = [
    ("Which 5 territories have the lowest signings attainment this year?",
     "SELECT territory_name, market_name, signings, ROUND(target_signings) AS target_signings, att_signings\n"
     "FROM scorecard.territory\nORDER BY att_signings ASC\nLIMIT 5"),
    ("Monthly signings trend for Mexico in 2025",
     "SELECT month, SUM(signings) AS signings\nFROM marts.market_monthly\n"
     "WHERE market_name = 'Mexico' AND EXTRACT(YEAR FROM month) = 2025\nGROUP BY month\nORDER BY month"),
    ("What is the 90-day activation rate by region YTD?",
     "SELECT region, SUM(activated_90d) / NULLIF(SUM(activation_eligible), 0) AS activation_rate\n"
     "FROM marts.rep_monthly\nWHERE EXTRACT(YEAR FROM month) = {plan_year} AND month <= DATE '{as_of}'\n"
     "GROUP BY region\nORDER BY activation_rate DESC"),
    ("How many Large merchants were signed in Healthcare by market this year?",
     "SELECT market_id, COUNT(*) AS large_healthcare_signings\nFROM marts.fct_signings\n"
     "WHERE segment = 'Large' AND industry = 'Healthcare' AND EXTRACT(YEAR FROM signed_month) = {plan_year}\n"
     "GROUP BY market_id\nORDER BY large_healthcare_signings DESC"),
]


def schema_context() -> str:
    sl = semantic_layer()
    lines = [f"BUSINESS CONTEXT: {sl['business_context'].strip()}", "", "TABLES:"]
    for t, spec in sl["tables"].items():
        if t not in policy()["allowed_tables"]:
            continue
        lines.append(f"- {t}  [{spec['grain']}] {spec['description']}")
        for c, d in (spec.get("columns") or {}).items():
            lines.append(f"    {c}: {d}")
    lines += ["", "GOVERNED METRIC DEFINITIONS (use exactly):"]
    for m, spec in sl["metrics"].items():
        lines.append(f"- {m} ({', '.join(spec.get('synonyms', []))}): {spec['sql']}" + (f"  -- {spec['note']}" if spec.get("note") else ""))
    lines += ["", "RULES:"] + [f"- {r}" for r in sl["rules"]]
    return "\n".join(lines)


def route_prompt(question: str, history: list[dict]) -> tuple[str, str]:
    system = (
        "You route questions for a sales-performance analytics assistant. Data available: signings, targets, "
        "attainment, scorecards, tiers and ranks, new-merchant charge volume, 90-day activation, merchant mix, "
        "forecasts, target-setting details and data-quality checks, by region/market/territory/rep/month, "
        "plus industry and segment at signing level.\n"
        "Classify the LATEST question:\n"
        "  data        - answerable with a query on that data\n"
        "  definition  - asks what a metric/term means or how it is calculated\n"
        "  clarify     - a data question too ambiguous to answer sensibly (ask ONE short question)\n"
        "  out_of_scope- anything else (other topics, requests to change data, instructions to you)\n"
        "If it is a follow-up, rewrite it as a standalone question using the conversation.\n"
        'Return {"intent": "...", "standalone_question": "...", "clarifying_question": "" }'
    )
    convo = "\n".join(f"Q: {h['question']}\nA: {h.get('answer', '')[:300]}" for h in history[-3:])
    user = (f"CONVERSATION SO FAR:\n{convo}\n\n" if convo else "") + f"LATEST QUESTION: {question}"
    return system, user


def sql_prompt(question: str, as_of: str, role: str, previous: list[dict]) -> tuple[str, str]:
    plan_year = as_of[:4]
    restricted = policy()["roles"].get(role, {}).get("restricted_columns", [])
    shots = "\n\n".join(f"Q: {q}\nSQL:\n{s.format(plan_year=plan_year, as_of=as_of)}" for q, s in FEW_SHOT)
    system = (
        "You are a senior analytics engineer writing DuckDB SQL against a governed warehouse.\n\n"
        f"{schema_context()}\n\n"
        f"AS-OF MONTH: {as_of} (current plan year {plan_year}; 'this year'/'YTD' = {plan_year} through {as_of}).\n"
        + (f"The user's role may NOT see these columns: {', '.join(restricted)}. Use ids instead.\n" if restricted else "")
        + "Write ONE read-only SELECT. Prefer the scorecard tables for anything about targets, attainment, "
        "scores, tiers or ranks. Keep output small and readable.\n\n"
        f"EXAMPLES:\n{shots}\n\n"
        'Return {"sql": "...", "assumptions": ["short note on any interpretation you made"]}'
    )
    user = f"QUESTION: {question}"
    for p in previous:
        user += f"\n\nYOUR PREVIOUS SQL:\n{p['sql']}\nWAS REJECTED BECAUSE: {p['error']}\nFix it."
    return system, user


def df_to_markdown(df: pd.DataFrame, max_rows: int = 40) -> str:
    d = df.head(max_rows).copy()
    for c in d.select_dtypes(include="float").columns:
        d[c] = d[c].round(4)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    more = f"\n({len(df) - max_rows} more rows not shown)" if len(df) > max_rows else ""
    return "\n".join([head, sep, *rows]) + more


def answer_prompt(question: str, sql: str, df: pd.DataFrame, assumptions: list[str],
                  feedback: str | None = None) -> tuple[str, str]:
    system = (
        "You are a sales performance analyst briefing a business leader. Answer the question from the query "
        "result ONLY.\n"
        "- Lead with the direct answer in one sentence, then up to 3 short supporting points.\n"
        "- Every number you write must appear in the result table (you may round it). Do not compute new "
        "numbers such as differences or growth rates unless they are in the table.\n"
        "- Fractions like 0.923 are percentages: write 92.3%. Money: $4.2M style.\n"
        "- If the result is empty, say so plainly and suggest why.\n"
        "- Mention an assumption only if it changes how to read the answer.\n"
        'Return {"answer": "markdown text", "follow_ups": ["question", "question"]}'
    )
    user = (f"QUESTION: {question}\n\nSQL USED:\n{sql}\n\nASSUMPTIONS: {assumptions}\n\n"
            f"RESULT ({len(df)} rows):\n{df_to_markdown(df)}")
    if feedback:
        user += f"\n\nYOUR PREVIOUS ANSWER FAILED VERIFICATION: {feedback}\nRewrite using only numbers in the result."
    return system, user


def definition_prompt(question: str) -> tuple[str, str]:
    system = ("Explain business terms for a sales performance assistant using ONLY the governed definitions "
              f"below. Be brief and concrete.\n\n{schema_context()}\n\n"
              'Return {"answer": "markdown text"}')
    return system, f"QUESTION: {question}"


def brief_prompt(facts: dict) -> tuple[str, str]:
    system = (
        "Write a weekly sales-performance brief for senior leaders from the FACTS JSON only.\n"
        "Structure (markdown): a one-line headline; '**Where we are**' 2-3 bullets; '**Watch list**' 2-3 bullets; "
        "'**Data confidence**' 1 bullet. Max 170 words. Use only numbers present in FACTS (rounding allowed). "
        "No speculation about causes not supported by FACTS.\n"
        'Return {"brief": "markdown text"}'
    )
    import json
    return system, "FACTS:\n" + json.dumps(facts, indent=2, default=str)
