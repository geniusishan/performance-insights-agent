"""Evaluation harness.

    python -m pia.evals.run_evals --offline   # guardrail red-team + reference checks, no API key needed
    python -m pia.evals.run_evals             # + full agent eval with your configured LLM

Metrics
  * execution accuracy   - golden questions whose result matches the reference result
  * behaviour pass rate  - definitions, refusals, role-based column restrictions
  * guardrail block rate - red-team SQL blocked (target 100%)
  * false-positive rate  - legitimate reference SQL wrongly blocked (target 0%)
  * grounding rate       - answers whose every number traced back to data
  * latency p50 / p95, tokens per question
Writes evals/report.md and evals/report.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pia import guardrails
from pia.db import run_query
from pia.settings import ROOT

GOLDEN = Path(__file__).with_name("golden.yaml")
OUT = ROOT / "evals"


def load() -> dict:
    return yaml.safe_load(GOLDEN.read_text())


def _norm(s: pd.Series) -> list:
    if pd.api.types.is_numeric_dtype(s):
        return sorted(float(x) for x in s.dropna())
    if pd.api.types.is_datetime64_any_dtype(s):
        return sorted(str(pd.Timestamp(x).date()) for x in s.dropna())
    return sorted(str(x).strip().lower() for x in s.dropna())


def _col_match(ref: list, pred: list) -> bool:
    if len(ref) != len(pred):
        return False
    if ref and isinstance(ref[0], float):
        if not pred or not isinstance(pred[0], float):
            return False
        r, p = np.array(ref), np.array(pred)
        for scale in (1, 100, 0.01):
            if np.allclose(np.sort(r * scale), p, rtol=0.01, atol=0.006):
                return True
        return False
    return [str(x)[:10] for x in ref] == [str(x)[:10] for x in pred]


def results_match(ref: pd.DataFrame, pred: pd.DataFrame | None) -> bool:
    """Every reference column must be matched by some predicted column."""
    if pred is None:
        return False
    pred_cols = [_norm(pred[c]) for c in pred.columns]
    return all(any(_col_match(_norm(ref[c]), pc) for pc in pred_cols) for c in ref.columns)


def guardrail_suite(spec: dict) -> dict:
    blocked, leaks = 0, []
    for sql in spec["red_team_sql"]:
        role = "viewer" if "as viewer" in sql else "analyst"
        if not guardrails.validate(sql.split("--")[0], role).ok:
            blocked += 1
        else:
            leaks.append(sql)
    fps = [q["id"] for q in spec["questions"] if not guardrails.validate(q["sql"], "analyst").ok]
    return {"red_team_total": len(spec["red_team_sql"]), "red_team_blocked": blocked, "leaked": leaks,
            "reference_total": len(spec["questions"]), "false_positives": fps}


def reference_suite(spec: dict) -> list[str]:
    """Every reference query must run and return rows - keeps the eval set honest."""
    bad = []
    for q in spec["questions"]:
        try:
            if run_query(guardrails.validate(q["sql"]).sql).df.empty:
                bad.append(f"{q['id']}: empty")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{q['id']}: {e}")
    return bad


def agent_suite(spec: dict, agent) -> tuple[list[dict], list[dict]]:
    rows = []
    for q in spec["questions"]:
        ref = run_query(guardrails.validate(q["sql"]).sql).df
        t0 = time.perf_counter()
        try:
            r = agent.ask(q["question"], session_id="eval")
            ok = results_match(ref, r.data)
            rows.append({"id": q["id"], "question": q["question"], "pass": ok, "status": r.status,
                         "grounded": not r.grounding_unsupported, "repairs": len(r.attempts),
                         "seconds": r.seconds, "tokens": r.tokens, "sql": r.sql})
        except Exception as e:  # noqa: BLE001
            rows.append({"id": q["id"], "question": q["question"], "pass": False, "status": f"error: {e}",
                         "grounded": False, "repairs": 0, "seconds": time.perf_counter() - t0, "tokens": 0, "sql": None})
        print(f"  {q['id']} {'PASS' if rows[-1]['pass'] else 'FAIL'}  {rows[-1]['status']}  {q['question']}")
    beh = []
    for b in spec["behaviour"]:
        r = agent.ask(b["question"], role=b.get("role", "analyst"), session_id="eval")
        ok = True
        if "expect_status" in b:
            ok &= r.status in b["expect_status"]
        if "expect_no_columns" in b and r.data is not None:
            ok &= not set(b["expect_no_columns"]) & set(r.data.columns)
        if "expect_tables_not" in b and r.sql:
            ok &= not any(t in r.sql.lower() for t in b["expect_tables_not"])
        beh.append({"id": b["id"], "question": b["question"], "pass": bool(ok), "status": r.status})
        print(f"  {b['id']} {'PASS' if ok else 'FAIL'}  {r.status}  {b['question']}")
    return rows, beh


def write_report(g: dict, ref_bad: list[str], rows: list[dict] | None, beh: list[dict] | None, model: str) -> dict:
    OUT.mkdir(exist_ok=True)
    summary = {
        "model": model,
        "guardrail_block_rate": g["red_team_blocked"] / g["red_team_total"],
        "guardrail_false_positive_rate": len(g["false_positives"]) / g["reference_total"],
        "reference_queries_valid": not ref_bad,
    }
    if rows:
        lat = sorted(r["seconds"] for r in rows)
        summary |= {
            "execution_accuracy": sum(r["pass"] for r in rows) / len(rows),
            "behaviour_pass_rate": sum(b["pass"] for b in beh) / len(beh),
            "grounded_answer_rate": sum(r["grounded"] for r in rows) / len(rows),
            "self_repairs": sum(r["repairs"] for r in rows),
            "latency_p50_s": round(statistics.median(lat), 2),
            "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2),
            "avg_tokens": round(sum(r["tokens"] for r in rows) / len(rows)),
        }
    md = [f"# Evaluation report\n\nModel: `{model}`  ·  generated {time.strftime('%Y-%m-%d %H:%M')}\n",
          "| Metric | Result |", "|---|---|"]
    fmt = lambda k, v: f"{v:.0%}" if isinstance(v, float) and ("rate" in k or "accuracy" in k) else str(v)  # noqa: E731
    md += [f"| {k} | {fmt(k, v)} |" for k, v in summary.items() if k != "model"]
    if g["leaked"]:
        md += ["", "**Leaked red-team queries:**"] + [f"- `{s}`" for s in g["leaked"]]
    if rows:
        md += ["", "## Golden questions", "", "| id | pass | status | repairs | s | question |", "|---|---|---|---|---|---|"]
        md += [f"| {r['id']} | {'✅' if r['pass'] else '❌'} | {r['status']} | {r['repairs']} | {r['seconds']:.1f} | {r['question']} |" for r in rows]
        md += ["", "## Behaviour", "", "| id | pass | status | question |", "|---|---|---|---|"]
        md += [f"| {b['id']} | {'✅' if b['pass'] else '❌'} | {b['status']} | {b['question']} |" for b in beh]
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    (OUT / "report.json").write_text(json.dumps({"summary": summary, "guardrails": g, "reference_errors": ref_bad,
                                                 "golden": rows, "behaviour": beh}, indent=2, default=str))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip LLM-based evals")
    args = ap.parse_args()
    spec = load()
    print("Guardrail red-team suite ...")
    g = guardrail_suite(spec)
    ref_bad = reference_suite(spec)
    rows = beh = None
    model = "n/a (offline)"
    if not args.offline:
        from pia.agent import InsightsAgent
        agent = InsightsAgent()
        model = agent.llm.name
        print(f"Agent suite with {model} ...")
        rows, beh = agent_suite(spec, agent)
    s = write_report(g, ref_bad, rows, beh, model)
    print(json.dumps(s, indent=2))
    if s["guardrail_block_rate"] < 1 or s["guardrail_false_positive_rate"] > 0 or ref_bad:
        raise SystemExit("Guardrail or reference-set regression")


if __name__ == "__main__":
    main()
