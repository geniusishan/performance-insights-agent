"""Weekly executive brief: facts are computed in SQL, the LLM only writes the prose.

    python -m pia.brief            # prints markdown, saves to briefs/

1. A fixed, reviewed set of SQL queries builds a FACTS pack (no LLM involved).
2. The LLM turns FACTS into a short leadership brief.
3. Every number in the brief is verified against FACTS. If verification fails twice,
   a deterministic template brief is published instead - the pipeline never ships
   an unverified number.
"""
from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from pia import grounding, prompts
from pia.db import run_query
from pia.llm import LLM, default_llm
from pia.settings import ROOT


def facts() -> dict:
    q = lambda sql: run_query(sql).df  # noqa: E731
    g = q("""SELECT MAX(as_of_month) AS as_of, SUM(signings) AS ytd_signings, SUM(target_signings) AS ytd_target,
                    SUM(signings)/SUM(target_signings) AS attainment,
                    SUM(new_volume) AS ytd_new_volume_usd, SUM(new_volume)/SUM(target_new_volume) AS volume_attainment,
                    SUM(activated_90d)/SUM(activation_eligible) AS activation_rate
             FROM scorecard.market""").iloc[0]
    as_of = pd.Timestamp(g["as_of"])
    prev = (as_of - pd.DateOffset(months=1)).date()
    markets = q("SELECT market_name, ROUND(score, 3) AS score, tier, biggest_gap FROM scorecard.market ORDER BY score DESC")
    mom = q(f"""SELECT market_name,
                       SUM(signings) FILTER (WHERE month = DATE '{as_of.date()}') AS this_month,
                       SUM(signings) FILTER (WHERE month = DATE '{prev}') AS last_month
                FROM marts.market_monthly GROUP BY 1""")
    mom["change_pct"] = (mom["this_month"] / mom["last_month"] - 1).round(3)
    mom = mom.sort_values("change_pct")
    watch = q("SELECT territory_name, ROUND(score, 3) AS score, biggest_gap FROM scorecard.territory ORDER BY score LIMIT 3")
    gaps = q("""SELECT market_name, ROUND(gap) AS gap, ROUND(gap_pct, 3) AS gap_pct, main_driver,
                       ROUND(headcount_effect) AS headcount_effect, ROUND(ramp_effect) AS ramp_effect,
                       ROUND(productivity_effect) AS productivity_effect
                FROM insights.gap_drivers WHERE level = 'market' ORDER BY gap LIMIT 3""")
    tiers = q("SELECT tier, COUNT(*) AS reps FROM scorecard.rep GROUP BY tier")
    dq = q("""SELECT COUNT(*) FILTER (WHERE severity='CRITICAL' AND NOT passed) AS critical_failures,
                     COUNT(*) AS checks,
                     COALESCE(SUM(failing_rows) FILTER (WHERE category='Validity' AND severity='WARNING'), 0) AS rows_quarantined
              FROM dq.results""").iloc[0]
    nq = q(f"""SELECT SUM(forecast) AS next_3m_forecast FROM forecast.signings
               WHERE month > DATE '{as_of.date()}' AND month <= DATE '{(as_of + pd.DateOffset(months=3)).date()}'""").iloc[0]
    return {
        "as_of_month": str(as_of.date()),
        "ytd": {"signings": int(g["ytd_signings"]), "target": round(float(g["ytd_target"])),
                "attainment": round(float(g["attainment"]), 3),
                "new_volume_usd_m": round(float(g["ytd_new_volume_usd"]) / 1e6, 1),
                "volume_attainment": round(float(g["volume_attainment"]), 3),
                "activation_rate_90d": round(float(g["activation_rate"]), 3)},
        "markets_by_score": markets.to_dict("records"),
        "month_over_month_signings": mom.to_dict("records"),
        "territories_to_watch": watch.to_dict("records"),
        "largest_gaps_to_target_with_drivers": gaps.to_dict("records"),
        "rep_tiers": dict(zip(tiers["tier"], tiers["reps"].astype(int))),
        "next_3_months_signings_forecast": round(float(nq["next_3m_forecast"])),
        "data_quality": {k: int(v) for k, v in dq.items()},
    }


def _flatten_numbers(obj) -> list[float]:
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _flatten_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _flatten_numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    elif isinstance(obj, str):
        out += [v for _, v, _, _ in grounding.extract_numbers(obj)]
    return out


def template_brief(f: dict) -> str:
    y, top, bottom = f["ytd"], f["markets_by_score"][0], f["markets_by_score"][-1]
    w = f["territories_to_watch"]
    return (
        f"**Signings at {y['attainment']:.1%} of YTD target ({y['signings']:,} vs {y['target']:,}).**\n\n"
        f"**Where we are**\n- New-merchant volume ${y['new_volume_usd_m']:,.1f}M ({y['volume_attainment']:.1%} of target); "
        f"90-day activation {y['activation_rate_90d']:.1%}.\n"
        f"- Leading market: {top['market_name']} (score {top['score']:.1%}); trailing: {bottom['market_name']} "
        f"({bottom['score']:.1%}, gap: {bottom['biggest_gap']}).\n"
        f"- Next 3 months forecast: {f['next_3_months_signings_forecast']:,} signings.\n\n"
        f"**Why we're behind**\n" + "".join(
            f"- {g['market_name']}: {g['gap']:+,.0f} signings ({g['gap_pct']:+.1%}), mainly {g['main_driver'].lower()} "
            f"(headcount {g['headcount_effect']:+,.0f}, ramp & mix {g['ramp_effect']:+,.0f}, productivity {g['productivity_effect']:+,.0f})\n"
            for g in f["largest_gaps_to_target_with_drivers"] if g["gap"] < 0)
        + f"\n**Watch list**\n" + "".join(f"- {t['territory_name']}: score {t['score']:.1%} ({t['biggest_gap']})\n" for t in w)
        + f"\n**Data confidence**\n- {f['data_quality']['critical_failures']} critical data-quality failures across "
        f"{f['data_quality']['checks']} checks; {f['data_quality']['rows_quarantined']:,} source rows quarantined.\n"
    )


def generate(llm: LLM | None = None) -> dict:
    f = facts()
    refs = pd.DataFrame({"v": _flatten_numbers(f)})
    try:
        llm = llm or default_llm()
        for attempt in range(2):
            system, user = prompts.brief_prompt(f)
            if attempt:
                user += f"\n\nPrevious draft used numbers not in FACTS: {g.unsupported}. Use only FACTS numbers."
            text = llm.complete_json("brief", system, user).data.get("brief", "")
            g = grounding.check(text, refs)
            if g.ok and text:
                return {"brief": text, "source": f"llm:{llm.name}", "verified_numbers": g.checked, "facts": f}
        reason = f"LLM draft failed verification ({g.unsupported})"
    except Exception as e:  # noqa: BLE001 - no key / network: still deliver the brief
        reason = f"LLM unavailable ({type(e).__name__})"
    return {"brief": template_brief(f), "source": f"template ({reason})", "verified_numbers": [], "facts": f}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true", help="skip the LLM, use the deterministic template")
    args = ap.parse_args()
    out = {"brief": template_brief(facts()), "source": "template"} if args.template else generate()
    d = ROOT / "briefs"
    d.mkdir(exist_ok=True)
    path = d / f"brief_{date.today():%Y%m%d}.md"
    path.write_text(f"# Weekly performance brief\n\n{out['brief']}\n\n_Source: {out['source']}_\n")
    print(path.read_text())


if __name__ == "__main__":
    main()
