import pytest

from pia.evals.run_evals import guardrail_suite, load
from pia.guardrails import validate


def test_red_team_fully_blocked_and_no_false_positives():
    r = guardrail_suite(load())
    assert r["leaked"] == []
    assert r["false_positives"] == []


@pytest.mark.parametrize("sql", [
    "select market_name, score from scorecard.market",
    "with t as (select market_id, sum(signings) s from marts.rep_monthly group by 1) select * from t",
    "select market_id from scorecard.market union all select market_id from forecast.signings",
])
def test_legitimate_queries_pass(sql):
    v = validate(sql)
    assert v.ok, v.reasons


def test_limit_added_and_tightened():
    assert "LIMIT 500" in validate("select * from marts.rep_monthly").sql
    assert "LIMIT 500" in validate("select * from marts.rep_monthly limit 99999").sql
    assert "LIMIT 10" in validate("select * from marts.rep_monthly limit 10").sql


def test_role_based_column_restriction():
    assert validate("select rep_name from scorecard.rep", "analyst").ok
    assert not validate("select rep_name from scorecard.rep", "viewer").ok
    assert not validate("select * from scorecard.rep", "viewer").ok
    assert validate("select rep_id, score from scorecard.rep", "viewer").ok
    assert validate("select * from scorecard.market", "viewer").ok  # no restricted column there


def test_cte_cannot_shadow_forbidden_schema():
    assert not validate("with raw as (select 1) select * from raw.merchants").ok
