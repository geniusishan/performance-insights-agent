"""Pick a sensible chart from the shape of the result - deterministic, no LLM needed."""
from __future__ import annotations

import pandas as pd

PREFERRED = ("score", "att_", "attainment", "rate", "signings", "volume", "forecast", "target")


def _pick_measure(nums: list[str]) -> str:
    for p in PREFERRED:
        for c in nums:
            if p in c.lower():
                return c
    return nums[0]


def choose(df: pd.DataFrame) -> dict | None:
    if df is None or df.empty or len(df) < 2:
        return None
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.lower().startswith(("rank", "plan_year"))]
    if not nums:
        return None
    time_cols = [c for c in df.columns if c.lower() in ("month", "signed_month", "date") or pd.api.types.is_datetime64_any_dtype(df[c])]
    cats = [c for c in df.columns if c not in nums and c not in time_cols and not pd.api.types.is_numeric_dtype(df[c])]
    y = _pick_measure(nums)
    is_pct = df[y].dropna().between(0, 2).all() and any(k in y.lower() for k in ("att", "rate", "mix", "score", "share", "pct"))
    if time_cols:
        color = next((c for c in cats if df[c].nunique() <= 10), None)
        return {"type": "line", "x": time_cols[0], "y": y, "color": color, "percent": is_pct}
    if cats and len(df) <= 40:
        return {"type": "bar", "x": cats[0], "y": y, "color": None, "percent": is_pct}
    return None
