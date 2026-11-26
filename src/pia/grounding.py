"""Numeric grounding check - catches hallucinated numbers before a leader sees them.

Every number in the LLM's narrative must be traceable to the query result (or to a
total derived from it). Numbers are matched with the precision they were written at,
across the forms a writer naturally uses: 0.9234 can appear as 92%, 92.3%, 0.92;
1,234,567 as 1.2M, $1.23 million, 1,234.6K.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NUM_RE = re.compile(
    r"(?<![\w.])[-+]?\$?\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s?(%|percent|pp|k|K|thousand|m|M|mn|million|bn|B|billion)?(?![\w])"
)
SCALE = {"k": 1e3, "K": 1e3, "thousand": 1e3, "m": 1e6, "M": 1e6, "mn": 1e6, "million": 1e6,
         "bn": 1e9, "B": 1e9, "billion": 1e9}


@dataclass
class GroundingResult:
    ok: bool
    checked: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


def extract_numbers(text: str) -> list[tuple[str, float, int, str | None]]:
    """-> (surface text, value, decimals shown, unit)"""
    out = []
    for m in NUM_RE.finditer(text):
        if re.match(r"-[A-Za-z]", text[m.end():m.end() + 2]):
            continue  # part of a label such as "90-day" or "12-month", not a statistic
        whole, frac, unit = m.group(1), m.group(2), m.group(3)
        val = float(whole.replace(",", "") + ("." + frac if frac else ""))
        out.append((m.group(0).strip(), val, len(frac) if frac else 0, unit))
    return out


def _reference_values(df: pd.DataFrame, extra: list[float] | None = None) -> np.ndarray:
    vals = []
    num = df.select_dtypes(include="number")
    for c in num.columns:
        col = num[c].dropna().astype(float)
        vals.extend(col.tolist())
        if len(col) > 1:
            vals.extend([col.sum(), col.mean(), col.max() - col.min()])
    # numbers embedded in text columns, e.g. 'India Territory 3'
    for c in df.select_dtypes(exclude="number").columns:
        for s in df[c].dropna().astype(str).head(500):
            vals.extend(v for _, v, _, _ in extract_numbers(s))
    vals.extend(extra or [])
    return np.array([v for v in vals if math.isfinite(v)], dtype=float)


def _matches(value: float, decimals: int, unit: str | None, refs: np.ndarray) -> bool:
    if refs.size == 0:
        return False
    candidates = []
    if unit in ("%", "percent", "pp"):
        candidates = [refs * 100, refs]          # fraction stored (0.92) or already percent
    elif unit in SCALE:
        candidates = [refs / SCALE[unit], refs]  # raw USD, or a fact already stated in millions
    else:
        candidates = [refs, refs * 100, refs / 1e3, refs / 1e6, refs / 1e9]
    tol = 0.5 * 10 ** (-decimals) + 1e-9           # "92%" covers 91.5-92.5
    for c in candidates:
        if np.any(np.abs(np.abs(c) - abs(value)) <= tol):
            return True
    return False


def check(text: str, df: pd.DataFrame, question: str = "", extra: list[float] | None = None) -> GroundingResult:
    refs = _reference_values(df, extra)
    q_numbers = {v for _, v, _, _ in extract_numbers(question)}
    res = GroundingResult(ok=True)
    for surface, value, decimals, unit in extract_numbers(text):
        if unit is None and decimals == 0 and (value <= 10 or 1990 <= value <= 2100):
            continue  # ordinals/counts like "top 5", "3 territories", and years
        if value in q_numbers:
            continue  # echoing the question
        res.checked.append(surface)
        if not _matches(value, decimals, unit, refs):
            res.unsupported.append(surface)
    res.ok = not res.unsupported
    return res
