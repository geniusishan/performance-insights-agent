"""Append-only audit trail: who asked what, which SQL ran, what the guardrails decided,
whether the answer's numbers verified, and how long each step took."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from pia.settings import audit_path

LOG_FILE = "audit_log.jsonl"


def record(entry: dict) -> None:
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    if entry.get("answer"):
        entry["answer_sha256"] = hashlib.sha256(entry["answer"].encode()).hexdigest()[:16]
    with open(audit_path() / LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def load(limit: int = 500) -> pd.DataFrame:
    p = audit_path() / LOG_FILE
    if not p.exists():
        return pd.DataFrame()
    lines = p.read_text(encoding="utf-8").splitlines()[-limit:]
    return pd.DataFrame([json.loads(l) for l in lines]).iloc[::-1].reset_index(drop=True)
