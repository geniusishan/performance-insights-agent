"""Read-only query execution with a hard timeout."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import duckdb
import pandas as pd

from pia.settings import policy, warehouse_path


class QueryTimeout(Exception):
    pass


@dataclass
class QueryResult:
    df: pd.DataFrame
    seconds: float


def connect() -> duckdb.DuckDBPyConnection:
    path = warehouse_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Warehouse not found at {path}. Build it with the Field Performance Hub "
            "(`python -m fph.pipeline run`) or set PIA_WAREHOUSE.")
    # read-only file + no filesystem/network access from SQL = second line of defence
    return duckdb.connect(str(path), read_only=True, config={"enable_external_access": False})


def run_query(sql: str, timeout: float | None = None) -> QueryResult:
    timeout = timeout or policy()["limits"]["query_timeout_seconds"]
    con = connect()
    timer = threading.Timer(timeout, con.interrupt)
    t0 = time.perf_counter()
    timer.start()
    try:
        df = con.execute(sql).df()
    except duckdb.InterruptException as e:
        raise QueryTimeout(f"query exceeded {timeout}s and was cancelled") from e
    finally:
        timer.cancel()
        con.close()
    return QueryResult(df=df, seconds=round(time.perf_counter() - t0, 3))


def as_of() -> str:
    con = connect()
    try:
        return str(con.execute("SELECT MAX(as_of_month) FROM scorecard.market").fetchone()[0])
    finally:
        con.close()
