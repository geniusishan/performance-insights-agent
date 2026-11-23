"""SQL guardrails: deterministic, AST-based policy enforcement.

The LLM is told the rules, but nothing depends on it obeying them. Every generated
query is parsed with sqlglot and must pass all of these before it can run:

  1. parses, and is exactly ONE statement
  2. is a read-only query (SELECT / UNION / CTE) - no DML, DDL, PRAGMA, COPY, ATTACH, SET ...
  3. reads only allow-listed, schema-qualified tables (CTE names are resolved, not trusted)
  4. calls no file / network / system functions
  5. touches no column restricted for the caller's role (including via SELECT *)
  6. returns at most max_rows (a LIMIT is added or tightened automatically)

Then it runs on a read-only DuckDB connection with external access disabled and a
timeout (see db.py): defence in depth, so a guardrail bug alone cannot cause harm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import sqlglot
from sqlglot import exp

from pia.settings import policy, semantic_layer

FORBIDDEN_NODES = tuple(
    getattr(exp, n) for n in (
        "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "AlterTable", "TruncateTable",
        "Command", "Copy", "Pragma", "Set", "Use", "Attach", "Detach", "Grant", "Revoke", "Transaction",
        "Commit", "Rollback", "Install", "Load", "Export",
    ) if hasattr(exp, n)
)


@dataclass
class Verdict:
    ok: bool
    sql: str = ""
    reasons: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> "Verdict":
        self.ok = False
        self.reasons.append(reason)
        return self


@lru_cache(maxsize=1)
def _table_columns() -> dict[str, set[str]]:
    """Real column lists from the warehouse catalog (what SELECT * would actually return);
    falls back to the semantic layer if the warehouse is unavailable."""
    try:
        from pia.db import connect
        con = connect()
        try:
            rows = con.execute("SELECT table_schema || '.' || table_name, column_name "
                               "FROM information_schema.columns").fetchall()
        finally:
            con.close()
        out: dict[str, set[str]] = {}
        for t, c in rows:
            out.setdefault(t.lower(), set()).add(c.lower())
        return out
    except Exception:  # noqa: BLE001
        return {t: set((spec.get("columns") or {}).keys()) for t, spec in semantic_layer()["tables"].items()}


def validate(sql: str, role: str = "analyst") -> Verdict:
    pol = policy()
    v = Verdict(ok=True)
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        return v.fail("empty query")

    # 1. parse, single statement
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as e:
        return v.fail(f"SQL does not parse: {str(e).splitlines()[0]}")
    if len(statements) != 1:
        return v.fail(f"exactly one statement allowed, got {len(statements)}")
    tree = statements[0]
    v.checks.append("single statement")

    # 2. read-only
    if not isinstance(tree, exp.Query):
        return v.fail(f"only SELECT queries are allowed (got {type(tree).__name__.upper()})")
    bad = [type(n).__name__.upper() for n in tree.walk() if isinstance(n, FORBIDDEN_NODES)]
    if bad:
        return v.fail(f"forbidden operation(s): {', '.join(sorted(set(bad)))}")
    v.checks.append("read-only")

    # 3. allow-listed tables
    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    allowed = {t.lower() for t in pol["allowed_tables"]}
    used = []
    for t in tree.find_all(exp.Table):
        if not isinstance(t.this, exp.Identifier):  # table functions: read_csv(...), range(...) etc.
            v.fail(f"table functions are not allowed: {t.this.sql()[:60]}")
            continue
        name = t.name.lower()
        db = (t.db or "").lower()
        if t.catalog:
            v.fail(f"cross-database reference not allowed: {t.sql()}")
            continue
        if not db and name in cte_names:
            continue
        full = f"{db}.{name}" if db else name
        if full not in allowed:
            v.fail(f"table not allowed: {full}" + ("" if db else " (tables must be schema-qualified)"))
        else:
            used.append(full)
    if not v.ok:
        return v
    if not used:
        return v.fail("query reads no allow-listed table")
    v.tables = sorted(set(used))
    v.checks.append(f"tables allowed: {', '.join(v.tables)}")

    # 4. blocked functions
    blocked = {f.lower() for f in pol["blocked_functions"]}
    for fn in tree.find_all(exp.Func):
        fname = (fn.name if isinstance(fn, exp.Anonymous) else fn.sql_name()).lower()
        if fname in blocked or (isinstance(fn, exp.Anonymous) and fname.startswith(("read_", "pragma_", "duckdb_"))):
            return v.fail(f"function not allowed: {fname}")
    v.checks.append("no system/file functions")

    # 5. restricted columns for this role
    restricted = {c.lower() for c in pol["roles"].get(role, pol["roles"]["viewer"])["restricted_columns"]}
    if restricted:
        cols = {c.name.lower() for c in tree.find_all(exp.Column)}
        hit = cols & restricted
        if hit:
            return v.fail(f"column(s) restricted for role '{role}': {', '.join(sorted(hit))}")
        if any(isinstance(s, exp.Star) or (isinstance(s, exp.Column) and isinstance(s.this, exp.Star))
               for s in tree.find_all(exp.Star, exp.Column)):
            tcols = _table_columns()
            exposed = set().union(*(tcols.get(t, set()) for t in v.tables)) & restricted
            if exposed:
                return v.fail(f"SELECT * would expose restricted column(s) for role '{role}': {', '.join(sorted(exposed))}")
    v.checks.append(f"column policy for role '{role}'")

    # 6. row limit
    max_rows = pol["limits"]["max_rows"]
    limit = tree.args.get("limit")
    current = None
    if limit is not None:
        try:
            current = int(limit.expression.this) if limit.expression is not None else None
        except (TypeError, ValueError, AttributeError):
            current = None
    if current is None or current > max_rows:
        tree = tree.limit(max_rows) if not isinstance(tree, exp.Union) else exp.select("*").from_(tree.subquery("q")).limit(max_rows)
        v.checks.append(f"row limit enforced ({max_rows})")
    else:
        v.checks.append(f"row limit {current}")

    v.sql = tree.sql(dialect="duckdb", pretty=True)
    return v
