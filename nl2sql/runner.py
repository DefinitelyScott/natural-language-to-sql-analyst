"""Safe, read-only execution of generated SQL.

Generated SQL is untrusted. We enforce that it is a single SELECT statement,
open the database read-only, and cap the number of returned rows.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|"
    r"detach|pragma|vacuum|reindex)\b",
    re.IGNORECASE,
)


class UnsafeQueryError(ValueError):
    """Raised when generated SQL is not a safe, single read-only statement."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]

    def __len__(self) -> int:
        return len(self.rows)


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql.strip()


def validate(sql: str) -> str:
    """Validate SQL is a single read-only SELECT. Return the cleaned SQL."""
    cleaned = _strip_comments(sql).rstrip(";").strip()
    if not cleaned:
        raise UnsafeQueryError("empty query")
    if ";" in cleaned:
        raise UnsafeQueryError("multiple statements are not allowed")
    head = cleaned.lstrip("(").lower()
    if not (head.startswith("select") or head.startswith("with")):
        raise UnsafeQueryError("only SELECT/WITH queries are allowed")
    if _FORBIDDEN.search(cleaned):
        raise UnsafeQueryError("query contains a forbidden keyword")
    return cleaned


def run(db_path: str, sql: str, *, max_rows: int = 1000) -> QueryResult:
    """Execute validated SQL against a read-only connection."""
    cleaned = validate(sql)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.execute(cleaned)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(max_rows)
        return QueryResult(columns=columns, rows=rows)
    finally:
        conn.close()
