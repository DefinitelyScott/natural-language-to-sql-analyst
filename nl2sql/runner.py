"""Safe, read-only execution of generated SQL.

Generated SQL is untrusted, so it passes through two independent layers:

1. :func:`validate` — a string-level check (single statement, SELECT-only, a
   denylist of write/DDL keywords). It runs first because it produces clear,
   specific error messages a user can act on.
2. An engine-level *authorizer* — SQLite consults a callback for every
   operation while compiling a statement, and anything outside a small
   read-only allowlist is denied. A denylist can only reject what it thought
   to name; the authorizer inverts that, so a construct the regex never
   anticipated (say, ``ATTACH`` reaching the engine through some phrasing the
   pattern misses — which would let a query open *other files on disk*) is
   still refused. ``mode=ro`` protects the target database file, but only the
   authorizer covers the whole engine surface.

The connection is additionally opened read-only, and results are capped at
``max_rows``.
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


#: SQLite action codes a read-only analytics query legitimately needs.
#:
#: - SQLITE_SELECT: the statement (and any subquery) itself.
#: - SQLITE_READ: each table/column access.
#: - SQLITE_FUNCTION: every function call — aggregates (SUM), scalar functions
#:   (strftime, ROUND), and window functions all authorize through this code.
#: - SQLITE_RECURSIVE: WITH RECURSIVE — recursion over data we may already
#:   read adds no new capability, so there is no reason to refuse it.
#:
#: Everything else — writes, DDL, ATTACH, PRAGMA, transaction control — is
#: denied. Allowlisting what a SELECT needs is a much shorter (and safer) list
#: than trying to enumerate everything that must be forbidden.
_ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)


def _authorizer(
    action: int,
    arg1: str | None,
    arg2: str | None,
    db_name: str | None,
    trigger: str | None,
) -> int:
    """SQLite authorizer callback: allow read-only actions, deny the rest.

    Installed on every connection :func:`run` opens. SQLite invokes it while
    *compiling* a statement, so a denied action fails at prepare time with a
    ``DatabaseError('not authorized')`` — before any part of the query runs.
    The unused arguments carry per-action detail (table name, column name);
    this policy is deliberately coarse and decides on the action code alone.
    """
    del arg1, arg2, db_name, trigger  # policy depends only on the action code
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


@dataclass
class QueryResult:
    """A materialized result set, plus whether the row cap cut it short.

    ``truncated`` is True when the query had more rows available than the
    caller's ``max_rows`` allowed. Without that flag a capped result is
    indistinguishable from a complete one, so an export could quietly drop rows
    while still looking like the whole answer. It defaults to False so a result
    constructed by hand (in tests, say) is a complete one unless it says
    otherwise.
    """

    columns: list[str]
    rows: list[tuple]
    truncated: bool = False

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
    """Execute validated SQL against a read-only connection.

    At most ``max_rows`` rows are returned. One row beyond the cap is fetched
    purely as a probe: the cursor gives no "there is more" signal, so fetching
    exactly ``max_rows`` cannot distinguish a result that happens to be that
    long from one that was cut short. The extra row is discarded and only its
    existence is reported, via ``QueryResult.truncated``.
    """
    cleaned = validate(sql)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.set_authorizer(_authorizer)
    try:
        try:
            cur = conn.execute(cleaned)
        except sqlite3.DatabaseError as exc:
            # The authorizer reports a denial as a generic "not authorized"
            # DatabaseError. Re-raise it as UnsafeQueryError so callers see one
            # exception type for "this SQL was refused", whichever layer
            # refused it; any other DatabaseError (bad table name, SQL syntax
            # error) propagates unchanged.
            if "not authorized" in str(exc):
                raise UnsafeQueryError(
                    f"query denied by the engine-level authorizer: {exc}"
                ) from exc
            raise
        columns = [d[0] for d in cur.description] if cur.description else []
        fetched = cur.fetchmany(max_rows + 1)
        return QueryResult(
            columns=columns,
            rows=fetched[:max_rows],
            truncated=len(fetched) > max_rows,
        )
    finally:
        conn.close()
