"""Safe, read-only execution of generated SQL.

Generated SQL is untrusted, so it passes through three independent layers:

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
3. A wall-clock *deadline*. The first two layers decide whether a query may
   run at all; neither says anything about how long it may run for, and a
   perfectly read-only ``SELECT`` can still be ruinously expensive — an
   accidental cross join is the usual way, and it is exactly the mistake a
   model writing SQL from a schema makes. Left alone it would hold the process
   open indefinitely, which is a denial of service whether or not anyone meant
   it. A progress handler checks the clock while the statement runs and
   cancels it once ``timeout_ms`` has elapsed.

The connection is additionally opened read-only, and results are capped at
``max_rows``.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|"
    r"detach|pragma|vacuum|reindex)\b",
    re.IGNORECASE,
)


class UnsafeQueryError(ValueError):
    """Raised when generated SQL is not a safe, single read-only statement."""


class QueryTimeoutError(RuntimeError):
    """Raised when a query was cancelled for exceeding its execution deadline.

    Deliberately *not* a subclass of :class:`UnsafeQueryError` or of
    ``sqlite3.DatabaseError``. A timeout is not a verdict about the SQL: the
    query may be entirely correct and merely expensive, and the error text says
    nothing a rewrite could act on. Keeping it outside the ``DatabaseError``
    hierarchy is what stops :func:`nl2sql.generator.answer_question` from
    handing it to the repair loop, where a second attempt could be no
    better-informed than the first and would cost another full timeout.
    """


#: Default wall-clock budget for one query, in milliseconds.
#:
#: Every query in the offline catalog answers in single-digit milliseconds
#: against the sample database, so five seconds leaves roughly three orders of
#: magnitude of headroom: a query that reaches this limit is not slow, it is
#: wrong. It is a default rather than a fixed constant because a legitimately
#: long analysis over a larger database is a real use — callers can raise it,
#: or pass ``None`` to opt out.
DEFAULT_TIMEOUT_MS = 5_000

#: SQLite virtual-machine instructions between deadline checks.
#:
#: How finely the deadline is sampled. Small enough that cancellation is prompt
#: even where individual instructions do little work, large enough that the
#: extra ``monotonic()`` calls are lost in the cost of running the query.
_PROGRESS_STEP_INTERVAL = 1_000


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


def _install_deadline(conn: sqlite3.Connection, timeout_ms: int) -> list[bool]:
    """Arm a wall-clock deadline on ``conn``; return its "expired" flag.

    SQLite calls the registered progress handler every
    :data:`_PROGRESS_STEP_INTERVAL` virtual-machine instructions and aborts the
    running statement if it returns non-zero. The handler runs on the calling
    thread, between instructions, so this needs no watchdog thread and has no
    race to get wrong: cancellation happens at a point where the engine is
    already prepared to stop. (``Connection.interrupt`` from a second thread is
    the alternative, and it buys nothing here while adding one.)

    The abort surfaces as a generic ``OperationalError``, indistinguishable by
    type from any other execution failure. The returned single-element list is
    how :func:`run` tells them apart: the handler sets it *before* asking for
    the abort, so a truthy flag means this deadline is what failed the query.
    Reading a flag we set ourselves beats matching on SQLite's message text,
    which is not part of its API and is free to change.
    """
    deadline = time.monotonic() + timeout_ms / 1000
    expired = [False]

    def handler() -> int:
        if time.monotonic() < deadline:
            return 0
        expired[0] = True
        return 1

    conn.set_progress_handler(handler, _PROGRESS_STEP_INTERVAL)
    return expired


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
    #: Rows as ``sqlite3`` returns them. The cell type is ``Any`` because
    #: SQLite's storage classes are per *value*, not per column: one column can
    #: hand back an int, a float, a str, bytes or None, and with no row factory
    #: configured the driver maps each to whichever Python type fits. Narrowing
    #: this to a union would be a claim the database does not make, and every
    #: consumer here (``output``, the eval harness) stringifies cells anyway.
    rows: list[tuple[Any, ...]]
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


def run(
    db_path: str,
    sql: str,
    *,
    max_rows: int = 1000,
    timeout_ms: int | None = DEFAULT_TIMEOUT_MS,
) -> QueryResult:
    """Execute validated SQL against a read-only connection.

    At most ``max_rows`` rows are returned. One row beyond the cap is fetched
    purely as a probe: the cursor gives no "there is more" signal, so fetching
    exactly ``max_rows`` cannot distinguish a result that happens to be that
    long from one that was cut short. The extra row is discarded and only its
    existence is reported, via ``QueryResult.truncated``.

    The query is cancelled with :class:`QueryTimeoutError` if it is still
    running ``timeout_ms`` milliseconds after execution begins. The deadline
    covers fetching as well as compiling and executing, because a query can be
    cheap to start and expensive to drain — a row cap bounds how much comes
    back, not how much work SQLite does to produce it. Pass ``timeout_ms=None``
    to run with no deadline.
    """
    # Checked before the SQL is even looked at: a bad ``timeout_ms`` is the
    # caller's own mistake, and reporting it as such is clearer than letting it
    # surface later as a query that never times out.
    if timeout_ms is not None and timeout_ms <= 0:
        raise ValueError(
            "timeout_ms must be a positive number of milliseconds, "
            "or None to run without a deadline"
        )

    cleaned = validate(sql)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.set_authorizer(_authorizer)
    expired = [False] if timeout_ms is None else _install_deadline(conn, timeout_ms)
    try:
        try:
            cur = conn.execute(cleaned)
            columns = [d[0] for d in cur.description] if cur.description else []
            fetched = cur.fetchmany(max_rows + 1)
        except sqlite3.DatabaseError as exc:
            # Tested first: an abort raises an ordinary OperationalError, so
            # without the flag it would fall through to the clauses below and
            # be reported as an unexplained engine error.
            if expired[0]:
                raise QueryTimeoutError(
                    f"query cancelled after exceeding its {timeout_ms} ms "
                    "execution deadline"
                ) from exc
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
        return QueryResult(
            columns=columns,
            rows=fetched[:max_rows],
            truncated=len(fetched) > max_rows,
        )
    finally:
        conn.close()
