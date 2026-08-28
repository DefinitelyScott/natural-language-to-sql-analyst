"""Orchestrate: question -> schema context -> SQL -> executed result.

Two entry points, deliberately separated:

* :func:`answer_question` generates SQL *and executes it* — what ``nl2sql ask``
  does.
* :func:`explain_question` generates SQL and inspects it *without touching the
  data* — what ``nl2sql explain`` does. It reports which offline rule the
  question routed to, which other rules also matched but were shadowed by it,
  and whether the SQL would survive ``runner.validate``.

Splitting them keeps the dry run honest: an explanation cannot accidentally run
a query, so it is safe to point at SQL you do not yet trust (an LLM's, say).

:func:`answer_question` also owns the *repair loop*: when generated SQL fails to
execute, a backend that supports it is shown the error and given one chance to
rewrite the query. See :data:`MAX_REPAIR_ATTEMPTS` for why the budget is one.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from . import llm, runner, schema

#: How many times :func:`answer_question` will ask a backend to rewrite SQL that
#: failed, before giving up and raising.
#:
#: One, deliberately. The errors a repair actually fixes are the shallow ones —
#: a hallucinated column, a function SQLite does not have — and the model is
#: told exactly what was wrong, so if a second attempt is needed the problem is
#: almost always a misreading of the schema that more retries will not resolve.
#: A fixed, small budget also keeps the worst-case cost of answering one
#: question bounded and obvious (at most two model calls, two executions)
#: instead of leaving latency and token spend open-ended.
MAX_REPAIR_ATTEMPTS = 1


class QueryFailedError(RuntimeError):
    """Generated SQL could not be executed, even after any repair attempts.

    Subclasses :class:`RuntimeError` so callers that already handle "the
    backend could not answer this" keep working. The message carries every
    attempt — each failed query and the error it raised — because with a repair
    loop in play the final error alone is misleading: it describes the *last*
    query tried, not the one the backend originally wrote.
    """


@dataclass(frozen=True)
class RepairAttempt:
    """A query that failed and the error that prompted a rewrite."""

    sql: str
    error: str


@dataclass
class Answer:
    question: str
    sql: str
    result: runner.QueryResult
    #: Failed attempts that preceded ``sql``, oldest first; empty on the common
    #: path. Recorded rather than discarded so a caller can disclose that the
    #: answer took a retry — a result that needed repairing is a weaker signal
    #: than one that ran first time, and silently hiding that would overstate
    #: how well the backend performed.
    repairs: list[RepairAttempt] = field(default_factory=list)


@dataclass
class Explanation:
    """A dry-run report: the SQL a question resolves to, and how it got there.

    ``matched_rule`` / ``shadowed_rules`` are populated only for the offline
    backend, which routes by an ordered regex catalog and can therefore say
    *why* a question produced this SQL. The LLM backend has no such structure,
    so both stay empty and only ``sql`` and the safety verdict are meaningful.

    ``safety_error`` is the message ``runner.validate`` rejected the SQL with,
    or ``None`` when it passed. It is captured rather than raised because
    reporting an unsafe query is the whole point of the command — the caller
    wants to see the offending SQL alongside the reason it was rejected.
    """

    question: str
    backend: str
    sql: str
    matched_rule: int | None = None
    matched_pattern: str | None = None
    shadowed_rules: list[tuple[int, str]] = field(default_factory=list)
    safety_error: str | None = None

    @property
    def is_safe(self) -> bool:
        """True when the generated SQL passed ``runner.validate``."""
        return self.safety_error is None


def _resolve(
    db_path: str, question: str, *, use_llm: bool
) -> tuple[llm.Backend, str, str]:
    """Return the backend, the schema text it was given, and the SQL it produced.

    The backend is handed back alongside the SQL so a caller that wants to
    introspect the routing (:func:`explain_question`) does not have to construct
    a second one — which for the LLM backend would mean a second API client. The
    schema text comes back for the same reason: a repair has to re-send it, and
    re-introspecting the database would risk repairing against a different
    schema than the one the failed query was written from.
    """
    schema_text = schema.schema_context(db_path)
    backend = llm.get_backend(use_llm)
    return backend, schema_text, backend.to_sql(question, schema_text)


def _describe_failure(
    sql: str, error: str, repairs: list[RepairAttempt]
) -> str:
    """Build the message for :class:`QueryFailedError`.

    Every attempt is listed, not just the last, so the reader can tell a repair
    that made things worse (a different error each time) from one that changed
    nothing (the same error twice) — which is the difference between a prompt
    problem and a schema the backend simply cannot work with.
    """
    if not repairs:
        return f"generated SQL failed to execute: {error}\n  SQL: {sql}"

    attempts = len(repairs)
    lines = [
        f"generated SQL failed to execute after {attempts} "
        f"repair attempt{'s' if attempts != 1 else ''}: {error}"
    ]
    for number, attempt in enumerate(repairs, start=1):
        lines.append(f"  attempt {number}: {attempt.error}")
        lines.append(f"    SQL: {attempt.sql}")
    lines.append(f"  attempt {attempts + 1}: {error}")
    lines.append(f"    SQL: {sql}")
    return "\n".join(lines)


def generate_sql(db_path: str, question: str, *, use_llm: bool = False) -> str:
    """Generate SQL for ``question`` without executing it."""
    _, _, sql = _resolve(db_path, question, use_llm=use_llm)
    return sql


def answer_question(
    db_path: str,
    question: str,
    *,
    use_llm: bool = False,
    max_rows: int = 1000,
    timeout_ms: int | None = runner.DEFAULT_TIMEOUT_MS,
) -> Answer:
    """Generate SQL for ``question``, execute it, and repair it if it fails.

    When execution fails and the backend implements
    :class:`llm.RepairingBackend`, the failed query and its error are handed
    back to the backend for a rewrite, up to :data:`MAX_REPAIR_ATTEMPTS` times.
    A rewritten query is re-executed through :func:`runner.run` exactly like the
    first one, so it faces the same validator, the same authorizer and the same
    read-only connection — the loop can change *what* is run, never what is
    allowed to run.

    Raises :class:`QueryFailedError` when the budget is exhausted, or
    immediately when the backend cannot repair. Both the execution errors
    SQLite raises and :class:`runner.UnsafeQueryError` are treated as
    repairable: a model that emitted two statements or a ``PRAGMA`` has made an
    ordinary mistake and can be told so, and the rewrite is re-validated
    anyway, so nothing rejected on the first pass can slip through on the
    second.

    A :class:`runner.QueryTimeoutError` is the one execution failure that is
    *not* repaired: it propagates immediately, ending the loop. The premise of
    a repair is that the engine's error names something specific to fix — a
    column, a function, a join key. A deadline names nothing; it reports only
    that the query was expensive, so a rewrite would be a blind guess costing
    another full ``timeout_ms``. That the exception sits outside the
    ``sqlite3.DatabaseError`` hierarchy is what enforces this, rather than a
    check here that a later edit could forget.
    """
    backend, schema_text, sql = _resolve(db_path, question, use_llm=use_llm)
    repairs: list[RepairAttempt] = []

    while True:
        try:
            result = runner.run(db_path, sql, max_rows=max_rows, timeout_ms=timeout_ms)
        except (sqlite3.DatabaseError, runner.UnsafeQueryError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            repairable = isinstance(backend, llm.RepairingBackend)
            if not repairable or len(repairs) >= MAX_REPAIR_ATTEMPTS:
                raise QueryFailedError(
                    _describe_failure(sql, error, repairs)
                ) from exc
            repairs.append(RepairAttempt(sql=sql, error=error))
            sql = backend.repair(question, schema_text, sql, error)
            continue

        return Answer(question=question, sql=sql, result=result, repairs=repairs)


def explain_question(
    db_path: str, question: str, *, use_llm: bool = False
) -> Explanation:
    """Generate SQL for ``question`` and describe it, without executing it.

    No repair happens here, and cannot: a repair is driven by an execution
    error, and a dry run never executes. What ``explain`` shows is therefore
    always the backend's first attempt.
    """
    backend, _, sql = _resolve(db_path, question, use_llm=use_llm)

    explanation = Explanation(
        question=question,
        backend="llm" if use_llm else "offline",
        sql=sql,
    )

    if isinstance(backend, llm.OfflineBackend):
        matches = backend.matching_rule_indexes(question)
        # ``to_sql`` raises when nothing matches, so reaching this line
        # guarantees at least one index: the first is the winner, the rest are
        # shadowed by it.
        explanation.matched_rule = matches[0]
        explanation.matched_pattern = backend.rule_pattern(matches[0])
        explanation.shadowed_rules = [
            (index, backend.rule_pattern(index)) for index in matches[1:]
        ]

    try:
        runner.validate(sql)
    except runner.UnsafeQueryError as exc:
        explanation.safety_error = str(exc)

    return explanation
