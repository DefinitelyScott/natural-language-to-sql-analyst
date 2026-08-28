"""Tests for the SQL repair loop in ``generator.answer_question``.

The loop only does anything when generated SQL *fails*, which the offline
backend never does — every rule is hand-written and pinned by the gold set. So
these tests drive it with stub backends whose SQL is chosen to fail in a
specific way, which also keeps them free of any network or API key.

The stubs deliberately implement the same two-method surface as ``LLMBackend``
rather than subclassing it: ``llm.RepairingBackend`` is a structural protocol,
so a stub that satisfies it here proves the same runtime check that selects the
real backend in production.
"""

from __future__ import annotations

import os

import pytest

from nl2sql import cli, generator, llm, runner

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")

needs_db = pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")

GOOD_SQL = "SELECT COUNT(*) AS customer_count FROM customers"
# References a column that does not exist, so SQLite rejects it at prepare time
# — the single most common way an LLM's text-to-SQL output fails in practice.
BAD_COLUMN_SQL = "SELECT nonexistent_column FROM customers"
# Rejected by the string-level validator before it ever reaches SQLite.
UNSAFE_SQL = "DROP TABLE customers"
# Passes every guardrail and would run for hours: the shape a deadline exists
# to stop. See ``tests/test_runner.py`` for the layer itself.
TIMEOUT_SQL = (
    "WITH RECURSIVE counter(n) AS ("
    "  SELECT 1 UNION ALL SELECT n + 1 FROM counter WHERE n < 100000000"
    ") SELECT COUNT(*) FROM counter"
)


class FixedBackend:
    """A backend that always returns one query and cannot repair it."""

    def __init__(self, sql: str) -> None:
        self._sql = sql

    def to_sql(self, question: str, schema: str) -> str:  # noqa: ARG002
        return self._sql


class ScriptedBackend:
    """A repairing backend that returns a predetermined sequence of queries.

    ``to_sql`` yields the first query and each ``repair`` yields the next, so a
    test states the exact attempt sequence it wants to exercise. Every repair
    call is recorded, which is what lets a test assert both the retry *budget*
    (how many calls happened) and the *content* handed over (that the failing
    SQL and the engine's own error text reached the backend, since those are the
    only things that make a second attempt better-informed than the first).
    """

    def __init__(self, *sqls: str) -> None:
        if not sqls:
            raise ValueError("ScriptedBackend needs at least one query")
        self._sqls = list(sqls)
        self.repair_calls: list[tuple[str, str, str, str]] = []

    def to_sql(self, question: str, schema: str) -> str:  # noqa: ARG002
        return self._sqls[0]

    def repair(self, question: str, schema: str, sql: str, error: str) -> str:
        self.repair_calls.append((question, schema, sql, error))
        index = len(self.repair_calls)
        if index >= len(self._sqls):
            raise AssertionError(
                f"repair called {index} time(s) but only "
                f"{len(self._sqls) - 1} rewrite(s) were scripted"
            )
        return self._sqls[index]


@pytest.fixture
def use_backend(monkeypatch):
    """Return a helper that installs a stub as the backend ``_resolve`` builds."""

    def install(backend):
        monkeypatch.setattr(llm, "get_backend", lambda use_llm: backend)  # noqa: ARG005
        return backend

    return install


# --------------------------------------------------------------------------- #
# Protocol membership
# --------------------------------------------------------------------------- #
def test_offline_backend_is_not_repairable():
    """The offline backend must not opt in: its SQL is fixed, so a retry is dead time."""
    assert not isinstance(llm.OfflineBackend(), llm.RepairingBackend)


def test_llm_backend_exposes_a_repair_method():
    """Checked on the class so no API key or client construction is needed."""
    assert callable(getattr(llm.LLMBackend, "repair", None))


def test_scripted_backend_satisfies_the_protocol():
    assert isinstance(ScriptedBackend(GOOD_SQL), llm.RepairingBackend)
    assert not isinstance(FixedBackend(GOOD_SQL), llm.RepairingBackend)


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
@needs_db
def test_working_sql_is_not_repaired(use_backend):
    """The common path must be untouched: no repair calls, no recorded attempts."""
    backend = use_backend(ScriptedBackend(GOOD_SQL))

    answer = generator.answer_question(DB, "How many customers do we have?")

    assert answer.repairs == []
    assert backend.repair_calls == []
    assert answer.result.rows == [(120,)]


@needs_db
def test_failed_sql_is_repaired_and_the_retry_answers(use_backend):
    use_backend(ScriptedBackend(BAD_COLUMN_SQL, GOOD_SQL))

    answer = generator.answer_question(DB, "How many customers do we have?")

    assert answer.sql == GOOD_SQL
    assert answer.result.rows == [(120,)]
    assert len(answer.repairs) == 1
    assert answer.repairs[0].sql == BAD_COLUMN_SQL
    assert "nonexistent_column" in answer.repairs[0].error


@needs_db
def test_repair_is_given_the_failed_sql_and_the_engine_error(use_backend):
    """The two inputs that make a rewrite better-informed than a re-ask."""
    backend = use_backend(ScriptedBackend(BAD_COLUMN_SQL, GOOD_SQL))

    generator.answer_question(DB, "How many customers do we have?")

    assert len(backend.repair_calls) == 1
    question, schema_text, sql, error = backend.repair_calls[0]
    assert question == "How many customers do we have?"
    assert "customers" in schema_text
    assert sql == BAD_COLUMN_SQL
    assert "nonexistent_column" in error


@needs_db
def test_unsafe_sql_is_repairable_too(use_backend):
    """A guardrail rejection is an ordinary mistake the backend can be told about."""
    backend = use_backend(ScriptedBackend(UNSAFE_SQL, GOOD_SQL))

    answer = generator.answer_question(DB, "How many customers do we have?")

    assert answer.result.rows == [(120,)]
    assert len(backend.repair_calls) == 1
    assert "UnsafeQueryError" in answer.repairs[0].error


@needs_db
def test_a_timed_out_query_is_not_repaired(use_backend):
    """The one execution failure the loop must decline to retry.

    A repair is worth making when the engine's error names something to fix. A
    deadline names nothing — the query may be perfectly correct and simply
    expensive — so a rewrite would be a guess, and the guess would cost another
    full timeout before failing the same way.

    The backend here *is* repairable and has a working query scripted behind
    the slow one, so if the loop retried at all this test would pass silently
    with a valid answer. Asserting that ``repair`` was never called is what
    pins the behaviour; ``QueryTimeoutError`` reaching the caller unchanged is
    what proves the failure was not swallowed and relabelled on the way out.
    """
    backend = use_backend(ScriptedBackend(TIMEOUT_SQL, GOOD_SQL))

    with pytest.raises(runner.QueryTimeoutError):
        generator.answer_question(DB, "How many customers do we have?", timeout_ms=5)

    assert backend.repair_calls == []


@needs_db
def test_a_repair_cannot_smuggle_unsafe_sql_past_the_guardrails(use_backend):
    """The rewrite is validated exactly like the first attempt.

    This is the property that makes the loop safe to have at all: repairing
    widens *what is attempted*, never what is permitted.
    """
    use_backend(ScriptedBackend(BAD_COLUMN_SQL, UNSAFE_SQL))

    with pytest.raises(generator.QueryFailedError) as excinfo:
        generator.answer_question(DB, "How many customers do we have?")

    assert "UnsafeQueryError" in str(excinfo.value)


@needs_db
def test_repair_budget_is_respected(use_backend):
    """One repair, then give up — however many further rewrites were available."""
    backend = use_backend(
        ScriptedBackend(BAD_COLUMN_SQL, BAD_COLUMN_SQL, GOOD_SQL)
    )

    with pytest.raises(generator.QueryFailedError):
        generator.answer_question(DB, "How many customers do we have?")

    assert len(backend.repair_calls) == generator.MAX_REPAIR_ATTEMPTS == 1


@needs_db
def test_failure_message_lists_every_attempt(use_backend):
    """The final error alone would hide the query the backend originally wrote."""
    use_backend(ScriptedBackend(BAD_COLUMN_SQL, "SELECT also_missing FROM orders"))

    with pytest.raises(generator.QueryFailedError) as excinfo:
        generator.answer_question(DB, "How many customers do we have?")

    message = str(excinfo.value)
    assert "1 repair attempt" in message
    assert BAD_COLUMN_SQL in message
    assert "also_missing" in message


@needs_db
def test_non_repairing_backend_fails_immediately(use_backend):
    """No repair method means no retry — and a clean error rather than a traceback."""
    use_backend(FixedBackend(BAD_COLUMN_SQL))

    with pytest.raises(generator.QueryFailedError) as excinfo:
        generator.answer_question(DB, "How many customers do we have?")

    message = str(excinfo.value)
    assert "repair attempt" not in message
    assert BAD_COLUMN_SQL in message


@needs_db
def test_explain_never_repairs(use_backend):
    """A dry run does not execute, so there is no error to repair from."""
    backend = use_backend(ScriptedBackend(BAD_COLUMN_SQL, GOOD_SQL))

    explanation = generator.explain_question(DB, "How many customers do we have?")

    assert explanation.sql == BAD_COLUMN_SQL
    assert backend.repair_calls == []


# --------------------------------------------------------------------------- #
# CLI disclosure
# --------------------------------------------------------------------------- #
@needs_db
def test_cli_reports_a_repair_on_stderr(use_backend, capsys):
    """A repaired run must not read as a clean one — but must stay pipeable."""
    use_backend(ScriptedBackend(BAD_COLUMN_SQL, GOOD_SQL))

    assert cli.main(["ask", "How many customers do we have?", "--db", DB]) == 0

    captured = capsys.readouterr()
    assert "was repaired" in captured.err
    # The disclosure is a diagnostic and must stay out of the data stream.
    assert "repaired" not in captured.out


@needs_db
def test_cli_reports_an_unrepairable_failure_as_an_error(use_backend, capsys):
    use_backend(FixedBackend(BAD_COLUMN_SQL))

    assert cli.main(["ask", "How many customers do we have?", "--db", DB]) == 1

    assert "Error:" in capsys.readouterr().err
