"""Tests for the dry-run explain path (`generator.explain_question`, `nl2sql explain`).

Two properties matter here and neither is visible from the output alone:

1. **It really is a dry run.** The report must be produced without executing the
   query. That is asserted directly, by explaining a question while the database
   is open read-only anyway — instead the test patches ``runner.run`` to fail
   loudly if anything calls it.
2. **The routing it reports is the routing that happened.** The matched rule
   index must be the one ``to_sql`` actually used, and the shadowed list must be
   the remaining matches in catalog order. A diagnostic that drifts from the
   router is worse than none.
"""

from __future__ import annotations

import os

import pytest

from nl2sql import cli, generator, llm, runner

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")

needs_db = pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")

# Matches both the date-scoped "orders in the last 30 days" rule and the broad
# "how many orders" rule, so it exercises the shadowed-rule reporting. The same
# question anchors the routing contract in test_rule_catalog.py.
SHADOWING_QUESTION = "How many orders were placed in the last 30 days?"


@needs_db
def test_explain_does_not_execute_the_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """The explanation must be assembled without ever running the SQL."""

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("explain_question executed the query")

    monkeypatch.setattr(runner, "run", explode)
    exp = generator.explain_question(DB, "How many customers do we have?")
    assert "SELECT" in exp.sql.upper()


@needs_db
def test_explain_reports_the_rule_that_actually_answered() -> None:
    """The reported rule index must be the one the offline router used."""
    exp = generator.explain_question(DB, SHADOWING_QUESTION)
    backend = llm.OfflineBackend()
    matches = backend.matching_rule_indexes(SHADOWING_QUESTION)

    assert exp.backend == "offline"
    assert exp.matched_rule == matches[0]
    assert exp.matched_pattern == backend.rule_pattern(matches[0])
    assert exp.sql == backend.to_sql(SHADOWING_QUESTION, "")


@needs_db
def test_explain_lists_shadowed_rules_in_catalog_order() -> None:
    """Every later rule that also matched is reported, in registration order."""
    exp = generator.explain_question(DB, SHADOWING_QUESTION)
    matches = llm.OfflineBackend().matching_rule_indexes(SHADOWING_QUESTION)

    assert len(matches) > 1, (
        "fixture question no longer matches more than one rule; pick another"
    )
    assert [index for index, _ in exp.shadowed_rules] == matches[1:]


@needs_db
def test_explain_generates_the_same_sql_as_ask() -> None:
    """A dry run must not be a second, divergent code path."""
    question = "Show revenue by category"
    assert generator.explain_question(DB, question).sql == generator.generate_sql(
        DB, question
    )


@needs_db
def test_explain_reports_unsafe_sql_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejected SQL is data in the report, not an exception.

    A backend that emits a write statement is the case the safety validator
    exists for; ``explain`` has to surface both the statement and the reason,
    which it cannot do if validation escapes as an error.
    """

    class WritingBackend:
        def to_sql(self, question: str, schema: str) -> str:
            return "DELETE FROM orders"

    monkeypatch.setattr(llm, "get_backend", lambda use_llm: WritingBackend())
    exp = generator.explain_question(DB, "delete everything")

    assert not exp.is_safe
    assert exp.safety_error
    assert exp.sql == "DELETE FROM orders"
    # A non-offline backend has no rule catalog to report on.
    assert exp.matched_rule is None
    assert exp.shadowed_rules == []


@needs_db
def test_explain_command_prints_sql_and_rule(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["explain", SHADOWING_QUESTION, "--db", DB]) == 0
    out = capsys.readouterr().out
    assert "SQL (not executed):" in out
    assert "Matched offline rule #" in out
    assert "Also matched (shadowed" in out
    assert "passes the read-only validator" in out


@needs_db
def test_explain_command_exits_nonzero_on_unsafe_sql(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unsafe SQL is a finding: the SQL still prints, but the exit code flags it."""

    class WritingBackend:
        def to_sql(self, question: str, schema: str) -> str:
            return "DROP TABLE orders"

    monkeypatch.setattr(llm, "get_backend", lambda use_llm: WritingBackend())
    assert cli.main(["explain", "drop the orders table", "--db", DB]) == 1

    captured = capsys.readouterr()
    assert "DROP TABLE orders" in captured.out
    assert "REJECTED" in captured.err


@needs_db
def test_explain_command_unrecognized_question_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["explain", "what is the meaning of life?", "--db", DB]) == 1
    assert "Error:" in capsys.readouterr().err


def test_explain_command_missing_db(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["explain", "How many customers do we have?", "--db", "/nope.db"]) == 2
    assert "Database not found" in capsys.readouterr().err
