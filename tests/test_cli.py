"""Tests for the command-line interface (`nl2sql.cli`)."""

import os

import pytest

from nl2sql import cli, llm
from nl2sql.runner import QueryTimeoutError

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")

needs_db = pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")


@needs_db
def test_schema_command_prints_tables(capsys):
    assert cli.main(["schema", "--db", DB]) == 0
    out = capsys.readouterr().out
    for table in ("customers", "products", "orders", "order_items"):
        assert f"TABLE {table}" in out
    # No counts unless requested.
    assert "Row counts:" not in out


@needs_db
def test_schema_command_with_counts(capsys):
    assert cli.main(["schema", "--db", DB, "--counts"]) == 0
    out = capsys.readouterr().out
    assert "Row counts:" in out
    # The sample DB is deterministic: 120 customers, 12 products, 900 orders.
    assert "customers" in out and "120" in out
    assert "900" in out


def test_schema_command_missing_db(capsys):
    assert cli.main(["schema", "--db", "/nonexistent/nope.db"]) == 2
    err = capsys.readouterr().err
    assert "Database not found" in err


@needs_db
def test_ask_command_table_output(capsys):
    assert cli.main(["ask", "How many customers do we have?", "--db", DB]) == 0
    out = capsys.readouterr().out
    assert "SQL:" in out
    assert "customer_count" in out


@needs_db
def test_ask_warns_on_stderr_when_the_row_cap_truncates(capsys):
    """A capped result must announce itself, in every output format.

    "Show revenue by category" returns one row per category (4 in the sample
    DB), so a cap of 2 reliably truncates it.
    """
    exit_code = cli.main(
        ["ask", "Show revenue by category", "--db", DB, "--max-rows", "2"]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "truncated to 2 rows" in captured.err
    # The warning is a diagnostic: it must not contaminate the data stream.
    assert "truncated" not in captured.out


@needs_db
def test_ask_csv_export_is_clean_and_unwarned_when_complete(capsys):
    """An export that fits under the cap is complete, so stderr carries no warning."""
    exit_code = cli.main(["ask", "Show revenue by category", "--db", DB, "--format", "csv"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "truncated" not in captured.err
    lines = captured.out.strip().splitlines()
    # Header + one row per category (4 categories in the sample DB).
    assert lines[0] == "category,revenue"
    assert len(lines) == 5


@needs_db
def test_ask_command_unrecognized_question_errors(capsys):
    assert cli.main(["ask", "what is the meaning of life?", "--db", DB]) == 1
    err = capsys.readouterr().err
    assert "Error:" in err


@needs_db
def test_ask_reports_a_timeout_with_a_way_out(capsys, monkeypatch):
    """A cancelled query must say what to change, not just that it failed.

    The offline catalog has nothing slow enough to trip a deadline — that is
    the point of the default — so the failure is injected at the boundary the
    CLI actually depends on: ``generator.answer_question`` raising. What is
    under test is the CLI's handling, and specifically that the clause sits
    ahead of the general ``RuntimeError`` one it would otherwise be swallowed
    by (``QueryTimeoutError`` subclasses it), which is the ordering an edit
    could quietly undo.
    """

    def timeout(*args, **kwargs):
        raise QueryTimeoutError("query cancelled after exceeding its 5 ms deadline")

    monkeypatch.setattr(cli.generator, "answer_question", timeout)

    assert cli.main(["ask", "How many customers do we have?", "--db", DB]) == 1
    err = capsys.readouterr().err
    assert "5 ms deadline" in err
    assert "--timeout-ms" in err


@needs_db
def test_ask_translates_timeout_ms_zero_into_no_deadline(monkeypatch):
    """``--timeout-ms 0`` is the CLI's spelling of the library's ``None``.

    ``runner.run`` rejects 0 outright, so if the CLI passed the flag through
    unchanged the "no deadline" case would fail with an argument error instead
    of running. This asserts on the value handed to the generator rather than
    on the query, because a query with no deadline has no observable behaviour
    to test against — only a value.
    """
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        raise llm.NoRuleMatchError("stop here; the argument is what matters")

    monkeypatch.setattr(cli.generator, "answer_question", capture)

    assert cli.main(["ask", "anything", "--db", DB, "--timeout-ms", "0"]) == 1
    assert seen["timeout_ms"] is None
