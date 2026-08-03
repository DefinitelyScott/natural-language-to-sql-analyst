"""Tests for the command-line interface (`nl2sql.cli`)."""

import os

import pytest

from nl2sql import cli

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
