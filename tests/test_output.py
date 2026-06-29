"""Tests for the result output formatters and the CLI ``--format`` option."""

import json
import os

import pytest

from nl2sql import cli, output

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")


def test_format_table_basic():
    out = output.format_table(["a", "b"], [(1, 2), (3, 4)])
    lines = out.splitlines()
    assert lines[0].split() == ["a", "b"]
    assert "1" in out and "4" in out


def test_format_table_truncates_with_limit():
    rows = [(i,) for i in range(50)]
    out = output.format_table(["n"], rows, limit=10)
    assert "... (40 more rows)" in out
    # header + 10 rows + truncation line
    assert len(out.splitlines()) == 12


def test_format_table_renders_none_as_empty_cell():
    out = output.format_table(["a", "b"], [(1, None)])
    # the second cell is blank, so the row trims to just the first value
    assert out.splitlines()[1].strip() == "1"


def test_format_table_no_columns():
    assert output.format_table([], []) == "(no columns)"


def test_format_csv_quotes_values_with_commas():
    out = output.format_csv(["a", "b"], [(1, 2), (3, "x,y")])
    lines = out.splitlines()
    assert lines[0] == "a,b"
    assert lines[1] == "1,2"
    assert lines[2] == '3,"x,y"'  # comma-containing value is quoted by csv writer


def test_format_json_is_array_of_objects():
    out = output.format_json(["a", "b"], [(1, 2)])
    assert json.loads(out) == [{"a": 1, "b": 2}]


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_cli_json_output_is_clean_stdout(capsys):
    code = cli.main(
        ["ask", "How many customers do we have?", "--db", DB, "--format", "json"]
    )
    assert code == 0
    captured = capsys.readouterr()
    # stdout must be parseable JSON with nothing else mixed in
    assert json.loads(captured.out) == [{"customer_count": 120}]
    # the generated SQL is a diagnostic and goes to stderr
    assert "SQL:" in captured.err


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_cli_csv_output(capsys):
    code = cli.main(
        ["ask", "How many orders do we have?", "--db", DB, "--format", "csv"]
    )
    assert code == 0
    out_lines = capsys.readouterr().out.strip().splitlines()
    assert out_lines[0] == "order_count"
    assert out_lines[1] == "900"
