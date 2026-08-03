"""Tests for the read-only SQL guardrails."""

import os
import sqlite3

import pytest

from nl2sql import runner
from nl2sql.runner import UnsafeQueryError

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers",
        "DROP TABLE orders",
        "UPDATE products SET price = 0",
        "INSERT INTO customers VALUES (1, 'x', 'North', '2024-01-01')",
        "SELECT 1; SELECT 2",
        "PRAGMA table_info(customers)",
        "",
    ],
)
def test_validate_rejects_unsafe(sql):
    with pytest.raises(UnsafeQueryError):
        runner.validate(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM customers",
        "SELECT name FROM customers WHERE region = 'North';",
        "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
        "select * from products -- a comment",
    ],
)
def test_validate_allows_select(sql):
    assert runner.validate(sql)


def test_run_caps_rows():
    if not os.path.exists(DB):
        pytest.skip("sample DB not built")
    res = runner.run(DB, "SELECT id FROM order_items", max_rows=5)
    assert len(res) == 5
    assert res.truncated is True


def test_run_does_not_flag_a_complete_result():
    """A result well under the cap is complete, so nothing is flagged."""
    if not os.path.exists(DB):
        pytest.skip("sample DB not built")
    res = runner.run(DB, "SELECT id FROM products", max_rows=1000)
    # The sample DB is deterministic: 12 products.
    assert len(res) == 12
    assert res.truncated is False


def test_run_result_exactly_at_the_cap_is_not_truncated():
    """The boundary case: as many rows as the cap allows is still complete.

    This is what the extra probe row buys. Fetching exactly ``max_rows`` would
    make a result of precisely that length look identical to a truncated one,
    and warning about a complete export is as misleading as staying silent
    about a partial one.
    """
    if not os.path.exists(DB):
        pytest.skip("sample DB not built")
    res = runner.run(DB, "SELECT id FROM products", max_rows=12)
    assert len(res) == 12
    assert res.truncated is False


def test_run_is_read_only():
    if not os.path.exists(DB):
        pytest.skip("sample DB not built")
    with pytest.raises((UnsafeQueryError, sqlite3.OperationalError)):
        runner.run(DB, "DELETE FROM customers")
