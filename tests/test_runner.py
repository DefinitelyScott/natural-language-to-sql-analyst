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


# --------------------------------------------------------------------------- #
# Engine-level authorizer (second guardrail layer)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "action",
    [
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    ],
)
def test_authorizer_allows_read_only_actions(action):
    assert runner._authorizer(action, None, None, None, None) == sqlite3.SQLITE_OK


@pytest.mark.parametrize(
    "action",
    [
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_PRAGMA,
        sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_CREATE_TABLE,
    ],
)
def test_authorizer_denies_everything_else(action):
    assert runner._authorizer(action, None, None, None, None) == sqlite3.SQLITE_DENY


def test_authorizer_catches_what_the_validator_misses(monkeypatch):
    """The two guardrail layers must be independent.

    ``run`` calls :func:`runner.validate` first, so under normal operation the
    string check refuses an ``ATTACH`` before the engine ever sees it — which
    also means the normal path can never *demonstrate* that the second layer
    works. Disabling the first layer is the only way to test the second one in
    isolation: with ``validate`` stubbed out to wave everything through, an
    ``ATTACH`` (which would open another file on disk) must still be refused,
    now by the authorizer at statement-compile time.
    """
    if not os.path.exists(DB):
        pytest.skip("sample DB not built")
    monkeypatch.setattr(runner, "validate", lambda sql: sql)
    with pytest.raises(UnsafeQueryError, match="engine-level authorizer"):
        runner.run(DB, "ATTACH ':memory:' AS other")


def test_authorizer_still_allows_real_analytics_queries():
    """Aggregates, CTEs, and window functions all pass the allowlist.

    This is the counterpart to the denial tests: an allowlist that is too
    narrow would break legitimate queries, and this query deliberately
    exercises every allowed action code at once — SELECT, table/column reads,
    function calls (strftime, SUM, a window function), and a CTE.
    """
    if not os.path.exists(DB):
        pytest.skip("sample DB not built")
    res = runner.run(
        DB,
        """
        WITH monthly AS (
            SELECT strftime('%Y-%m', o.order_date) AS month,
                   SUM(oi.quantity * oi.unit_price) AS revenue
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.id
            GROUP BY month
        )
        SELECT month, revenue, SUM(revenue) OVER (ORDER BY month) AS running
        FROM monthly
        ORDER BY month
        """,
    )
    assert res.columns == ["month", "revenue", "running"]
    assert len(res) > 0
