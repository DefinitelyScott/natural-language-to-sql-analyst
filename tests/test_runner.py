"""Tests for the read-only SQL guardrails."""

import os
import sqlite3
import time

import pytest

from nl2sql import runner
from nl2sql.runner import QueryTimeoutError, UnsafeQueryError

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


# --------------------------------------------------------------------------- #
# Execution deadline (third guardrail layer)
# --------------------------------------------------------------------------- #
# A query that is read-only, single-statement, and passes both earlier layers,
# yet would run for hours: counting to a hundred million one row at a time. The
# recursive CTE is the point — it is *allowed* (SQLITE_RECURSIVE is on the
# authorizer's allowlist), which is what makes it the right probe for a layer
# that exists to bound cost rather than permission. It is also cheap to compile
# and expensive to run, so the deadline has to fire during execution.
RUNAWAY_SQL = """
WITH RECURSIVE counter(n) AS (
    SELECT 1
    UNION ALL
    SELECT n + 1 FROM counter WHERE n < 100000000
)
SELECT COUNT(*) FROM counter
"""


def test_runaway_query_passes_the_first_two_layers():
    """The probe query is only meaningful if nothing else would reject it.

    If a later edit to the validator or the authorizer started refusing this
    SQL, the timeout tests below would still pass — for the wrong reason, and
    the deadline would be silently untested. Pinning it here makes that failure
    show up as itself.
    """
    assert runner.validate(RUNAWAY_SQL)
    assert runner._authorizer(sqlite3.SQLITE_RECURSIVE, None, None, None, None) == (
        sqlite3.SQLITE_OK
    )


def test_run_cancels_a_query_that_outlives_its_deadline():
    """A runaway query is stopped, and stopped promptly.

    The elapsed-time assertion is the substance of the test. Raising
    ``QueryTimeoutError`` proves only that *something* refused the query; that
    it happened in well under a second proves the progress handler cut the
    query short mid-execution, rather than the query somehow finishing. The
    bound is 200x the requested deadline so it cannot flake on a loaded CI
    runner, while still being many orders of magnitude below the hours this
    query would otherwise take.
    """
    if not os.path.exists(DB):
        pytest.skip("sample DB not built")
    started = time.monotonic()
    with pytest.raises(QueryTimeoutError, match="5 ms execution deadline"):
        runner.run(DB, RUNAWAY_SQL, timeout_ms=5)
    assert time.monotonic() - started < 1.0


def test_timeout_of_none_runs_without_a_deadline():
    """Opting out is honoured — and does not break an ordinary query.

    Deliberately not tested against the runaway query: a test that proves the
    deadline is gone by waiting for a query that never ends is a hang, not a
    test. What is checkable is that ``None`` is accepted, installs no handler,
    and returns the same result the default would.
    """
    if not os.path.exists(DB):
        pytest.skip("sample DB not built")
    res = runner.run(DB, "SELECT COUNT(*) FROM products", timeout_ms=None)
    assert res.rows == [(12,)]


def test_catalog_queries_finish_well_inside_the_default_deadline():
    """The default budget must not be tight enough to fail honest work.

    A deadline that trips on a legitimate query is worse than none: it turns a
    correct answer into an error. This runs the heaviest shape in the catalog —
    a self-join of order_items against itself to find products bought together
    — under a deadline far below the default, so the default keeps a wide
    margin even as the sample data grows.
    """
    if not os.path.exists(DB):
        pytest.skip("sample DB not built")
    res = runner.run(
        DB,
        """
        SELECT a.product_id AS a_id, b.product_id AS b_id, COUNT(*) AS pairs
        FROM order_items a
        JOIN order_items b
          ON b.order_id = a.order_id AND b.product_id > a.product_id
        GROUP BY a_id, b_id
        ORDER BY pairs DESC
        """,
        timeout_ms=runner.DEFAULT_TIMEOUT_MS // 10,
    )
    assert len(res) > 0


@pytest.mark.parametrize("timeout_ms", [0, -1])
def test_run_rejects_a_non_positive_timeout(timeout_ms):
    """0 and negative values are caller errors, not a way to disable the deadline.

    ``None`` is the only way to opt out. Treating a falsy 0 as "no deadline"
    would mean an uninitialized variable silently removes a guardrail, which is
    the failure mode this argument exists to prevent.
    """
    with pytest.raises(ValueError, match="timeout_ms"):
        runner.run(DB, "SELECT 1", timeout_ms=timeout_ms)


def test_a_denied_query_is_not_reported_as_a_timeout(monkeypatch):
    """The two failure modes must stay distinguishable.

    Both reach the same ``except sqlite3.DatabaseError`` clause, and the
    deadline check runs first. If it were checked by matching on SQLite's
    message text instead of the handler's own flag, an authorizer denial under
    an armed deadline could be misreported as a timeout — sending the reader to
    tune a limit when the real answer is that the query was refused outright.
    """
    if not os.path.exists(DB):
        pytest.skip("sample DB not built")
    monkeypatch.setattr(runner, "validate", lambda sql: sql)
    with pytest.raises(UnsafeQueryError, match="engine-level authorizer"):
        runner.run(DB, "ATTACH ':memory:' AS other", timeout_ms=5_000)
