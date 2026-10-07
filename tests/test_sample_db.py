"""Tests for the sample-database generator, ``scripts/build_sample_db.py``.

Every number this project reports — execution accuracy, the gold answers, the
row counts quoted in the docs — is computed against the database this script
writes, yet until now the script itself was exercised only indirectly: the rest
of the suite reads ``data/store.db`` and trusts whatever is in it. These tests
check the generator's own promises directly:

* **Deterministic.** The module docstring says the database, and therefore the
  evaluation results, are reproducible for anyone who clones the repo. That is
  a claim about the *contents*, so it is checked by comparing every row of two
  independent builds, not by comparing file bytes (which would also pin SQLite's
  page layout, something the claim never made).
* **Internally consistent.** Foreign keys resolve, dates fall in the windows the
  generator draws from, and each order line is priced at its product's list
  price. A rule's SQL can be correct and still return a wrong-looking answer if
  the data under it breaks one of these.
* **No order predates its customer's signup.** The generator constrains this on
  purpose (see the comment in ``build``). ``test_offline_backend`` already
  checks that no signup cohort has a *negative average* time to first order,
  but an average can stay non-negative while individual orders violate the
  rule, so it is checked here per order.

Each test builds into ``tmp_path`` rather than reading ``data/store.db``, so the
file runs without a prior build step and never races or overwrites the shared
database other tests read.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from scripts import build_sample_db

TABLES = ("customers", "products", "orders", "order_items")


def _build(path: Path) -> Path:
    """Build the sample database at ``path`` and return the path."""
    build_sample_db.build(str(path))
    return path


def _dump(path: Path) -> dict[str, list[tuple[object, ...]]]:
    """Return every row of every table, in primary-key order."""
    with closing(sqlite3.connect(path)) as conn:
        return {
            table: conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
            for table in TABLES
        }


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Iterator[sqlite3.Connection]:
    """A read-only connection to one freshly built sample database.

    Module-scoped because the invariant checks below only read, and building
    once keeps the file fast.
    """
    path = _build(tmp_path_factory.mktemp("sample") / "store.db")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        yield conn
    finally:
        conn.close()


def _scalar(conn: sqlite3.Connection, sql: str) -> object:
    """Run a query that returns one value and return it."""
    return conn.execute(sql).fetchone()[0]


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def test_two_builds_produce_identical_rows(tmp_path: Path) -> None:
    first = _dump(_build(tmp_path / "a.db"))
    second = _dump(_build(tmp_path / "b.db"))
    assert first == second


def test_rebuilding_replaces_rather_than_appends(tmp_path: Path) -> None:
    # `build` deletes an existing file first. Without that, a second run would
    # fail on duplicate primary keys at best, or — if the schema ever dropped
    # them — silently double every count.
    path = tmp_path / "store.db"
    once = _dump(_build(path))
    twice = _dump(_build(path))
    assert once == twice


def test_row_counts(db: sqlite3.Connection) -> None:
    # Pinned because the docs, the CLI test and every gold answer assume them;
    # a change here is a change to every reported result and should be
    # deliberate, not a side effect of touching the generator.
    counts = {table: _scalar(db, f"SELECT COUNT(*) FROM {table}") for table in TABLES}
    assert counts == {
        "customers": 120,
        "products": 12,
        "orders": 900,
        "order_items": 2295,
    }


# --------------------------------------------------------------------------- #
# Referential and domain integrity
# --------------------------------------------------------------------------- #
def test_every_foreign_key_resolves(db: sqlite3.Connection) -> None:
    # PRAGMA foreign_key_check reports violations even though enforcement is
    # off by default in SQLite — which is exactly why it needs checking.
    assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_categorical_columns_use_only_the_declared_values(db: sqlite3.Connection) -> None:
    regions = {row[0] for row in db.execute("SELECT DISTINCT region FROM customers")}
    categories = {row[0] for row in db.execute("SELECT DISTINCT category FROM products")}
    assert regions == set(build_sample_db.REGIONS)
    assert categories == set(build_sample_db.CATEGORIES)


def test_dates_fall_inside_the_generated_windows(db: sqlite3.Connection) -> None:
    # Signups: 2023-01-01 plus up to 700 days (2024-12-01). Orders: calendar 2024.
    # ISO-8601 text compares correctly as a string, which is what SQLite does.
    signup_min, signup_max = db.execute(
        "SELECT MIN(signup_date), MAX(signup_date) FROM customers"
    ).fetchone()
    order_min, order_max = db.execute(
        "SELECT MIN(order_date), MAX(order_date) FROM orders"
    ).fetchone()
    assert "2023-01-01" <= signup_min <= signup_max <= "2024-12-01"
    assert "2024-01-01" <= order_min <= order_max <= "2024-12-31"


def test_no_order_predates_its_customers_signup(db: sqlite3.Connection) -> None:
    early = _scalar(
        db,
        """
        SELECT COUNT(*)
        FROM orders AS o
        JOIN customers AS c ON c.id = o.customer_id
        WHERE o.order_date < c.signup_date
        """,
    )
    assert early == 0


def test_order_lines_are_priced_at_list_price(db: sqlite3.Connection) -> None:
    # The generator records the product's price on each line, so revenue from
    # `unit_price * quantity` and from `price * quantity` must agree. Rules use
    # the former; this keeps a future price-change feature from silently
    # splitting the two.
    mismatched = _scalar(
        db,
        """
        SELECT COUNT(*)
        FROM order_items AS oi
        JOIN products AS p ON p.id = oi.product_id
        WHERE oi.unit_price <> p.price
        """,
    )
    assert mismatched == 0


def test_quantities_and_basket_sizes_stay_in_range(db: sqlite3.Connection) -> None:
    qty_min, qty_max = db.execute(
        "SELECT MIN(quantity), MAX(quantity) FROM order_items"
    ).fetchone()
    lines_min, lines_max = db.execute(
        """
        SELECT MIN(n), MAX(n)
        FROM (SELECT COUNT(*) AS n FROM order_items GROUP BY order_id)
        """
    ).fetchone()
    assert (qty_min, qty_max) == (1, 3)
    assert (lines_min, lines_max) == (1, 4)
    # Every order has at least one line: an order with none would count toward
    # order totals while contributing nothing to revenue.
    assert _scalar(
        db, "SELECT COUNT(*) FROM orders WHERE id NOT IN (SELECT order_id FROM order_items)"
    ) == 0
