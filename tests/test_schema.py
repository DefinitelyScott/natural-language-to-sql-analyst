"""Tests for schema introspection and the rendered prompt context."""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Iterator

import pytest

from nl2sql import schema

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    """An in-memory database with one categorical and one free-text column."""
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE regions (
            id INTEGER PRIMARY KEY,
            code TEXT NOT NULL,
            note TEXT
        );
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            region_id INTEGER NOT NULL,
            channel TEXT NOT NULL,
            amount REAL NOT NULL,
            FOREIGN KEY (region_id) REFERENCES regions(id)
        );
        """
    )
    connection.executemany(
        "INSERT INTO regions VALUES (?,?,?)",
        [(1, "North", "a" * 60), (2, "South", "b" * 60)],
    )
    connection.executemany(
        "INSERT INTO sales VALUES (?,?,?,?)",
        [(1, 1, "web", 10.0), (2, 2, "retail", 20.0), (3, 1, "web", 30.0)],
    )
    try:
        yield connection
    finally:
        connection.close()


# --------------------------------------------------------------------------- #
# column_values
# --------------------------------------------------------------------------- #
def test_column_values_returns_sorted_distinct_values(conn) -> None:
    assert schema.column_values(conn, "sales", "channel") == ["retail", "web"]


def test_column_values_is_none_above_the_cap(conn) -> None:
    """Over the cap means "not a category", reported as None rather than a list."""
    assert schema.column_values(conn, "sales", "channel", max_distinct=1) is None
    # Exactly at the cap is still categorical -- the boundary is inclusive.
    assert schema.column_values(conn, "sales", "channel", max_distinct=2) == [
        "retail",
        "web",
    ]


def test_column_values_is_none_for_long_values(conn) -> None:
    """Few distinct values is not enough; free text is rejected on length too."""
    assert schema.column_values(conn, "regions", "note") is None


def test_column_values_excludes_nulls(conn) -> None:
    conn.execute("INSERT INTO regions VALUES (3, 'East', NULL)")
    assert schema.column_values(conn, "regions", "code") == ["East", "North", "South"]


def test_column_values_of_empty_table_is_empty_not_none(conn) -> None:
    """An empty categorical column is distinguishable from an uncategorizable one."""
    conn.execute("DELETE FROM sales")
    assert schema.column_values(conn, "sales", "channel") == []


def test_column_values_disabled_by_zero_cap(conn) -> None:
    assert schema.column_values(conn, "sales", "channel", max_distinct=0) is None


# --------------------------------------------------------------------------- #
# sample_categorical_values
# --------------------------------------------------------------------------- #
def test_sampling_skips_keys_and_non_text_columns(conn) -> None:
    sampled = schema.sample_categorical_values(conn, schema.introspect(conn))

    assert sampled == {
        ("regions", "code"): ["North", "South"],
        ("sales", "channel"): ["retail", "web"],
    }
    # id is a primary key, region_id a foreign key, amount is REAL, and note is
    # free text -- each excluded for a different reason.
    for key in (("sales", "id"), ("sales", "region_id"), ("sales", "amount")):
        assert key not in sampled
    assert ("regions", "note") not in sampled


def test_sampling_omits_columns_with_no_values(conn) -> None:
    """An empty column contributes nothing rather than an empty hint."""
    conn.execute("DELETE FROM sales")
    sampled = schema.sample_categorical_values(conn, schema.introspect(conn))
    assert ("sales", "channel") not in sampled


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def test_render_without_values_is_structure_only(conn) -> None:
    text = schema.render(schema.introspect(conn))
    assert "-- one of:" not in text
    assert "TABLE sales (" in text


def test_render_annotates_categorical_columns(conn) -> None:
    tables = schema.introspect(conn)
    text = schema.render(tables, schema.sample_categorical_values(conn, tables))
    assert "-- one of: 'retail', 'web'" in text
    # A column that is not categorical is left as a bare declaration.
    assert "\n  amount REAL\n" in text


def test_render_keeps_the_comma_before_the_comment(conn) -> None:
    """The comma separates declarations; a comment after it would swallow it."""
    tables = schema.introspect(conn)
    text = schema.render(tables, schema.sample_categorical_values(conn, tables))
    assert "channel TEXT,  -- one of: 'retail', 'web'" in text
    # The final column of a table takes no trailing comma.
    assert "\n  note TEXT\n" in text


def test_render_escapes_single_quotes_in_values(conn) -> None:
    conn.execute("UPDATE sales SET channel = 'in''store' WHERE channel = 'retail'")
    tables = schema.introspect(conn)
    text = schema.render(tables, schema.sample_categorical_values(conn, tables))
    assert "'in''store'" in text


# --------------------------------------------------------------------------- #
# Against the sample database
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_introspect_finds_tables() -> None:
    text = schema.schema_context(DB)
    for table in ("customers", "products", "orders", "order_items"):
        assert f"TABLE {table}" in text


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_schema_includes_foreign_keys() -> None:
    text = schema.schema_context(DB)
    assert "FOREIGN KEY" in text
    assert "REFERENCES customers(id)" in text


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_schema_context_lists_the_sample_dimensions() -> None:
    """The sample DB's real dimensions reach the prompt as exact literals.

    These are the two columns a question like "revenue in the North region" or
    "revenue for Electronics" has to filter on, and the exact spelling and
    capitalisation is the part a model cannot infer from the column name.
    """
    text = schema.schema_context(DB)
    assert "region TEXT,  -- one of: 'East', 'North', 'South', 'West'" in text
    assert (
        "category TEXT,  -- one of: 'Electronics', 'Fitness', 'Home', 'Office'" in text
    )


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_schema_context_omits_high_cardinality_columns() -> None:
    """Dates and customer names are row data, not categories -- they stay out.

    Both are TEXT columns that are neither a primary nor a foreign key, so the
    cardinality cap is the only thing keeping 900 order dates and 120 names out
    of every prompt.
    """
    text = schema.schema_context(DB)
    # The columns are still declared; only their contents are withheld.
    assert "order_date TEXT" in text and "signup_date TEXT" in text
    assert not re.search(r"'\d{4}-\d{2}-\d{2}'", text), "a date literal leaked in"

    customers_block = text.split("TABLE customers (", 1)[1].split(")", 1)[0]
    name_line = next(
        line for line in customers_block.splitlines() if line.strip().startswith("name ")
    )
    assert "-- one of:" not in name_line


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_schema_context_can_be_reduced_to_structure() -> None:
    """``max_distinct=0`` is the escape hatch behind the CLI's --no-values."""
    structural = schema.schema_context(DB, max_distinct=0)
    assert "-- one of:" not in structural
    assert "TABLE customers (" in structural


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_table_row_counts() -> None:
    counts = dict(schema.table_row_counts(DB))
    # The sample DB build is seeded, so these are exact.
    assert counts["customers"] == 120
    assert counts["products"] == 12
    assert counts["orders"] == 900
    assert counts["order_items"] > 0


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_no_order_predates_its_customers_signup() -> None:
    """No order may be attributed to a customer who had not yet signed up.

    The generator used to pick each order's customer uniformly from all 120,
    which put 42% of first orders before the customer's own ``signup_date``.
    Nothing in the schema forbids that -- there is no CHECK constraint spanning
    the two tables -- and no query in the catalog noticed, but it made every
    signup-relative metric meaningless: time to first order came out negative
    for those customers. This test is the constraint the schema cannot express.
    """
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        offenders = conn.execute(
            """
            SELECT COUNT(*)
            FROM orders o
            JOIN customers c ON c.id = o.customer_id
            WHERE o.order_date < c.signup_date
            """
        ).fetchone()[0]
    finally:
        conn.close()
    assert offenders == 0
