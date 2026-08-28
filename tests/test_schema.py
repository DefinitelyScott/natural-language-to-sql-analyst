"""Tests for schema introspection."""

import os

import pytest

from nl2sql import schema

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_introspect_finds_tables():
    text = schema.schema_context(DB)
    for table in ("customers", "products", "orders", "order_items"):
        assert f"TABLE {table}" in text


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_schema_includes_foreign_keys():
    text = schema.schema_context(DB)
    assert "FOREIGN KEY" in text
    assert "REFERENCES customers(id)" in text


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_table_row_counts():
    counts = dict(schema.table_row_counts(DB))
    # The sample DB build is seeded, so these are exact.
    assert counts["customers"] == 120
    assert counts["products"] == 12
    assert counts["orders"] == 900
    assert counts["order_items"] > 0


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_no_order_predates_its_customers_signup():
    """No order may be attributed to a customer who had not yet signed up.

    The generator used to pick each order's customer uniformly from all 120,
    which put 42% of first orders before the customer's own ``signup_date``.
    Nothing in the schema forbids that -- there is no CHECK constraint spanning
    the two tables -- and no query in the catalog noticed, but it made every
    signup-relative metric meaningless: time to first order came out negative
    for those customers. This test is the constraint the schema cannot express.
    """
    import sqlite3

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
