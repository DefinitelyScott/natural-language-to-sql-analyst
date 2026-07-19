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
