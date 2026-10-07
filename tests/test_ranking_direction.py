"""Leaderboard rules must answer the end of the ranking they were asked about.

Two broad matchers used to answer inverted questions with the opposite ranking:
"bottom 5 products by revenue" reached the top-products rule through
``products?.*by revenue``, and "which customers spent the least?" reached the
top-customers rule through ``customers?.*spent``. Each came back correctly
labelled, plausible, and backwards. These tests pin both halves of the fix: the
``_DESCENDING_ONLY`` guard that makes the descending rules decline, and the
bottom-products rule that answers the one inverted question the catalog does
implement.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

from nl2sql import generator
from nl2sql.llm import NoRuleMatchError, OfflineBackend

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")


def _sql(question: str) -> str:
    return OfflineBackend().to_sql(question, schema="")


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "question",
    [
        "What are the bottom 5 products by revenue?",
        "Which products generated the least sales?",
        "Which products have the lowest revenue?",
        "What are the lowest-revenue products?",
        "Show the worst-selling products by sales.",
    ],
)
def test_low_end_phrasings_reach_the_bottom_products_rule(question: str) -> None:
    sql = _sql(question)
    assert "LEFT JOIN order_items" in sql
    assert "ORDER BY revenue ASC" in sql


@pytest.mark.parametrize(
    "question",
    [
        "What are the top 5 products by revenue?",
        "Rank products by revenue.",
        "Which 5 customers spent the most?",
        "Who are our five biggest customers?",
    ],
)
def test_high_end_phrasings_still_reach_the_descending_rules(question: str) -> None:
    # The guard must only remove the inverted questions, not the ones these
    # rules were written for.
    assert "DESC" in _sql(question)


@pytest.mark.parametrize(
    "question",
    [
        # No bottom-customers rule exists, so this must be declined rather
        # than answered with the biggest spenders.
        "Which customers spent the least?",
        "Which 5 customers spent the least?",
        # Thresholds, not rankings: "at least" is not "least".
        "Which customers spent at least $500?",
        "Which products had at least $1,000 in sales?",
        # The low end by units is not implemented; the revenue ranking must
        # not stand in for it.
        "What are the bottom 5 products by units sold?",
        "What is the least popular product?",
    ],
)
def test_unimplemented_inverted_questions_are_declined(question: str) -> None:
    with pytest.raises(NoRuleMatchError):
        _sql(question)


# --------------------------------------------------------------------------- #
# Semantics
# --------------------------------------------------------------------------- #


def _tiny_store() -> sqlite3.Connection:
    """Three products: one strong, one weak, one that has never sold."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY, order_id INTEGER, product_id INTEGER,
            quantity INTEGER, unit_price REAL
        );
        INSERT INTO products VALUES
            (1, 'Strong', 'A', 10.0),
            (2, 'Weak',   'A',  2.5),
            (3, 'Unsold', 'B', 99.0);
        INSERT INTO order_items VALUES
            (1, 1, 1, 3, 10.0),
            (2, 2, 2, 1,  2.5);
        """
    )
    return conn


def test_a_product_that_never_sold_ranks_first_with_zero_revenue() -> None:
    # The reason this rule is not the top-N rule with ASC: an inner join would
    # drop "Unsold" entirely and the list would start at "Weak".
    conn = _tiny_store()
    try:
        rows = conn.execute(_sql("What are the bottom 5 products by revenue?")).fetchall()
    finally:
        conn.close()
    assert rows == [("Unsold", 0.0), ("Weak", 2.5), ("Strong", 30.0)]


def test_revenue_ties_break_on_product_id() -> None:
    conn = _tiny_store()
    try:
        conn.execute("INSERT INTO products VALUES (0, 'AlsoUnsold', 'B', 5.0)")
        rows = conn.execute(_sql("Which products have the lowest revenue?")).fetchall()
    finally:
        conn.close()
    assert [name for name, _ in rows[:2]] == ["AlsoUnsold", "Unsold"]


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_bottom_products_match_a_direct_recomputation() -> None:
    ans = generator.answer_question(DB, "What are the bottom 5 products by revenue?")
    assert ans.result.columns == ["name", "revenue"]

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        products = conn.execute("SELECT id, name FROM products").fetchall()
        lines = conn.execute(
            "SELECT product_id, quantity, unit_price FROM order_items"
        ).fetchall()
    finally:
        conn.close()

    revenue = {pid: 0.0 for pid, _ in products}
    for pid, qty, price in lines:
        revenue[pid] += qty * price
    expected = sorted(products, key=lambda p: (round(revenue[p[0]], 2), p[0]))[:5]

    assert [row[0] for row in ans.result.rows] == [name for _, name in expected]
    for (_, rev), (pid, _) in zip(ans.result.rows, expected, strict=True):
        assert rev == pytest.approx(revenue[pid], abs=0.005)
