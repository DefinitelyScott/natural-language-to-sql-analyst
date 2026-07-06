"""Tests for the offline rule-based backend and end-to-end generation."""

import os

import pytest

from nl2sql import generator
from nl2sql.llm import OfflineBackend

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")


def test_offline_matches_known_question():
    sql = OfflineBackend().to_sql("Show revenue by category", schema="")
    assert "GROUP BY p.category" in sql
    assert sql.lower().startswith("select")


def test_offline_raises_on_unknown_question():
    with pytest.raises(ValueError):
        OfflineBackend().to_sql("What is the meaning of life?", schema="")


@pytest.mark.parametrize(
    "question, fragment",
    [
        ("What is the total revenue?", "SUM(oi.quantity * oi.unit_price)"),
        ("How many orders do we have?", "FROM orders"),
        ("How many products are in the catalog?", "FROM products"),
    ],
)
def test_offline_matches_aggregate_questions(question, fragment):
    sql = OfflineBackend().to_sql(question, schema="")
    assert sql.lower().startswith("select")
    assert fragment in sql


def test_specific_rule_wins_over_broad_order_count():
    # "How many orders ... last 30 days" must hit the date-scoped rule, not the
    # broad order-count rule that also contains the phrase "how many orders".
    sql = OfflineBackend().to_sql(
        "How many orders were placed in the last 30 days?", schema=""
    )
    assert "-30 day" in sql


def test_top_products_by_revenue_is_distinct_from_units():
    # "by revenue" must rank on quantity * unit_price, not units sold, and must
    # not be swallowed by the "best selling product" (units) rule.
    revenue_sql = OfflineBackend().to_sql(
        "What are the top 5 products by revenue?", schema=""
    )
    assert "SUM(oi.quantity * oi.unit_price)" in revenue_sql
    assert "LIMIT 5" in revenue_sql

    units_sql = OfflineBackend().to_sql("What is the best selling product?", schema="")
    assert "SUM(oi.quantity) AS units_sold" in units_sql
    assert revenue_sql != units_sql


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_top_products_by_revenue():
    ans = generator.answer_question(DB, "What are the top 5 products by revenue?")
    assert ans.result.columns == ["name", "revenue"]
    assert len(ans.result.rows) == 5
    revenues = [row[1] for row in ans.result.rows]
    assert revenues == sorted(revenues, reverse=True)


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_offline():
    ans = generator.answer_question(DB, "How many customers do we have?")
    assert ans.result.columns == ["customer_count"]
    assert ans.result.rows[0][0] == 120


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_order_count():
    ans = generator.answer_question(DB, "How many orders do we have?")
    assert ans.result.columns == ["order_count"]
    assert ans.result.rows[0][0] == 900


@pytest.mark.parametrize(
    "question",
    [
        "Show month-over-month revenue growth in 2024.",
        "What was the revenue growth each month?",
        "Break down monthly sales growth.",
    ],
)
def test_offline_matches_revenue_growth_phrasings(question):
    # The growth rule uses a window function; several natural phrasings should
    # all resolve to it.
    sql = OfflineBackend().to_sql(question, schema="")
    assert "LAG(revenue) OVER (ORDER BY month)" in sql
    assert sql.lower().startswith("with")


@pytest.mark.parametrize(
    "question",
    [
        "Show revenue by quarter in 2024.",
        "What was revenue per quarter?",
        "Break down quarterly sales.",
    ],
)
def test_offline_matches_revenue_by_quarter_phrasings(question):
    # Several quarterly phrasings should resolve to the quarter-bucketing rule,
    # which derives the quarter from the month rather than a (nonexistent)
    # SQLite quarter function.
    sql = OfflineBackend().to_sql(question, schema="")
    assert "GROUP BY quarter" in sql
    assert "/ 3" in sql  # month -> quarter integer arithmetic
    assert sql.lower().startswith("select")


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_revenue_by_quarter():
    # All sample orders fall in 2024, so every quarter is represented exactly
    # once, ordered Q1..Q4, and the quarter totals must sum to total revenue.
    ans = generator.answer_question(DB, "Show revenue by quarter in 2024.")
    assert ans.result.columns == ["quarter", "revenue"]
    quarters = [row[0] for row in ans.result.rows]
    assert quarters == ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"]

    quarter_total = round(sum(row[1] for row in ans.result.rows), 2)
    total = generator.answer_question(DB, "What is the total revenue?")
    assert quarter_total == total.result.rows[0][0]


@pytest.mark.parametrize(
    "question",
    [
        "Show revenue by day of week.",
        "What are sales by weekday?",
        "Break down revenue by day of the week.",
    ],
)
def test_offline_matches_revenue_by_weekday_phrasings(question):
    # Several day-of-week phrasings should resolve to the weekday-bucketing rule,
    # which maps SQLite's numeric strftime('%w') weekday to a readable name via a
    # CASE expression and orders by the numeric weekday (calendar order).
    sql = OfflineBackend().to_sql(question, schema="")
    assert "strftime('%w'" in sql
    assert "WHEN 1 THEN 'Monday'" in sql
    assert sql.lower().startswith("select")


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_revenue_by_weekday():
    # Orders span a full year, so all seven weekdays are present, returned in
    # Sunday..Saturday order, and the weekday totals must sum to total revenue.
    ans = generator.answer_question(DB, "Show revenue by day of week.")
    assert ans.result.columns == ["weekday", "revenue"]
    weekdays = [row[0] for row in ans.result.rows]
    assert weekdays == [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ]

    weekday_total = round(sum(row[1] for row in ans.result.rows), 2)
    total = generator.answer_question(DB, "What is the total revenue?")
    assert weekday_total == total.result.rows[0][0]


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_revenue_growth():
    # The first month has no prior month, so its revenue_change is NULL; every
    # later month reports the signed change from the month before it.
    ans = generator.answer_question(DB, "Show month-over-month revenue growth in 2024.")
    assert ans.result.columns == ["month", "revenue", "revenue_change"]
    assert len(ans.result.rows) == 12

    assert ans.result.rows[0][2] is None
    for (_, revenue, change), (_, prev_revenue, _) in zip(
        ans.result.rows[1:], ans.result.rows[:-1]
    ):
        assert change == round(revenue - prev_revenue, 2)


@pytest.mark.parametrize(
    "question",
    [
        "What percentage of revenue comes from each category?",
        "Show revenue share by category.",
        "What share of total sales does each category represent?",
        "Break down revenue by category as a percentage.",
    ],
)
def test_offline_matches_revenue_share_by_category_phrasings(question):
    # Several share/percentage phrasings resolve to the share-of-total rule,
    # which uses SUM(revenue) OVER () as the grand-total denominator.
    sql = OfflineBackend().to_sql(question, schema="")
    assert "SUM(revenue) OVER ()" in sql
    assert "pct_of_total" in sql
    assert sql.lower().startswith("with")


def test_revenue_share_does_not_shadow_plain_revenue_by_category():
    # A bare "revenue by category" question (no share/percentage word) must
    # still route to the simpler non-window rule, not the share rule.
    plain = OfflineBackend().to_sql("Show revenue by category", schema="")
    assert "OVER ()" not in plain
    assert "pct_of_total" not in plain


@pytest.mark.parametrize(
    "question",
    [
        "What is the average order value by region?",
        "Show average order value per region.",
        "Break down avg order value by region.",
    ],
)
def test_offline_matches_avg_order_value_by_region_phrasings(question):
    # Several phrasings should resolve to the region-scoped AOV rule, which
    # averages per-order totals (computed in a subquery) grouped by region.
    sql = OfflineBackend().to_sql(question, schema="")
    assert "GROUP BY c.region" in sql
    assert "AVG(order_total)" in sql
    assert sql.lower().startswith("select")


def test_avg_order_value_by_region_does_not_shadow_plain_aov():
    # A bare "average order value" question (no region word) must still route to
    # the simpler overall rule that returns a single average_order_value figure.
    plain = OfflineBackend().to_sql("What is the average order value?", schema="")
    assert "average_order_value" in plain
    assert "region" not in plain.lower()


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_avg_order_value_by_region():
    # Every seeded customer has one of four regions and all regions receive
    # orders, so the result has exactly four rows, ordered by descending AOV.
    # Each region's AOV must equal its own revenue divided by its order count,
    # confirming the average is taken over orders (not order-item rows).
    import sqlite3

    ans = generator.answer_question(DB, "What is the average order value by region?")
    assert ans.result.columns == ["region", "avg_order_value"]
    assert len(ans.result.rows) == 4

    values = [row[1] for row in ans.result.rows]
    assert values == sorted(values, reverse=True)

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        for region, avg_order_value in ans.result.rows:
            revenue, order_count = conn.execute(
                """
                SELECT SUM(oi.quantity * oi.unit_price), COUNT(DISTINCT o.id)
                FROM orders o
                JOIN order_items oi ON oi.order_id = o.id
                JOIN customers c ON c.id = o.customer_id
                WHERE c.region = ?
                """,
                (region,),
            ).fetchone()
            assert avg_order_value == round(revenue / order_count, 2)
    finally:
        conn.close()


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_revenue_share_by_category():
    # Every category is present exactly once; the percentages are computed from
    # the rounded per-category revenue over the grand total, so they sum to
    # ~100 (within rounding), rows are revenue-descending, and the category
    # revenues sum to total revenue.
    ans = generator.answer_question(
        DB, "What percentage of revenue comes from each category?"
    )
    assert ans.result.columns == ["category", "revenue", "pct_of_total"]

    revenues = [row[1] for row in ans.result.rows]
    assert revenues == sorted(revenues, reverse=True)

    pct_total = sum(row[2] for row in ans.result.rows)
    assert abs(pct_total - 100.0) < 0.5

    category_total = round(sum(revenues), 2)
    total = generator.answer_question(DB, "What is the total revenue?")
    assert category_total == total.result.rows[0][0]


@pytest.mark.parametrize(
    "question",
    [
        "Show revenue by region and category.",
        "Break down sales by category and region.",
        "What is revenue per region and category?",
    ],
)
def test_offline_matches_revenue_by_region_and_category_phrasings(question):
    # Questions naming BOTH region and category resolve to the two-dimensional
    # rule, which groups by region and category and joins all four tables so the
    # products (category) join is present alongside the customers (region) join.
    sql = OfflineBackend().to_sql(question, schema="")
    assert "GROUP BY c.region, p.category" in sql
    assert "JOIN products p ON p.id = oi.product_id" in sql
    assert sql.lower().startswith("select")


def test_region_and_category_does_not_shadow_single_dimension_rules():
    # A bare "revenue by region" (no category word) must still route to the
    # one-dimension region rule, and a bare "revenue by category" (no region
    # word) to the one-dimension category rule -- neither should hit the
    # two-dimensional region+category rule.
    by_region = OfflineBackend().to_sql("Show revenue by region", schema="")
    assert "GROUP BY c.region" in by_region
    assert "p.category" not in by_region

    by_category = OfflineBackend().to_sql("Show revenue by category", schema="")
    assert "GROUP BY p.category" in by_category
    assert "c.region" not in by_category


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_revenue_by_region_and_category():
    # Every (region, category) pair that has sales appears once; rows are grouped
    # by region then revenue-descending within each region, and the full grid of
    # region+category revenues must sum to total revenue (every order item is
    # counted exactly once).
    ans = generator.answer_question(DB, "Show revenue by region and category.")
    assert ans.result.columns == ["region", "category", "revenue"]

    # Rows are ordered by region ascending, then revenue descending within it.
    from itertools import groupby

    regions_in_order = [region for region, _ in groupby(r[0] for r in ans.result.rows)]
    assert regions_in_order == sorted(regions_in_order)
    for _, group in groupby(ans.result.rows, key=lambda r: r[0]):
        revenues = [row[2] for row in group]
        assert revenues == sorted(revenues, reverse=True)

    grid_total = round(sum(row[2] for row in ans.result.rows), 2)
    total = generator.answer_question(DB, "What is the total revenue?")
    assert grid_total == total.result.rows[0][0]
