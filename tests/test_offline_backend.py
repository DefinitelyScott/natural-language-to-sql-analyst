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


@pytest.mark.parametrize(
    "question",
    [
        "Who is the top-spending customer in each region?",
        "Show the highest spending customer per region.",
        "For each region, who is the top customer?",
        "Best customer in each region.",
    ],
)
def test_offline_matches_top_customer_per_region_phrasings(question):
    # Per-region top-spender phrasings resolve to the greatest-N-per-group rule,
    # which ranks customers inside each region with a PARTITION BY window and
    # keeps only the top rank per region.
    sql = OfflineBackend().to_sql(question, schema="")
    assert "PARTITION BY region" in sql
    assert "ROW_NUMBER()" in sql
    assert "WHERE rn = 1" in sql
    assert sql.lower().startswith("with")


def test_top_customer_per_region_does_not_shadow_global_top_spenders():
    # A per-region question hits the partitioned rule; the global "top 5
    # customers by spend" question (no region word) must still route to the
    # simpler LIMIT-5 rule, which does not partition or filter by rank.
    per_region = OfflineBackend().to_sql(
        "Who is the top-spending customer in each region?", schema=""
    )
    global_top = OfflineBackend().to_sql("Which 5 customers spent the most?", schema="")
    assert "PARTITION BY region" in per_region
    assert "PARTITION BY" not in global_top
    assert "LIMIT 5" in global_top
    assert per_region != global_top


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_top_customer_per_region():
    # Exactly one winner per region (the sample data has four regions), and each
    # winner must genuinely be the maximum spender within their own region.
    import sqlite3

    ans = generator.answer_question(
        DB, "Who is the top-spending customer in each region?"
    )
    assert ans.result.columns == ["region", "name", "total_spent"]

    regions = [row[0] for row in ans.result.rows]
    assert regions == sorted(regions)
    assert len(regions) == len(set(regions))  # one row per region

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        for region, _name, total_spent in ans.result.rows:
            region_max = conn.execute(
                """
                SELECT MAX(spend) FROM (
                    SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) AS spend
                    FROM customers c
                    JOIN orders o ON o.customer_id = c.id
                    JOIN order_items oi ON oi.order_id = o.id
                    WHERE c.region = ?
                    GROUP BY c.id
                )
                """,
                (region,),
            ).fetchone()[0]
            assert total_spent == region_max
    finally:
        conn.close()


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
        "How many unique customers placed an order each month in 2024?",
        "Show distinct customers per month in 2024.",
        "How many active buyers did we have each month in 2024?",
    ],
)
def test_offline_matches_unique_customers_per_month_phrasings(question):
    # Several "monthly active buyers" phrasings resolve to the distinct-customer
    # rule, which counts each customer once per month via COUNT(DISTINCT ...).
    sql = OfflineBackend().to_sql(question, schema="")
    assert "COUNT(DISTINCT o.customer_id)" in sql
    assert "GROUP BY month" in sql
    assert sql.lower().startswith("select")


def test_unique_customers_does_not_shadow_customer_count_or_signups():
    # A bare "how many customers" must still hit the simple total-count rule, and
    # "new customers by month" must still hit the signup rule -- neither should be
    # swallowed by the distinct-buyers-per-month rule (which needs a distinctness
    # word plus a customer/buyer word).
    count_sql = OfflineBackend().to_sql("How many customers do we have?", schema="")
    assert count_sql == "SELECT COUNT(*) AS customer_count FROM customers"

    signup_sql = OfflineBackend().to_sql(
        "How many new customers signed up by month in 2024?", schema=""
    )
    assert "FROM customers" in signup_sql
    assert "signup_date" in signup_sql
    assert "COUNT(DISTINCT" not in signup_sql


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_unique_customers_per_month():
    # All sample orders fall in 2024, so every month is present exactly once in
    # calendar order. Each month's distinct-buyer count must never exceed the raw
    # order count for that month (a customer can place several orders in a month
    # but is counted once), and it is bounded by the 120-customer base.
    import sqlite3

    ans = generator.answer_question(
        DB, "How many unique customers placed an order each month in 2024?"
    )
    assert ans.result.columns == ["month", "unique_customers"]
    months = [row[0] for row in ans.result.rows]
    assert months == sorted(months)
    assert len(months) == 12

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        for month, unique_customers in ans.result.rows:
            orders_that_month = conn.execute(
                "SELECT COUNT(*) FROM orders WHERE strftime('%Y-%m', order_date) = ?",
                (month,),
            ).fetchone()[0]
            assert 0 < unique_customers <= orders_that_month
            assert unique_customers <= 120
    finally:
        conn.close()


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


@pytest.mark.parametrize(
    "question",
    [
        "Show cumulative revenue by month in 2024.",
        "What is the running total of revenue by month?",
        "Give me the running sum of sales per month.",
        "Show revenue to date by month.",
    ],
)
def test_offline_matches_cumulative_revenue_phrasings(question):
    # Several cumulative/running-total phrasings resolve to the running-total
    # rule, which carries a progressive sum with an *ordered* ROWS window frame
    # (distinct from the empty OVER () frame used for category share).
    sql = OfflineBackend().to_sql(question, schema="")
    assert "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW" in sql
    assert "cumulative_revenue" in sql
    assert sql.lower().startswith("with")


def test_cumulative_revenue_does_not_shadow_total_or_growth_rules():
    # A bare "total revenue" question (no cumulative/running word) must still hit
    # the single-figure total rule, and a "revenue growth" question must still
    # hit the month-over-month LAG rule -- neither should be swallowed by the
    # running-total rule.
    total_sql = OfflineBackend().to_sql("What is the total revenue?", schema="")
    assert total_sql == (
        "SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue "
        "FROM order_items oi"
    )

    growth_sql = OfflineBackend().to_sql(
        "Show month-over-month revenue growth in 2024.", schema=""
    )
    assert "LAG(revenue) OVER (ORDER BY month)" in growth_sql
    assert "cumulative_revenue" not in growth_sql


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_cumulative_revenue():
    # All sample orders fall in 2024, so every month appears once in calendar
    # order. The running total must be monotonically non-decreasing (all monthly
    # revenues are positive), each row must equal the prior cumulative plus the
    # current month's revenue, and the final row must equal total 2024 revenue.
    ans = generator.answer_question(DB, "Show cumulative revenue by month in 2024.")
    assert ans.result.columns == ["month", "revenue", "cumulative_revenue"]
    assert len(ans.result.rows) == 12

    months = [row[0] for row in ans.result.rows]
    assert months == sorted(months)

    cumulatives = [row[2] for row in ans.result.rows]
    assert cumulatives == sorted(cumulatives)  # non-decreasing
    assert cumulatives[0] == ans.result.rows[0][1]  # first month: no prior sum
    for (_, revenue, cumulative), prev_cumulative in zip(
        ans.result.rows[1:], cumulatives[:-1]
    ):
        assert cumulative == round(prev_cumulative + revenue, 2)

    total = generator.answer_question(DB, "What is the total revenue?")
    assert cumulatives[-1] == total.result.rows[0][0]


@pytest.mark.parametrize(
    "question",
    [
        "What is the average number of items per order?",
        "average items per order",
        "avg units per order",
        "What is the mean number of units per order?",
        "What is the average basket size?",
    ],
)
def test_offline_matches_avg_units_per_order_phrasings(question):
    sql = OfflineBackend().to_sql(question, schema="")
    assert "avg_units_per_order" in sql
    # Basket size sums quantity (units), not quantity * unit_price (money).
    assert "SUM(oi.quantity)" in sql
    assert "unit_price" not in sql


def test_avg_units_per_order_distinct_from_order_value():
    # "items per order" (basket size, units) and "order value" (money) must route
    # to different rules: the units question must not be caught by the order-value
    # rule, and vice versa.
    units = OfflineBackend().to_sql("average items per order", schema="")
    value = OfflineBackend().to_sql("What is the average order value?", schema="")
    assert "avg_units_per_order" in units
    assert "average_order_value" in value
    assert units != value


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_avg_units_per_order():
    # A single figure: the mean over each order's total unit count. Verify it
    # equals total units sold divided by the number of orders, which confirms the
    # average is taken per order (after the subquery rollup) rather than over the
    # raw order-item rows.
    import sqlite3

    ans = generator.answer_question(DB, "What is the average number of items per order?")
    assert ans.result.columns == ["avg_units_per_order"]
    assert len(ans.result.rows) == 1
    avg_units = ans.result.rows[0][0]

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        total_units, order_count = conn.execute(
            "SELECT SUM(quantity), COUNT(DISTINCT order_id) FROM order_items"
        ).fetchone()
    finally:
        conn.close()

    assert avg_units == round(total_units / order_count, 2)
    assert avg_units > 0


@pytest.mark.parametrize(
    "question",
    [
        "Segment customers into spend quartiles.",
        "Show customer spending quartiles.",
        "Break customers into spend tiers.",
        "Which quartile does each customer fall into by spend?",
    ],
)
def test_offline_matches_spend_quartile_phrasings(question):
    # Several segmentation phrasings resolve to the quartile rule, which buckets
    # customers into four tiers with NTILE(4) over their descending spend.
    sql = OfflineBackend().to_sql(question, schema="")
    assert "NTILE(4) OVER (ORDER BY total_spent DESC, customer_id)" in sql
    assert "GROUP BY quartile" in sql
    assert sql.lower().startswith("with")


def test_spend_quartile_does_not_shadow_top_spenders_or_quarter():
    # "quartile" must not be confused with "quarter": a bare "top customers by
    # spend" (no quartile/tier word) must still hit the LIMIT-5 rule, and a
    # "revenue by quarter" question must still hit the quarter-bucketing rule.
    top_spenders = OfflineBackend().to_sql("Which 5 customers spent the most?", schema="")
    assert "LIMIT 5" in top_spenders
    assert "NTILE" not in top_spenders

    by_quarter = OfflineBackend().to_sql("Show revenue by quarter in 2024.", schema="")
    assert "GROUP BY quarter" in by_quarter
    assert "NTILE" not in by_quarter


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_spend_quartiles():
    # Exactly four quartiles in ascending order. NTILE makes the buckets as equal
    # in size as possible (counts differ by at most one), their customer counts
    # sum to the number of customers who have placed an order, average spend is
    # non-increasing from quartile 1 (top spenders) to quartile 4, and the
    # quartile spend totals sum to total revenue (every buying customer is
    # counted exactly once).
    import sqlite3

    ans = generator.answer_question(DB, "Segment customers into spend quartiles.")
    assert ans.result.columns == ["quartile", "customers", "total_spent", "avg_spent"]

    quartiles = [row[0] for row in ans.result.rows]
    assert quartiles == [1, 2, 3, 4]

    counts = [row[1] for row in ans.result.rows]
    assert max(counts) - min(counts) <= 1  # NTILE keeps buckets near-equal

    avgs = [row[3] for row in ans.result.rows]
    assert avgs == sorted(avgs, reverse=True)  # quartile 1 spends the most

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        buying_customers = conn.execute(
            "SELECT COUNT(DISTINCT customer_id) FROM orders"
        ).fetchone()[0]
    finally:
        conn.close()
    assert sum(counts) == buying_customers

    quartile_total = round(sum(row[2] for row in ans.result.rows), 2)
    total = generator.answer_question(DB, "What is the total revenue?")
    assert quartile_total == total.result.rows[0][0]


@pytest.mark.parametrize(
    "question",
    [
        "Which categories have above-average revenue?",
        "Show categories with above the average revenue.",
        "Which categories earn more than average?",
        "List categories that beat the average revenue.",
    ],
)
def test_offline_matches_above_average_category_phrasings(question):
    # Several "above-average" phrasings resolve to the filter rule, which keeps
    # only categories whose revenue exceeds the mean via a scalar subquery in the
    # WHERE clause rather than a window function.
    sql = OfflineBackend().to_sql(question, schema="")
    assert "WHERE revenue > (SELECT AVG(revenue) FROM category_revenue)" in sql
    assert sql.lower().startswith("with")


def test_above_average_category_does_not_shadow_plain_or_share_category():
    # A bare "revenue by category" (no above-average word) must still route to the
    # simple non-window rule, and a "share by category" question to the
    # percentage rule -- neither should be swallowed by the above-average filter.
    plain = OfflineBackend().to_sql("Show revenue by category", schema="")
    assert "SELECT AVG(revenue)" not in plain

    share = OfflineBackend().to_sql(
        "What percentage of revenue comes from each category?", schema=""
    )
    assert "pct_of_total" in share
    assert "SELECT AVG(revenue)" not in share


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_above_average_categories():
    # The result must contain only categories whose revenue is strictly above the
    # mean category revenue, returned revenue-descending. Cross-check against an
    # independent recomputation of every category's revenue and their average, and
    # confirm the filter is a proper non-empty subset of all categories (with four
    # distinct category totals, at least one is above and at least one below the
    # mean).
    import sqlite3

    ans = generator.answer_question(DB, "Which categories have above-average revenue?")
    assert ans.result.columns == ["category", "revenue"]

    revenues = [row[1] for row in ans.result.rows]
    assert revenues == sorted(revenues, reverse=True)

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        all_categories = conn.execute(
            """
            SELECT p.category, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
            FROM products p
            JOIN order_items oi ON oi.product_id = p.id
            GROUP BY p.category
            """
        ).fetchall()
    finally:
        conn.close()

    mean_revenue = sum(rev for _, rev in all_categories) / len(all_categories)
    expected = {cat for cat, rev in all_categories if rev > mean_revenue}
    returned = {row[0] for row in ans.result.rows}

    assert returned == expected
    assert 0 < len(returned) < len(all_categories)  # a proper, non-empty subset


@pytest.mark.parametrize(
    "question",
    [
        "What is the median order value?",
        "Show me the median order total.",
        "What's the median basket?",
        "For order value, what is the median?",
    ],
)
def test_offline_matches_median_order_value_phrasings(question):
    # Several "median" phrasings resolve to the LIMIT/OFFSET middle-row rule,
    # which is how a median is computed in SQLite (there is no MEDIAN function).
    sql = OfflineBackend().to_sql(question, schema="")
    assert "median_order_value" in sql
    assert "LIMIT 2 - (SELECT COUNT(*) FROM order_totals) % 2" in sql


def test_median_and_average_order_value_do_not_shadow_each_other():
    # The two rules answer different questions over the same per-order rollup and
    # must stay distinct: "average" must not be routed to the median rule, and
    # "median" must not fall through to the average rule.
    average = OfflineBackend().to_sql("What is the average order value?", schema="")
    assert "average_order_value" in average
    assert "OFFSET" not in average

    median = OfflineBackend().to_sql("What is the median order value?", schema="")
    assert "median_order_value" in median


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_median_order_value():
    # Cross-check the SQL median against an independent Python computation over
    # every order total, and assert the defining property of a median: half the
    # orders fall at or below it. The sample DB has an even number of orders, so
    # this also exercises the two-middle-rows branch of the LIMIT expression.
    import sqlite3
    import statistics

    ans = generator.answer_question(DB, "What is the median order value?")
    assert ans.result.columns == ["median_order_value"]
    median = ans.result.rows[0][0]

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        totals = [
            row[0]
            for row in conn.execute(
                """
                SELECT SUM(oi.quantity * oi.unit_price)
                FROM orders o
                JOIN order_items oi ON oi.order_id = o.id
                GROUP BY o.id
                """
            )
        ]
    finally:
        conn.close()

    assert len(totals) % 2 == 0  # even count -> averages the two middle values
    assert median == round(statistics.median(totals), 2)
    assert sum(1 for t in totals if t <= median) >= len(totals) / 2

    # The distribution is right-skewed, so the median should sit below the mean;
    # this is the reason both rules exist rather than one replacing the other.
    average = generator.answer_question(DB, "What is the average order value?")
    assert median < average.result.rows[0][0]


@pytest.mark.parametrize(
    "question",
    [
        "Which products are most frequently bought together?",
        "What products are often purchased together?",
        "Show me the market basket pairs.",
        "Which product pairs co-occur most?",
    ],
)
def test_offline_matches_bought_together_phrasings(question):
    # Every affinity/market-basket phrasing resolves to the self-join rule and
    # not to a product-count or basket-size rule.
    sql = OfflineBackend().to_sql(question, schema="")
    assert "FROM order_items oi1" in sql
    assert "oi1.product_id < oi2.product_id" in sql
    assert "COUNT(DISTINCT oi1.order_id)" in sql


def test_bought_together_does_not_shadow_basket_size():
    # "basket size" (units per order) and "market basket" (product affinity) are
    # different questions; each must route to its own rule.
    affinity = OfflineBackend().to_sql(
        "Which products are frequently bought together?", schema=""
    )
    assert "orders_together" in affinity

    basket = OfflineBackend().to_sql("What is the average basket size?", schema="")
    assert "avg_units_per_order" in basket
    assert affinity != basket


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_bought_together():
    # Cross-check the self-join affinity query against an independent Python
    # computation: for every order, count each unordered pair of distinct
    # products once, then confirm the SQL returns the same top pairs by
    # co-occurring order count (with the same deterministic tiebreak).
    import sqlite3
    from collections import Counter
    from itertools import combinations

    ans = generator.answer_question(
        DB, "Which products are most frequently bought together?"
    )
    assert ans.result.columns == ["product_a", "product_b", "orders_together"]
    assert len(ans.result.rows) == 5

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        names_by_id = dict(conn.execute("SELECT id, name FROM products"))
        order_products: dict[int, set[int]] = {}
        for order_id, product_id in conn.execute(
            "SELECT order_id, product_id FROM order_items"
        ):
            order_products.setdefault(order_id, set()).add(product_id)
    finally:
        conn.close()

    # Mirror the SQL exactly: each unordered pair is keyed by product *id* order
    # (oi1.product_id < oi2.product_id), and the displayed a/b names follow that
    # id order -- not alphabetical order.
    pair_counts: Counter[tuple[str, str]] = Counter()
    for product_ids in order_products.values():
        for id_a, id_b in combinations(sorted(product_ids), 2):
            pair_counts[(names_by_id[id_a], names_by_id[id_b])] += 1

    expected = sorted(
        pair_counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])
    )[:5]
    expected_rows = [(a, b, count) for (a, b), count in expected]
    got_rows = [(row[0], row[1], row[2]) for row in ans.result.rows]
    assert got_rows == expected_rows


@pytest.mark.parametrize(
    "question",
    [
        "What is the average customer lifespan?",
        "Show average customer tenure in days.",
        "How long do customers stay active on average?",
        "What is the typical customer lifespan?",
    ],
)
def test_offline_matches_customer_lifespan_phrasings(question):
    # Several lifespan/tenure phrasings resolve to the date-arithmetic rule, which
    # measures each customer's first-to-last-order span with julianday() date
    # differences (SQLite has no DATEDIFF) and averages those spans.
    sql = OfflineBackend().to_sql(question, schema="")
    assert "julianday(MAX(order_date))" in sql
    assert "avg_customer_lifespan_days" in sql
    assert sql.lower().startswith("with")


def test_customer_lifespan_does_not_shadow_customer_count():
    # "customer lifespan" (a date-span metric) must not collide with the plain
    # "how many customers" count rule; each routes to its own rule.
    lifespan = OfflineBackend().to_sql("What is the average customer lifespan?", schema="")
    count = OfflineBackend().to_sql("How many customers do we have?", schema="")
    assert "avg_customer_lifespan_days" in lifespan
    assert count == "SELECT COUNT(*) AS customer_count FROM customers"
    assert lifespan != count


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_customer_lifespan():
    # A single figure: the mean per-customer first-to-last-order span in days.
    # Cross-check against an independent Python recomputation over each buyer's
    # order dates, and confirm the span is bounded by the one-year data window
    # (all orders fall in calendar 2024, so no customer's span can reach 366 days)
    # and is strictly positive (customers place multiple orders across the year).
    import sqlite3
    from datetime import date

    ans = generator.answer_question(DB, "What is the average customer lifespan?")
    assert ans.result.columns == ["avg_customer_lifespan_days"]
    assert len(ans.result.rows) == 1
    avg_days = ans.result.rows[0][0]

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        dates_by_customer: dict[int, list[date]] = {}
        for customer_id, order_date in conn.execute(
            "SELECT customer_id, order_date FROM orders"
        ):
            dates_by_customer.setdefault(customer_id, []).append(
                date.fromisoformat(order_date)
            )
    finally:
        conn.close()

    spans = [
        (max(dates) - min(dates)).days for dates in dates_by_customer.values()
    ]
    expected = round(sum(spans) / len(spans), 1)
    assert avg_days == expected
    assert 0 < avg_days < 366


@pytest.mark.parametrize(
    "question",
    [
        "Which customers haven't ordered in the last 90 days?",
        "List at-risk customers.",
        "Show lapsed customers.",
        "Which customers have not placed an order recently?",
        "Find inactive customers.",
    ],
)
def test_offline_matches_at_risk_customers_phrasings(question):
    # Several churn/recency phrasings resolve to the at-risk rule, which keeps
    # customers whose most recent order predates a cutoff anchored to the
    # dataset's newest order (reproducible) rather than the wall-clock today().
    sql = OfflineBackend().to_sql(question, schema="")
    assert "WITH customer_last_order AS" in sql
    assert "MAX(o.order_date) AS last_order_date" in sql
    assert "date((SELECT MAX(order_date) FROM orders), '-90 day')" in sql
    assert sql.lower().startswith("with")


def test_at_risk_customers_does_not_shadow_repeat_or_recent_orders():
    # "at-risk customers" (a per-customer recency filter) must not collide with
    # the repeat-customer count rule or the global recent-orders count rule;
    # each routes to its own rule.
    at_risk = OfflineBackend().to_sql(
        "Which customers haven't ordered in the last 90 days?", schema=""
    )
    assert "last_order_date" in at_risk

    repeat = OfflineBackend().to_sql("How many repeat customers are there?", schema="")
    assert "repeat_customers" in repeat
    assert "last_order_date" not in repeat

    recent = OfflineBackend().to_sql(
        "How many orders were placed in the last 30 days?", schema=""
    )
    assert "recent_orders" in recent
    assert "-30 day" in recent
    assert "customer_last_order" not in recent


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_at_risk_customers():
    # Every returned customer's last order must fall strictly before the 90-day
    # cutoff (computed independently), the rows must be ordered by last order date
    # then customer id, and the result must be a proper, non-empty subset of the
    # buying customers -- cross-checked against a direct recomputation.
    import sqlite3

    ans = generator.answer_question(
        DB, "Which customers haven't ordered in the last 90 days?"
    )
    assert ans.result.columns == ["customer_id", "name", "last_order_date"]

    keys = [(row[2], row[0]) for row in ans.result.rows]  # (last_order_date, id)
    assert keys == sorted(keys)

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        cutoff = conn.execute(
            "SELECT date((SELECT MAX(order_date) FROM orders), '-90 day')"
        ).fetchone()[0]
        expected = conn.execute(
            """
            SELECT c.id, MAX(o.order_date)
            FROM customers c
            JOIN orders o ON o.customer_id = c.id
            GROUP BY c.id
            HAVING MAX(o.order_date) < ?
            """,
            (cutoff,),
        ).fetchall()
        buying_customers = conn.execute(
            "SELECT COUNT(DISTINCT customer_id) FROM orders"
        ).fetchone()[0]
    finally:
        conn.close()

    for _customer_id, _name, last_order_date in ans.result.rows:
        assert last_order_date < cutoff

    assert {row[0] for row in ans.result.rows} == {cid for cid, _ in expected}
    assert 0 < len(ans.result.rows) < buying_customers  # proper, non-empty subset


@pytest.mark.parametrize(
    "question",
    [
        "What is the average order value by month in 2024?",
        "Show average order value per month.",
        "How has monthly average order value trended?",
        "What was the avg order value each month?",
    ],
)
def test_offline_matches_monthly_avg_order_value_phrasings(question):
    # Several monthly-AOV phrasings resolve to the time-series rule, which
    # averages per-order totals (computed in a CTE) grouped by month.
    sql = OfflineBackend().to_sql(question, schema="")
    assert "WITH order_totals AS" in sql
    assert "GROUP BY month" in sql
    assert sql.lower().startswith("with")


def test_monthly_avg_order_value_does_not_shadow_overall_or_region_rules():
    # A bare "average order value" must still route to the overall rule, and an
    # "average order value by region" to the region rule -- the monthly rule
    # only claims questions that name a month/over-time dimension.
    overall = OfflineBackend().to_sql("What is the average order value?", schema="")
    assert overall.lower().startswith("select")
    assert "GROUP BY month" not in overall

    by_region = OfflineBackend().to_sql(
        "What is the average order value by region?", schema=""
    )
    assert "GROUP BY c.region" in by_region
    assert "GROUP BY month" not in by_region


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_monthly_avg_order_value():
    # All sample orders fall in 2024, so all 12 months appear in calendar order.
    # Each month's value is cross-checked against an independent Python
    # recomputation: average of that month's per-order totals, rounded to cents.
    import sqlite3
    from collections import defaultdict

    ans = generator.answer_question(
        DB, "What is the average order value by month in 2024?"
    )
    assert ans.result.columns == ["month", "avg_order_value"]
    months = [row[0] for row in ans.result.rows]
    assert months == [f"2024-{m:02d}" for m in range(1, 13)]

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        totals_by_month: dict[str, list[float]] = defaultdict(list)
        for month, order_total in conn.execute(
            """
            SELECT strftime('%Y-%m', o.order_date) AS month,
                   SUM(oi.quantity * oi.unit_price) AS order_total
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.id
            GROUP BY o.id
            """
        ):
            totals_by_month[month].append(order_total)
    finally:
        conn.close()

    for month, avg_value in ans.result.rows:
        totals = totals_by_month[month]
        expected = round(sum(totals) / len(totals), 2)
        assert avg_value == expected


def test_time_between_orders_uses_partitioned_lag():
    # The purchase-cadence rule must use a per-customer (PARTITION BY) LAG so
    # gaps never span two different customers, and must not be swallowed by the
    # broad order-count rule that also mentions "orders".
    sql = OfflineBackend().to_sql(
        "What is the average time between orders?", schema=""
    )
    assert "PARTITION BY o.customer_id" in sql
    assert "LAG(o.order_date)" in sql
    assert "julianday" in sql
    # Not the plain order-count fallback.
    assert sql != "SELECT COUNT(*) AS order_count FROM orders"


def test_time_between_orders_does_not_shadow_order_count():
    # A bare "how many orders" must still route to the order-count rule, proving
    # the cadence rule's matcher is specific to between-orders phrasing.
    count_sql = OfflineBackend().to_sql("How many orders do we have?", schema="")
    assert count_sql == "SELECT COUNT(*) AS order_count FROM orders"


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_time_between_orders():
    # The single average is cross-checked against an independent Python
    # recomputation: for each customer, sort their order dates and sum the
    # day-gaps between consecutive orders, then average every gap across all
    # customers (customers with a single order contribute no gap).
    import sqlite3
    from collections import defaultdict
    from datetime import date

    ans = generator.answer_question(DB, "What is the average time between orders?")
    assert ans.result.columns == ["avg_days_between_orders"]
    assert len(ans.result.rows) == 1

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        dates_by_customer: dict[int, list[str]] = defaultdict(list)
        for customer_id, order_date in conn.execute(
            "SELECT customer_id, order_date FROM orders"
        ):
            dates_by_customer[customer_id].append(order_date)
    finally:
        conn.close()

    gaps: list[int] = []
    for order_dates in dates_by_customer.values():
        parsed = sorted(date.fromisoformat(d) for d in order_dates)
        gaps.extend(
            (parsed[i] - parsed[i - 1]).days for i in range(1, len(parsed))
        )

    expected = round(sum(gaps) / len(gaps), 1)
    assert ans.result.rows[0][0] == expected


@pytest.mark.parametrize(
    "question",
    [
        "What is the average revenue per customer?",
        "Show revenue per customer.",
        "What is ARPU?",
        "What is the mean spend per customer?",
        "average sales per customer",
    ],
)
def test_offline_matches_arpu_phrasings(question):
    # Several ARPU phrasings resolve to the per-customer-rollup rule, which
    # averages each customer's total spend (a CTE grouped by customer) rather
    # than averaging orders or raw order-item rows.
    sql = OfflineBackend().to_sql(question, schema="")
    assert "WITH customer_revenue AS" in sql
    assert "avg_revenue_per_customer" in sql
    assert sql.lower().startswith("with")


def test_arpu_does_not_shadow_order_value_or_top_spenders():
    # ARPU (revenue per customer) must stay distinct from average order value
    # (revenue per order) and from the top-spenders ranking: a bare "average
    # order value" question must still route to the per-order rule, and "top 5
    # customers" to the LIMIT-5 ranking -- neither should hit the ARPU rule.
    arpu = OfflineBackend().to_sql("What is the average revenue per customer?", schema="")
    assert "avg_revenue_per_customer" in arpu

    aov = OfflineBackend().to_sql("What is the average order value?", schema="")
    assert "average_order_value" in aov
    assert "avg_revenue_per_customer" not in aov

    top = OfflineBackend().to_sql("Which 5 customers spent the most?", schema="")
    assert "LIMIT 5" in top
    assert "customer_revenue" not in top


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_arpu():
    # A single summary row. ARPU is cross-checked against an independent Python
    # recomputation over each customer's total spend, and must strictly exceed
    # average order value on this data (customers place several orders each), the
    # property that makes ARPU and AOV different metrics rather than one figure.
    import sqlite3
    from collections import defaultdict

    ans = generator.answer_question(DB, "What is the average revenue per customer?")
    assert ans.result.columns == [
        "paying_customers",
        "total_revenue",
        "avg_revenue_per_customer",
    ]
    assert len(ans.result.rows) == 1
    paying_customers, total_revenue, arpu = ans.result.rows[0]

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        revenue_by_customer: dict[int, float] = defaultdict(float)
        for customer_id, line_total in conn.execute(
            """
            SELECT o.customer_id, oi.quantity * oi.unit_price
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.id
            """
        ):
            revenue_by_customer[customer_id] += line_total
    finally:
        conn.close()

    per_customer = list(revenue_by_customer.values())
    assert paying_customers == len(per_customer)
    assert total_revenue == round(sum(per_customer), 2)
    assert arpu == round(sum(per_customer) / len(per_customer), 2)

    # ARPU divides by customers, AOV by orders, so ARPU must be the larger figure.
    aov = generator.answer_question(DB, "What is the average order value?")
    assert arpu > aov.result.rows[0][0]


@pytest.mark.parametrize(
    "question",
    [
        "Show the distribution of orders per customer.",
        "What is the distribution of the number of orders?",
        "How many orders does each customer place?",
        "orders per customer",
        "order-count distribution",
    ],
)
def test_offline_matches_orders_per_customer_distribution_phrasings(question):
    # Every phrasing must route to the nested-aggregation histogram rule: an
    # inner per-customer COUNT wrapped by an outer GROUP BY over those counts.
    sql = OfflineBackend().to_sql(question, schema="")
    assert "WITH orders_per_customer AS" in sql
    assert "COUNT(*) AS order_count" in sql
    assert "GROUP BY order_count" in sql
    assert "ORDER BY order_count" in sql


def test_orders_per_customer_does_not_shadow_order_or_customer_counts():
    # The distribution rule must be specific enough that bare count questions
    # still fall through to their own broad rules.
    order_count_sql = OfflineBackend().to_sql("How many orders do we have?", schema="")
    assert order_count_sql == "SELECT COUNT(*) AS order_count FROM orders"

    customer_count_sql = OfflineBackend().to_sql(
        "How many customers do we have?", schema=""
    )
    assert "FROM customers" in customer_count_sql
    assert "order_count" not in customer_count_sql


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_orders_per_customer_distribution():
    # The histogram is cross-checked against an independent Python recomputation:
    # count each customer's orders, then count how many customers share each
    # order count. Rows must come back ordered by order_count ascending.
    import sqlite3
    from collections import Counter

    ans = generator.answer_question(
        DB, "Show the distribution of orders per customer."
    )
    assert ans.result.columns == ["order_count", "customers"]

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        orders_by_customer = Counter(
            customer_id
            for (customer_id,) in conn.execute("SELECT customer_id FROM orders")
        )
    finally:
        conn.close()

    expected = sorted(Counter(orders_by_customer.values()).items())
    assert [tuple(row) for row in ans.result.rows] == expected
    # Every customer that placed an order is accounted for exactly once.
    assert sum(customers for _, customers in ans.result.rows) == len(orders_by_customer)


@pytest.mark.parametrize(
    "question",
    [
        "What is the best-selling product in each category?",
        "Show the top selling product per category.",
        "Best product by category",
        "Which category's best-selling products lead in units?",
    ],
)
def test_best_selling_product_per_category_routes_to_partition_rule(question):
    # Every phrasing must route to the greatest-N-per-group rule that ranks
    # products within each category, not the global single-product rule.
    sql = OfflineBackend().to_sql(question, schema="")
    assert "PARTITION BY category" in sql
    assert "WHERE rn = 1" in sql


def test_best_selling_per_category_does_not_shadow_global_best_seller():
    # A bare "best selling product" question (no category) must still fall
    # through to the single-product global ranking, not the per-category rule.
    sql = OfflineBackend().to_sql("What is the best selling product?", schema="")
    assert "PARTITION BY category" not in sql
    assert "LIMIT 1" in sql


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_best_selling_product_per_category():
    # Cross-check the per-category winner against an independent Python
    # recomputation: sum units per product, then pick the max-units product in
    # each category (ties broken by product id, matching the SQL tiebreaker).
    import sqlite3

    ans = generator.answer_question(
        DB, "What is the best-selling product in each category?"
    )
    assert ans.result.columns == ["category", "name", "units_sold"]

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT p.category, p.id, p.name, SUM(oi.quantity) AS units
            FROM products p
            JOIN order_items oi ON oi.product_id = p.id
            GROUP BY p.id
            """
        ).fetchall()
    finally:
        conn.close()

    winners: dict[str, tuple] = {}
    for category, pid, name, units in rows:
        # Higher units win; on a tie the smaller product id wins.
        best = winners.get(category)
        if best is None or (units, -pid) > (best[0], -best[1]):
            winners[category] = (units, pid, name)
    expected = [
        (category, winners[category][2], winners[category][0])
        for category in sorted(winners)
    ]

    assert [tuple(row) for row in ans.result.rows] == expected
    # One winner per category, and every category is represented.
    assert len(ans.result.rows) == len(winners)


@pytest.mark.parametrize(
    "question",
    [
        "How much revenue comes from new vs returning customers?",
        "Split revenue between new and returning customers.",
        "Show revenue from new versus returning customers.",
        "Break down sales by first-time vs repeat customers.",
    ],
)
def test_offline_matches_new_vs_returning_phrasings(question):
    # Several new-vs-returning phrasings resolve to the revenue-attribution rule,
    # which labels each order as a customer's first ('new') or a later one
    # ('returning') with a per-customer ROW_NUMBER, then sums revenue per label.
    sql = OfflineBackend().to_sql(question, schema="")
    assert "PARTITION BY customer_id" in sql
    assert "THEN 'new'" in sql
    assert "GROUP BY customer_type" in sql
    assert sql.lower().startswith("with")


def test_new_vs_returning_does_not_shadow_repeat_customer_count():
    # The new-vs-returning revenue split names "returning customers", but a bare
    # "how many repeat customers" question must still route to the simple count
    # rule -- the split rule only claims the new-vs-returning contrast phrasing.
    split = OfflineBackend().to_sql(
        "How much revenue comes from new vs returning customers?", schema=""
    )
    assert "customer_type" in split

    repeat = OfflineBackend().to_sql("How many repeat customers are there?", schema="")
    assert repeat == (
        "SELECT COUNT(*) AS repeat_customers FROM ( "
        "SELECT customer_id FROM orders GROUP BY customer_id "
        "HAVING COUNT(*) > 1 )"
    )
    assert split != repeat


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_new_vs_returning_revenue():
    # Exactly two labelled buckets returned alphabetically ('new', 'returning').
    # Each customer contributes exactly one 'new' order (their first), so the new
    # order count must equal the number of buying customers; the two buckets'
    # order counts must sum to the total order count and their revenues to total
    # revenue -- every order is counted once. Cross-check against an independent
    # Python recomputation that labels each order by first-order date per customer.
    import sqlite3

    ans = generator.answer_question(
        DB, "How much revenue comes from new vs returning customers?"
    )
    assert ans.result.columns == ["customer_type", "orders", "revenue"]

    types = [row[0] for row in ans.result.rows]
    assert types == ["new", "returning"]
    by_type = {row[0]: (row[1], row[2]) for row in ans.result.rows}

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        buying_customers = conn.execute(
            "SELECT COUNT(DISTINCT customer_id) FROM orders"
        ).fetchone()[0]
        order_totals = conn.execute(
            """
            SELECT o.customer_id, o.id, o.order_date,
                   SUM(oi.quantity * oi.unit_price) AS order_total
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.id
            GROUP BY o.id
            """
        ).fetchall()
    finally:
        conn.close()

    # One 'new' order per buying customer.
    assert by_type["new"][0] == buying_customers

    # Independent recomputation: the first order (by date, then id) per customer
    # is 'new', the rest 'returning'.
    first_order_id: dict[int, tuple] = {}
    for customer_id, order_id, order_date, _total in order_totals:
        key = (order_date, order_id)
        if customer_id not in first_order_id or key < first_order_id[customer_id]:
            first_order_id[customer_id] = key

    expected = {"new": [0, 0.0], "returning": [0, 0.0]}
    for customer_id, order_id, order_date, total in order_totals:
        label = "new" if (order_date, order_id) == first_order_id[customer_id] else "returning"
        expected[label][0] += 1
        expected[label][1] += total

    for label in ("new", "returning"):
        assert by_type[label][0] == expected[label][0]
        assert by_type[label][1] == round(expected[label][1], 2)

    # The two buckets partition every order and all revenue.
    total_orders = sum(by_type[label][0] for label in by_type)
    total_revenue = round(sum(by_type[label][1] for label in by_type), 2)
    assert total_orders == len(order_totals)
    total = generator.answer_question(DB, "What is the total revenue?")
    assert total_revenue == total.result.rows[0][0]


@pytest.mark.parametrize(
    "question",
    [
        "Show monthly cohort retention.",
        "What does our customer retention look like?",
        "Break customers into cohorts by first order month.",
        "How many customers are retained after their first purchase?",
    ],
)
def test_offline_matches_cohort_retention_phrasings(question):
    # The cohort rule owns the cohort/retention vocabulary. It buckets customers
    # by the month of their first order and reports, per month offset, what share
    # of that cohort was active again.
    sql = OfflineBackend().to_sql(question, schema="")
    assert sql.lower().startswith("with")
    assert "cohort_month" in sql
    assert "month_offset" in sql
    assert "COUNT(DISTINCT a.customer_id)" in sql


def test_cohort_retention_does_not_shadow_neighbouring_customer_rules():
    # The cohort rule is registered first, so this is the direction of shadowing
    # that actually needs guarding: it claims only the cohort/retention wording,
    # so questions about repeat customers, lapsed customers, and the
    # new-vs-returning split must still reach their own rules further down --
    # all four concern the same customers but answer different things.
    cohort = OfflineBackend().to_sql("Show monthly cohort retention.", schema="")
    repeat = OfflineBackend().to_sql("How many repeat customers are there?", schema="")
    lapsed = OfflineBackend().to_sql(
        "Which customers haven't ordered in the last 90 days?", schema=""
    )
    split = OfflineBackend().to_sql(
        "How much revenue comes from new vs returning customers?", schema=""
    )

    assert "repeat_customers" in repeat
    assert "cohort_month" not in lapsed
    assert "customer_type" in split
    assert len({cohort, repeat, lapsed, split}) == 4


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_cohort_retention():
    # Verified three ways: (1) structural invariants that must hold for any
    # cohort grid -- offset 0 is a cohort's own first month, so its retention is
    # 100% and its active count equals the cohort size; offsets are never
    # negative; and the cohort sizes sum to the number of buying customers;
    # (2) retention_pct is consistent with its own numerator and denominator;
    # (3) an independent Python recomputation straight from the raw orders.
    import sqlite3
    from collections import defaultdict

    ans = generator.answer_question(DB, "Show monthly cohort retention.")
    assert ans.result.columns == [
        "cohort_month",
        "month_offset",
        "cohort_size",
        "active_customers",
        "retention_pct",
    ]
    rows = [tuple(row) for row in ans.result.rows]
    assert rows, "the cohort grid should not be empty"

    # Rows arrive as a stable grid: cohort ascending, then offset ascending.
    assert rows == sorted(rows, key=lambda row: (row[0], row[1]))

    sizes: dict[str, int] = {}
    for cohort_month, offset, cohort_size, active, pct in rows:
        assert offset >= 0, "a customer cannot be active before their first order"
        assert 0 < active <= cohort_size
        assert pct == round(100.0 * active / cohort_size, 1)
        # cohort_size is a property of the cohort, so it is identical in every
        # cell of that cohort's row.
        assert sizes.setdefault(cohort_month, cohort_size) == cohort_size
        if offset == 0:
            assert active == cohort_size
            assert pct == 100.0

    # Every cohort has an offset-0 baseline row.
    assert {row[0] for row in rows if row[1] == 0} == set(sizes)

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        buying_customers = conn.execute(
            "SELECT COUNT(DISTINCT customer_id) FROM orders"
        ).fetchone()[0]
        raw_orders = conn.execute("SELECT customer_id, order_date FROM orders").fetchall()
    finally:
        conn.close()

    # The cohorts partition the set of customers who have ever ordered.
    assert sum(sizes.values()) == buying_customers

    # Independent recomputation from the raw orders.
    first_month: dict[int, str] = {}
    active_months: dict[int, set[str]] = defaultdict(set)
    for customer_id, order_date in raw_orders:
        month = order_date[:7]
        active_months[customer_id].add(month)
        if customer_id not in first_month or month < first_month[customer_id]:
            first_month[customer_id] = month

    def month_number(month: str) -> int:
        """Absolute month index, so subtracting two gives a whole-month gap."""
        return int(month[:4]) * 12 + int(month[5:7])

    expected_sizes: dict[str, int] = defaultdict(int)
    for cohort in first_month.values():
        expected_sizes[cohort] += 1

    expected_active: dict[tuple[str, int], int] = defaultdict(int)
    for customer_id, months in active_months.items():
        cohort = first_month[customer_id]
        for month in months:
            expected_active[(cohort, month_number(month) - month_number(cohort))] += 1

    expected_rows = [
        (
            cohort,
            offset,
            expected_sizes[cohort],
            active,
            round(100.0 * active / expected_sizes[cohort], 1),
        )
        for (cohort, offset), active in sorted(expected_active.items())
    ]
    assert rows == expected_rows


@pytest.mark.parametrize(
    "question",
    [
        "Score customers into RFM segments.",
        "Build an RFM segmentation of our customer base.",
        "Rank customers by recency, frequency, and monetary value.",
        "Show each customer's recency frequency and monetary scores.",
    ],
)
def test_offline_matches_rfm_phrasings(question):
    # The RFM rule owns the acronym and the spelled-out three-measure phrasing.
    sql = OfflineBackend().to_sql(question, schema="")
    assert sql.lower().startswith("with")
    assert "NTILE(5)" in sql
    assert "rfm_cell" in sql


def test_rfm_scores_order_recency_opposite_to_frequency_and_monetary():
    # The one thing that is easy to get wrong and invisible in the output: a
    # high score must always mean "good", so recency (where a smaller gap is
    # better) sorts DESC while frequency and monetary sort ASC.
    sql = OfflineBackend().to_sql("Score customers into RFM segments.", schema="")
    assert "NTILE(5) OVER ( ORDER BY r.recency_days DESC, r.customer_id )" in sql
    assert "NTILE(5) OVER ( ORDER BY r.frequency ASC, r.customer_id )" in sql
    assert "NTILE(5) OVER ( ORDER BY r.monetary ASC, r.customer_id )" in sql


def test_rfm_does_not_shadow_quartile_or_at_risk_rules():
    # RFM composes the recency and monetary lenses that two neighbouring rules
    # own individually, so both must still reach their own patterns: the
    # quartile rule (monetary-only buckets) and the at-risk rule (recency-only
    # filter). All three concern customer value but answer different questions.
    backend = OfflineBackend()
    rfm = backend.to_sql("Score customers into RFM segments.", schema="")
    quartiles = backend.to_sql("Segment customers into spend quartiles.", schema="")
    at_risk = backend.to_sql("Which customers haven't ordered in the last 90 days?", schema="")

    assert "NTILE(4)" in quartiles and "rfm_cell" not in quartiles
    assert "last_order_date" in at_risk and "NTILE" not in at_risk
    assert len({rfm, quartiles, at_risk}) == 3


def _ntile(sorted_ids: list[int], buckets: int) -> dict[int, int]:
    """Reproduce SQL ``NTILE`` in Python: map each id to its 1-based bucket.

    ``NTILE`` splits ``n`` ordered rows into ``buckets`` groups as equal in size
    as possible, giving the first ``n % buckets`` groups one extra row rather
    than distributing the remainder evenly. Recomputing that rule here — instead
    of assuming the groups divide evenly — is what makes the test independent of
    how many customers the sample database happens to contain.
    """
    total = len(sorted_ids)
    base, remainder = divmod(total, buckets)
    assignment: dict[int, int] = {}
    index = 0
    for bucket in range(1, buckets + 1):
        size = base + (1 if bucket <= remainder else 0)
        for identifier in sorted_ids[index : index + size]:
            assignment[identifier] = bucket
        index += size
    return assignment


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_end_to_end_rfm_segmentation():
    # Verified two ways: (1) structural invariants that must hold for any RFM
    # grid -- one row per buying customer, every score in 1..5, the cell string
    # being the three scores concatenated, and the rows ranked by spend; and
    # (2) an independent Python recomputation of all three measures and all
    # three NTILE assignments straight from the raw orders.
    import sqlite3
    from collections import defaultdict
    from datetime import date

    ans = generator.answer_question(DB, "Score customers into RFM segments.")
    assert ans.result.columns == [
        "name",
        "recency_days",
        "frequency",
        "monetary",
        "r_score",
        "f_score",
        "m_score",
        "rfm_cell",
    ]
    rows = [tuple(row) for row in ans.result.rows]
    assert rows, "the RFM table should not be empty"

    for _name, recency, frequency, monetary, r, f, m, cell in rows:
        assert recency >= 0
        assert frequency >= 1
        assert monetary > 0
        assert {r, f, m} <= {1, 2, 3, 4, 5}
        assert cell == f"{r}{f}{m}"

    # Ranked by spend, so the ordering is part of the answer.
    assert [row[3] for row in rows] == sorted((row[3] for row in rows), reverse=True)

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        raw = conn.execute(
            "SELECT o.customer_id, o.id, o.order_date, oi.quantity, oi.unit_price "
            "FROM orders o JOIN order_items oi ON oi.order_id = o.id"
        ).fetchall()
        names = dict(conn.execute("SELECT id, name FROM customers").fetchall())
        as_of = conn.execute("SELECT MAX(order_date) FROM orders").fetchone()[0]
    finally:
        conn.close()

    last_order: dict[int, str] = {}
    order_ids: dict[int, set[int]] = defaultdict(set)
    spend: dict[int, float] = defaultdict(float)
    for customer_id, order_id, order_date, quantity, unit_price in raw:
        last_order[customer_id] = max(last_order.get(customer_id, ""), order_date)
        order_ids[customer_id].add(order_id)
        spend[customer_id] += quantity * unit_price

    anchor = date.fromisoformat(as_of)
    recency = {
        customer_id: (anchor - date.fromisoformat(latest)).days
        for customer_id, latest in last_order.items()
    }
    frequency = {customer_id: len(ids) for customer_id, ids in order_ids.items()}
    monetary = {customer_id: round(total, 2) for customer_id, total in spend.items()}
    customers = sorted(recency)
    assert len(rows) == len(customers)

    r_score = _ntile(sorted(customers, key=lambda c: (-recency[c], c)), 5)
    f_score = _ntile(sorted(customers, key=lambda c: (frequency[c], c)), 5)
    m_score = _ntile(sorted(customers, key=lambda c: (monetary[c], c)), 5)

    expected_rows = [
        (
            names[customer_id],
            recency[customer_id],
            frequency[customer_id],
            monetary[customer_id],
            r_score[customer_id],
            f_score[customer_id],
            m_score[customer_id],
            f"{r_score[customer_id]}{f_score[customer_id]}{m_score[customer_id]}",
        )
        for customer_id in sorted(customers, key=lambda c: (-monetary[c], c))
    ]
    assert rows == expected_rows


@pytest.mark.parametrize(
    "question",
    [
        "Which products grew or declined in the second half of 2024?",
        "Compare product revenue in the first half of 2024 with the rest of the year.",
        "Show me first half vs second half sales by product.",
        "Break product revenue into H1 and H2.",
        "Give me half-over-half revenue by product.",
    ],
)
def test_offline_matches_half_over_half_phrasings(question):
    sql = OfflineBackend().to_sql(question, schema="")
    assert "h1_revenue" in sql and "h2_revenue" in sql
    assert "CASE WHEN o.order_date < '2024-07-01'" in sql


def test_half_over_half_does_not_shadow_quarter_or_top_products():
    # The half comparison and the quarterly time series both slice revenue by a
    # date range, and both this rule and the top-products rule mention products
    # and revenue. Each pair must keep routing to its own pattern.
    backend = OfflineBackend()

    quarter_sql = backend.to_sql("Show revenue by quarter in 2024.", schema="")
    assert "'2024-Q'" in quarter_sql
    assert "h1_revenue" not in quarter_sql

    top_products_sql = backend.to_sql("What are the top 5 products by revenue?", schema="")
    assert "LIMIT 5" in top_products_sql
    assert "h1_revenue" not in top_products_sql

    half_sql = backend.to_sql(
        "Which products grew or declined in the second half of 2024?", schema=""
    )
    assert "'2024-Q'" not in half_sql and "LIMIT" not in half_sql


def test_half_over_half_keeps_products_sold_in_only_one_half():
    """``ELSE 0`` is what keeps an appeared/vanished product in the result.

    With ``ELSE NULL`` the untouched half would be NULL, the subtraction would
    yield NULL, and the products with the most extreme change would sort out of
    the answer they exist to surface. Pinning the branch keeps that from being
    "simplified" away later.
    """
    sql = OfflineBackend().to_sql("Show half-over-half revenue by product.", schema="")
    assert sql.count("ELSE 0 END") == 2
    assert "ELSE NULL" not in sql
    # The percentage, unlike the absolute change, must decline to divide by zero.
    assert "NULLIF(h1_revenue, 0)" in sql


def test_end_to_end_half_over_half_product_revenue():
    import sqlite3
    from collections import defaultdict

    ans = generator.answer_question(
        DB, "Which products grew or declined in the second half of 2024?"
    )
    assert ans.result.columns == [
        "product",
        "h1_revenue",
        "h2_revenue",
        "revenue_change",
        "pct_change",
    ]
    rows = [tuple(row) for row in ans.result.rows]
    assert rows, "the half-over-half table should not be empty"

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        raw = conn.execute(
            "SELECT p.name, o.order_date, oi.quantity, oi.unit_price "
            "FROM products p "
            "JOIN order_items oi ON oi.product_id = p.id "
            "JOIN orders o ON o.id = oi.order_id"
        ).fetchall()
    finally:
        conn.close()

    h1: dict[str, float] = defaultdict(float)
    h2: dict[str, float] = defaultdict(float)
    for name, order_date, quantity, unit_price in raw:
        if not ("2024-01-01" <= order_date < "2025-01-01"):
            continue
        half = h1 if order_date < "2024-07-01" else h2
        half[name] += quantity * unit_price

    products = sorted(set(h1) | set(h2))
    expected = sorted(
        (
            (
                name,
                round(h1[name], 2),
                round(h2[name], 2),
                round(h2[name] - h1[name], 2),
            )
            for name in products
        ),
        key=lambda row: (-row[3], row[0]),
    )
    assert [row[:4] for row in rows] == expected

    for name, first_half, second_half, change, pct in rows:
        # The two halves must partition the year: nothing double-counted, and
        # nothing (an order in neither half) dropped.
        assert round(first_half + second_half, 2) == round(
            h1[name] + h2[name], 2
        )
        assert change == round(second_half - first_half, 2)
        # Rounded to one decimal by SQL, so compare within half a step.
        assert pct == pytest.approx(100.0 * change / first_half, abs=0.05)
