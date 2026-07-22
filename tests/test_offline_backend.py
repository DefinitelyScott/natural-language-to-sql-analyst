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
