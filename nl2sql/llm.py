"""Resolve a natural-language question to SQL.

Two backends:

* ``OfflineBackend`` — a deterministic, rule-based matcher over a catalog of
  known analytical question patterns. Requires no network or API key, so the
  test suite and CI use it. It is intentionally small and transparent.
* ``LLMBackend`` — sends the schema + question to an OpenAI-compatible chat
  model and returns the SQL it produces. Used when ``OPENAI_API_KEY`` is set.

Both return a raw SQL string; validation and execution happen in ``runner``.
"""

from __future__ import annotations

import os
import re
from typing import Protocol


class Backend(Protocol):
    def to_sql(self, question: str, schema: str) -> str: ...


# --------------------------------------------------------------------------- #
# Offline rule-based backend
# --------------------------------------------------------------------------- #
class OfflineBackend:
    """Map a question to SQL via lightweight keyword rules.

    This is not meant to be a general NL parser. It recognizes a fixed catalog
    of common analytics questions so the project is runnable and verifiable
    offline. Each rule is a (matcher, sql) pair.
    """

    def __init__(self) -> None:
        self._rules: list[tuple[re.Pattern[str], str]] = [
            # Monthly cohort retention: for each acquisition cohort (customers
            # grouped by the month of their *first* order), what share of that
            # cohort ordered again N months later. This is the standard retention
            # grid -- one row per (cohort, month offset) cell -- and it is the
            # only rule here that measures behavior *relative to each customer's
            # own start date* rather than against the calendar.
            #
            # It is registered first because its vocabulary (cohort / retention /
            # retained) is the narrowest in the catalog and no other rule uses
            # any of those words, so it cannot shadow anything -- while several
            # broad rules below would otherwise swallow it. "How many customers
            # are retained after their first purchase?" is a retention question,
            # but it also contains "how many customers", which the customer-count
            # rule matches; first-rule-wins ordering is what keeps it here.
            #
            # Three CTEs keep each step separable: ``first_order`` assigns every
            # customer their cohort; ``cohort_size`` is the denominator (the
            # cohort's headcount, computed once so it is not re-derived per
            # cell); ``activity`` lists the distinct months each customer was
            # active in, tagged with their cohort.
            #
            # The month offset is deliberately arithmetic on the 'YYYY-MM'
            # strings rather than a date function: SQLite has no month-difference
            # builtin, and julianday() measures *days*, which would make offsets
            # drift across months of unequal length. Converting each month to an
            # absolute month number (year * 12 + month) and subtracting gives an
            # exact whole-month distance. COUNT(DISTINCT customer_id) is used
            # rather than COUNT(*) so the numerator is unambiguously "customers",
            # independent of how ``activity`` happens to be deduplicated.
            #
            # Offset 0 is every cohort's own first month, so its retention is
            # 100% by construction -- that row is the baseline the later
            # percentages are read against, not a result.
            (
                re.compile(r"\bcohorts?\b|\bretention\b|\bretained\b", re.I),
                """
                WITH first_order AS (
                    SELECT customer_id,
                           strftime('%Y-%m', MIN(order_date)) AS cohort_month
                    FROM orders
                    GROUP BY customer_id
                ),
                cohort_size AS (
                    SELECT cohort_month, COUNT(*) AS customers
                    FROM first_order
                    GROUP BY cohort_month
                ),
                activity AS (
                    SELECT DISTINCT f.cohort_month AS cohort_month,
                           o.customer_id AS customer_id,
                           strftime('%Y-%m', o.order_date) AS active_month
                    FROM orders o
                    JOIN first_order f ON f.customer_id = o.customer_id
                )
                SELECT a.cohort_month,
                       (CAST(substr(a.active_month, 1, 4) AS INTEGER) * 12
                        + CAST(substr(a.active_month, 6, 2) AS INTEGER))
                       - (CAST(substr(a.cohort_month, 1, 4) AS INTEGER) * 12
                          + CAST(substr(a.cohort_month, 6, 2) AS INTEGER))
                           AS month_offset,
                       s.customers AS cohort_size,
                       COUNT(DISTINCT a.customer_id) AS active_customers,
                       ROUND(
                           100.0 * COUNT(DISTINCT a.customer_id) / s.customers, 1
                       ) AS retention_pct
                FROM activity a
                JOIN cohort_size s ON s.cohort_month = a.cohort_month
                GROUP BY a.cohort_month, month_offset
                ORDER BY a.cohort_month, month_offset
                """,
            ),
            (
                re.compile(r"total sales by month.*2024", re.I),
                """
                SELECT strftime('%Y-%m', o.order_date) AS month,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                FROM orders o
                JOIN order_items oi ON oi.order_id = o.id
                WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'
                GROUP BY month
                ORDER BY month
                """,
            ),
            # Month-over-month revenue growth. A ``LAG`` window function over a
            # monthly-revenue CTE yields each month's change from the previous
            # month; the first month's change is NULL because there is no prior
            # month to compare against. This is the only rule that uses a window
            # function, and it is placed ahead of the broad "total revenue" rule
            # so "revenue growth" phrasings are not shadowed by it.
            (
                re.compile(
                    r"month[- ]over[- ]month|(revenue|sales)\s+growth|"
                    r"growth.*(revenue|sales)",
                    re.I,
                ),
                """
                WITH monthly AS (
                    SELECT strftime('%Y-%m', o.order_date) AS month,
                           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'
                    GROUP BY month
                )
                SELECT month,
                       revenue,
                       ROUND(revenue - LAG(revenue) OVER (ORDER BY month), 2)
                           AS revenue_change
                FROM monthly
                ORDER BY month
                """,
            ),
            # Cumulative (running-total) revenue by month for 2024. A first CTE
            # totals each month's revenue; the outer query then carries a running
            # sum with SUM(revenue) OVER (ORDER BY month ROWS BETWEEN UNBOUNDED
            # PRECEDING AND CURRENT ROW) -- an *explicit window frame* that sums
            # every month up to and including the current one. This differs from
            # the other window rules here: LAG (month-over-month) looks one row
            # back, and the category-share rule's SUM(...) OVER () has an empty,
            # unordered frame that spans the whole result; an ordered ROWS frame
            # is what turns a plain total into a progressive running total. The
            # final row's cumulative value therefore equals total 2024 revenue.
            # It requires a cumulative/running-total phrasing and is registered
            # ahead of the broad "total revenue" rule so it is not shadowed.
            (
                re.compile(
                    r"cumulative\s+(revenue|sales)|"
                    r"running\s+(total|sum)\s+(of\s+)?(revenue|sales)|"
                    r"(revenue|sales)\s+running\s+total|"
                    r"(revenue|sales)\s+to\s+date",
                    re.I,
                ),
                """
                WITH monthly AS (
                    SELECT strftime('%Y-%m', o.order_date) AS month,
                           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'
                    GROUP BY month
                )
                SELECT month,
                       revenue,
                       ROUND(
                           SUM(revenue) OVER (
                               ORDER BY month
                               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                           ), 2
                       ) AS cumulative_revenue
                FROM monthly
                ORDER BY month
                """,
            ),
            # The single highest-spending customer within each region: the classic
            # "greatest-N-per-group" (top-1-per-partition) problem. A first CTE
            # totals each customer's spend and carries their region; the second
            # ranks customers *inside* each region with
            # ROW_NUMBER() OVER (PARTITION BY region ORDER BY total_spent DESC);
            # the outer query then keeps only rank 1 per region. PARTITION BY
            # restarts the numbering for every region, which is what makes this a
            # per-group ranking rather than one global ranking -- distinct from
            # the other window rules here (LAG for month-over-month, SUM() OVER ()
            # for category share), neither of which partitions. customer_id is a
            # deterministic tiebreaker in the ORDER BY so ties resolve the same
            # way on every run. This rule requires BOTH a top/best/highest word
            # and "region", and is registered ahead of the plain "top 5 customers
            # by spend" rule below so a per-region question is not shadowed by the
            # global top-spenders rule.
            (
                re.compile(
                    r"(top|best|highest)[-\s]*(spending|spender)?\s*customers?.*"
                    r"(in|per|by|within|for)\s+(each\s+)?region|"
                    r"region.*(top|best|highest)[-\s]*(spending|spender)?\s*customers?",
                    re.I,
                ),
                """
                WITH customer_spend AS (
                    SELECT c.id AS customer_id,
                           c.name AS name,
                           c.region AS region,
                           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spent
                    FROM customers c
                    JOIN orders o ON o.customer_id = c.id
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY c.id
                ),
                ranked AS (
                    SELECT region,
                           name,
                           total_spent,
                           ROW_NUMBER() OVER (
                               PARTITION BY region
                               ORDER BY total_spent DESC, customer_id
                           ) AS rn
                    FROM customer_spend
                )
                SELECT region, name, total_spent
                FROM ranked
                WHERE rn = 1
                ORDER BY region
                """,
            ),
            # Segment customers into spend quartiles (four equal-sized tiers by
            # lifetime spend). A first CTE totals each customer's spend; the
            # second assigns a quartile with NTILE(4) OVER (ORDER BY total_spent
            # DESC, customer_id) -- NTILE splits the ordered rows into four
            # groups as equal in size as possible, so quartile 1 is the
            # top-spending 25% of customers and quartile 4 the bottom 25%. The
            # outer query then rolls each quartile up into a one-row summary
            # (customer count, total and average spend). It differs from the
            # ROW_NUMBER top-per-region rule (which ranks every row to pick a
            # single winner) in that NTILE only needs the bucket number, not a
            # full ranking; the RFM rule below is the only other NTILE user, and
            # it buckets on three measures rather than one. customer_id is a
            # deterministic tiebreaker so tied spends fall on the same side of a
            # bucket boundary on every run. The matcher requires a quartile/tier
            # or "segment customers by spend" phrasing, so it does not shadow the
            # plain top-spenders rule below it, and its "quartile" wording will
            # not collide with the "revenue by quarter" rule.
            (
                re.compile(
                    r"\bquartiles?\b|"
                    r"(spend|spending)\s+tiers?|"
                    r"segment\s+customers?\b.*\b(spend|spending|value|revenue)",
                    re.I,
                ),
                """
                WITH customer_spend AS (
                    SELECT c.id AS customer_id,
                           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spent
                    FROM customers c
                    JOIN orders o ON o.customer_id = c.id
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY c.id
                ),
                bucketed AS (
                    SELECT customer_id,
                           total_spent,
                           NTILE(4) OVER (ORDER BY total_spent DESC, customer_id)
                               AS quartile
                    FROM customer_spend
                )
                SELECT quartile,
                       COUNT(*) AS customers,
                       ROUND(SUM(total_spent), 2) AS total_spent,
                       ROUND(AVG(total_spent), 2) AS avg_spent
                FROM bucketed
                GROUP BY quartile
                ORDER BY quartile
                """,
            ),
            # RFM segmentation: score every buying customer on the three classic
            # customer-value dimensions at once -- Recency (how long since their
            # last order), Frequency (how many orders), Monetary (how much they
            # spent) -- and label them with the combined RFM cell. Two other rules
            # already look at one of these lenses in isolation: the at-risk rule
            # is recency-only (a filter), and the spend-quartile rule is
            # monetary-only (four buckets). This rule is what composes them, and
            # the composition is the point: a customer can be a heavy spender and
            # still be lapsing, which neither single-lens rule can express.
            #
            # ``customer_rfm`` reduces the order history to one row per customer
            # holding all three raw measures. Recency is measured against the
            # dataset's own MAX(order_date) rather than wall-clock today(), for
            # the same reproducibility reason as the at-risk and last-30-days
            # rules -- anchoring to the data keeps the answer stable for anyone
            # who clones the repo. julianday() is the right function here (unlike
            # in the cohort rule, which needs whole *months*) because recency is
            # naturally counted in days, and CAST(... AS INTEGER) truncates the
            # half-day artifact of differencing two date-only julian values.
            # COUNT(DISTINCT o.id) is required rather than COUNT(*): the join to
            # order_items multiplies each order by its line count, so COUNT(*)
            # would measure items and silently inflate Frequency.
            #
            # ``scored`` then buckets each measure into fifths with NTILE(5).
            # The three window functions deliberately do not sort the same way:
            # Frequency and Monetary are ordered ASC so that more is better and
            # 5 is the best score, while Recency is ordered DESC because for
            # recency a *smaller* number is better -- the largest gap since the
            # last order lands in bucket 1. Getting that inversion wrong is the
            # classic RFM bug, and it is invisible in the output because the
            # scores still look plausible. customer_id is a tiebreaker in every
            # window so customers with identical measures fall on the same side
            # of a bucket boundary on every run. The scores cannot be
            # concatenated in the same SELECT that computes them (a window
            # function's result is not addressable by alias in its own select
            # list), which is why ``scored`` is a separate CTE and ``rfm_cell``
            # is built in the outer query.
            #
            # The INNER JOIN means only customers who have ordered are scored: an
            # RFM cell is undefined for someone with no recency and no frequency,
            # and treating them as the worst-scoring segment would mix
            # acquisition into a retention metric. The matcher owns the RFM
            # acronym and the "recency, frequency, monetary" phrasing, which no
            # other rule keys on.
            (
                re.compile(
                    r"\brfm\b|"
                    r"recency\s*,?\s*(and\s+)?frequency|"
                    r"frequency\s*,?\s*(and\s+)?monetary",
                    re.I,
                ),
                """
                WITH customer_rfm AS (
                    SELECT o.customer_id AS customer_id,
                           CAST(
                               julianday((SELECT MAX(order_date) FROM orders))
                               - julianday(MAX(o.order_date)) AS INTEGER
                           ) AS recency_days,
                           COUNT(DISTINCT o.id) AS frequency,
                           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS monetary
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY o.customer_id
                ),
                scored AS (
                    SELECT r.customer_id AS customer_id,
                           c.name AS name,
                           r.recency_days AS recency_days,
                           r.frequency AS frequency,
                           r.monetary AS monetary,
                           NTILE(5) OVER (
                               ORDER BY r.recency_days DESC, r.customer_id
                           ) AS r_score,
                           NTILE(5) OVER (
                               ORDER BY r.frequency ASC, r.customer_id
                           ) AS f_score,
                           NTILE(5) OVER (
                               ORDER BY r.monetary ASC, r.customer_id
                           ) AS m_score
                    FROM customer_rfm r
                    JOIN customers c ON c.id = r.customer_id
                )
                SELECT name,
                       recency_days,
                       frequency,
                       monetary,
                       r_score,
                       f_score,
                       m_score,
                       r_score || f_score || m_score AS rfm_cell
                FROM scored
                ORDER BY monetary DESC, customer_id
                """,
            ),
            # Average revenue per customer (ARPU): mean lifetime spend per paying
            # customer. A first CTE rolls the money up to one row per customer
            # (their total spend); the outer query then AVGs those per-customer
            # totals. The per-customer rollup is essential and is the whole point
            # of this rule: ARPU divides revenue by *customers*, whereas average
            # order value (AOV) divides the same revenue by *orders* -- so on data
            # where customers place several orders each, ARPU is many times larger
            # than AOV, and averaging the raw order-item rows would give neither.
            # The denominator is customers who have actually ordered (the INNER
            # JOIN drops never-buyers), which is the standard "per paying customer"
            # definition and is reproducible from the data alone. The matcher owns
            # the "per customer" revenue/spend phrasings and the ARPU acronym,
            # which no other rule keys on; it is registered ahead of the broad
            # "total revenue" and top-spenders rules so a per-customer question is
            # not shadowed by them.
            (
                re.compile(
                    r"\barpu\b|"
                    r"(average|avg|mean).*(revenue|spend|sales).*per\s+customer|"
                    r"(revenue|spend|sales)\s+per\s+customer|"
                    r"per[- ]customer\s+(revenue|spend|sales)",
                    re.I,
                ),
                """
                WITH customer_revenue AS (
                    SELECT o.customer_id AS customer_id,
                           SUM(oi.quantity * oi.unit_price) AS revenue
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY o.customer_id
                )
                SELECT COUNT(*) AS paying_customers,
                       ROUND(SUM(revenue), 2) AS total_revenue,
                       ROUND(AVG(revenue), 2) AS avg_revenue_per_customer
                FROM customer_revenue
                """,
            ),
            (
                re.compile(r"customers?\b.*\bspent|top\s*(5|five)\s*customers", re.I),
                """
                SELECT c.name,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spent
                FROM customers c
                JOIN orders o ON o.customer_id = c.id
                JOIN order_items oi ON oi.order_id = o.id
                GROUP BY c.id
                ORDER BY total_spent DESC
                LIMIT 5
                """,
            ),
            # Revenue broken out by two dimensions at once: customer region and
            # product category. This is the only rule that groups by two columns,
            # and it needs all four tables -- customers (region) -> orders ->
            # order_items (the money) -> products (category) -- so the region and
            # category labels meet on the same order-item rows. The matcher
            # requires BOTH "region" and "category" words and is registered ahead
            # of the single-dimension "revenue by region" and "revenue by
            # category" rules so a combined question is not shadowed by whichever
            # one-dimension rule would otherwise match first.
            (
                re.compile(
                    r"(revenue|sales).*(region.*categor|categor.*region)|"
                    r"(region.*categor|categor.*region).*(revenue|sales)",
                    re.I,
                ),
                """
                SELECT c.region,
                       p.category,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                FROM customers c
                JOIN orders o ON o.customer_id = c.id
                JOIN order_items oi ON oi.order_id = o.id
                JOIN products p ON p.id = oi.product_id
                GROUP BY c.region, p.category
                ORDER BY c.region, revenue DESC
                """,
            ),
            # Categories whose revenue is above the average category revenue.
            # A first CTE totals each category's revenue; the outer query then
            # keeps only the categories whose revenue exceeds the mean, obtained
            # with a scalar subquery SELECT AVG(revenue) FROM category_revenue
            # that re-reads the same CTE. This "above-average" filter is a
            # distinct idiom from the window rules here: SUM(revenue) OVER ()
            # (category share) annotates every row with the grand total but keeps
            # all rows, whereas this scalar subquery is used in the WHERE clause
            # to *drop* rows below the mean. The comparison is against the rounded
            # per-category revenue, so the threshold matches the revenue values
            # shown. The matcher requires an above/over-average phrase next to a
            # category word and is registered ahead of both the category-share
            # and the plain "revenue by category" rules so those bare questions
            # are not shadowed.
            (
                re.compile(
                    r"categor(y|ies).*(above|over|greater than|more than|beat)"
                    r"[-\s]*(the\s+)?average|"
                    r"(above|over)[-\s]*average.*categor",
                    re.I,
                ),
                """
                WITH category_revenue AS (
                    SELECT p.category AS category,
                           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                    FROM products p
                    JOIN order_items oi ON oi.product_id = p.id
                    GROUP BY p.category
                )
                SELECT category, revenue
                FROM category_revenue
                WHERE revenue > (SELECT AVG(revenue) FROM category_revenue)
                ORDER BY revenue DESC
                """,
            ),
            # Each category's revenue as a share (percentage) of total revenue.
            # A CTE first computes per-category revenue; the outer query divides
            # each category by the grand total obtained with SUM(revenue) OVER ()
            # -- a window function with an empty OVER () that sums across every
            # row, i.e. the whole result. The 100.0 literal forces float division
            # so the percentages are not integer-truncated. Requiring a
            # share/percentage word in the matcher, and registering this rule
            # ahead of the plain "revenue by category" rule below, keeps a bare
            # "revenue by category" question routed to that simpler rule.
            (
                re.compile(
                    r"(percent(age)?|share|proportion).*categor|"
                    r"categor.*(percent(age)?|share|proportion)",
                    re.I,
                ),
                """
                WITH category_revenue AS (
                    SELECT p.category AS category,
                           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                    FROM products p
                    JOIN order_items oi ON oi.product_id = p.id
                    GROUP BY p.category
                )
                SELECT category,
                       revenue,
                       ROUND(100.0 * revenue / SUM(revenue) OVER (), 2)
                           AS pct_of_total
                FROM category_revenue
                ORDER BY revenue DESC
                """,
            ),
            (
                re.compile(r"(revenue|sales).*(by|per).*(category)", re.I),
                """
                SELECT p.category,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                FROM products p
                JOIN order_items oi ON oi.product_id = p.id
                GROUP BY p.category
                ORDER BY revenue DESC
                """,
            ),
            # Average order value bucketed by month -- the AOV time-series. Like
            # the other order-value rules, order value is a per-order quantity,
            # so each order's total is computed one level down (in the
            # ``order_totals`` CTE, grouping by order id) before averaging;
            # averaging the raw order_items rows would weight the mean by
            # line-item count rather than by order. The CTE carries each order's
            # month up so the outer query can AVG per month. This differs from
            # the two AOV rules below it: the region rule groups the same
            # per-order totals by customer region, and the plain rule averages
            # them over all orders at once, whereas this one groups them by
            # calendar month to show the trend across the year. The matcher
            # requires a month/over-time phrase and is registered ahead of the
            # plain "average order value" rule so a monthly question is not
            # shadowed by it, while a bare "average order value" still routes to
            # the plain rule.
            (
                re.compile(
                    r"(average|avg).*order value.*"
                    r"(by\s+month|per\s+month|each\s+month|monthly|over\s+time)|"
                    r"monthly.*(average|avg).*order value",
                    re.I,
                ),
                """
                WITH order_totals AS (
                    SELECT o.id AS order_id,
                           strftime('%Y-%m', o.order_date) AS month,
                           SUM(oi.quantity * oi.unit_price) AS order_total
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'
                    GROUP BY o.id
                )
                SELECT month,
                       ROUND(AVG(order_total), 2) AS avg_order_value
                FROM order_totals
                GROUP BY month
                ORDER BY month
                """,
            ),
            # Average order value broken out by customer region. Order value is a
            # per-order quantity, so it must be computed one level down (each
            # order's total in the subquery) before averaging -- averaging the
            # raw order_items rows would weight the mean by line-item count, not
            # by order. The subquery carries customer_id up so the outer query can
            # join to customers for the region label and then AVG per region.
            # Requiring the word "region" and registering this rule ahead of the
            # plain "average order value" rule below keeps a bare "average order
            # value" question routed to that simpler overall rule.
            (
                re.compile(r"(average|avg).*order value.*region", re.I),
                """
                SELECT c.region,
                       ROUND(AVG(order_total), 2) AS avg_order_value
                FROM (
                    SELECT o.id AS order_id,
                           o.customer_id AS customer_id,
                           SUM(oi.quantity * oi.unit_price) AS order_total
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY o.id
                ) t
                JOIN customers c ON c.id = t.customer_id
                GROUP BY c.region
                ORDER BY avg_order_value DESC
                """,
            ),
            (
                re.compile(r"(average|avg).*order value", re.I),
                """
                SELECT ROUND(AVG(order_total), 2) AS average_order_value
                FROM (
                    SELECT o.id, SUM(oi.quantity * oi.unit_price) AS order_total
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY o.id
                )
                """,
            ),
            # Median order value: the *middle* order total, which SQLite has no
            # built-in function for. A first CTE rolls each order up to its total
            # (the same per-order rollup the average-order-value rules use); the
            # subquery then walks the totals in ascending order and keeps only the
            # middle row(s) with LIMIT/OFFSET:
            #   OFFSET (COUNT(*) - 1) / 2  skips the bottom half, and
            #   LIMIT  2 - COUNT(*) % 2    takes 1 row when the count is odd and
            #                              2 rows when it is even.
            # AVG over that 1-or-2-row window is the median by definition: the
            # single middle value, or the mean of the two middle values. This is
            # deliberately paired with the average rule rather than replacing it —
            # order totals are right-skewed (a few large orders pull the mean up),
            # so the median is the more representative "typical order" and the gap
            # between the two is itself informative. The matcher requires the word
            # "median", which no other rule uses, so it neither shadows nor is
            # shadowed by the average-order-value rules.
            (
                re.compile(
                    r"median\s+(order\s+(value|total)|basket)|"
                    r"order\s+(value|total).*median",
                    re.I,
                ),
                """
                WITH order_totals AS (
                    SELECT o.id AS order_id,
                           SUM(oi.quantity * oi.unit_price) AS order_total
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY o.id
                )
                SELECT ROUND(AVG(order_total), 2) AS median_order_value
                FROM (
                    SELECT order_total
                    FROM order_totals
                    ORDER BY order_total
                    LIMIT 2 - (SELECT COUNT(*) FROM order_totals) % 2
                    OFFSET (SELECT (COUNT(*) - 1) / 2 FROM order_totals)
                )
                """,
            ),
            # Average basket size: the mean number of *units* per order, where an
            # order's unit count is SUM(quantity) across its line items. Like the
            # average-order-value rule above, the per-order rollup happens in the
            # subquery and the AVG is taken over those order totals -- averaging
            # the raw order_items rows instead would weight the mean by how many
            # line items an order has, not by order. This differs from average
            # order value only in the numerator: value sums quantity * unit_price
            # (money), basket size sums quantity (units). The matcher requires an
            # items/units-per-order or basket-size phrase, so it does not shadow
            # the "order value" rule (which owns the "order value" wording) or the
            # broad order/product count rules.
            (
                re.compile(
                    r"(average|avg|mean)\s+(number\s+of\s+)?(items?|units?)\s+per\s+order|"
                    r"(items?|units?)\s+per\s+order|"
                    r"average\s+basket\s+size|basket\s+size",
                    re.I,
                ),
                """
                SELECT ROUND(AVG(order_units), 2) AS avg_units_per_order
                FROM (
                    SELECT o.id AS order_id,
                           SUM(oi.quantity) AS order_units
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY o.id
                )
                """,
            ),
            (
                re.compile(r"how many (customers|users)", re.I),
                "SELECT COUNT(*) AS customer_count FROM customers",
            ),
            # The best-selling product (by units) *within each category*: another
            # "greatest-N-per-group" ranking, the product-and-category analogue of
            # the top-customer-per-region rule above. A first CTE rolls units up to
            # one row per product, carrying its category; the second ranks products
            # *inside* each category with
            # ROW_NUMBER() OVER (PARTITION BY category ORDER BY units_sold DESC);
            # the outer query keeps only rank 1 per category. PARTITION BY restarts
            # the numbering for every category, which is what turns one global
            # ranking into a per-category one. product_id breaks ties in the
            # ORDER BY so a category whose top two products are level resolves the
            # same way on every run. This rule requires BOTH a best/top product
            # phrase AND a category word, and is registered ahead of the broad
            # "best selling product" rule below so a per-category question is not
            # shadowed by the single-product global ranking.
            (
                re.compile(
                    r"(best|top)[-\s]*(selling\s+)?products?.*"
                    r"(in|per|by|within|for)\s+(each\s+)?categor(y|ies)|"
                    r"categor(y|ies).*(best|top)[-\s]*(selling\s+)?products?",
                    re.I,
                ),
                """
                WITH product_units AS (
                    SELECT p.category AS category,
                           p.id AS product_id,
                           p.name AS name,
                           SUM(oi.quantity) AS units_sold
                    FROM products p
                    JOIN order_items oi ON oi.product_id = p.id
                    GROUP BY p.id
                ),
                ranked AS (
                    SELECT category,
                           name,
                           units_sold,
                           ROW_NUMBER() OVER (
                               PARTITION BY category
                               ORDER BY units_sold DESC, product_id
                           ) AS rn
                    FROM product_units
                )
                SELECT category, name, units_sold
                FROM ranked
                WHERE rn = 1
                ORDER BY category
                """,
            ),
            (
                re.compile(r"(best|top).*selling product", re.I),
                """
                SELECT p.name,
                       SUM(oi.quantity) AS units_sold
                FROM products p
                JOIN order_items oi ON oi.product_id = p.id
                GROUP BY p.id
                ORDER BY units_sold DESC
                LIMIT 1
                """,
            ),
            # "Top products by revenue" ranks products by sales value
            # (quantity * unit_price), which is distinct from the units-sold
            # ranking above: a cheap high-volume item can top units-sold while a
            # pricier item tops revenue. Requiring the word "revenue" keeps this
            # rule from shadowing the best-selling (units) rule.
            (
                re.compile(
                    r"top.*products?.*revenue|products?.*by revenue|"
                    r"highest[- ]revenue products?",
                    re.I,
                ),
                """
                SELECT p.name,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                FROM products p
                JOIN order_items oi ON oi.product_id = p.id
                GROUP BY p.id
                ORDER BY revenue DESC
                LIMIT 5
                """,
            ),
            # Products frequently bought together (market-basket / affinity
            # analysis): the pairs of products that co-occur in the most orders.
            # This is the only rule that *self-joins* a table -- order_items to
            # itself on the same order_id -- so each matched row is two line
            # items from one order. The join condition oi1.product_id <
            # oi2.product_id does two jobs at once: it drops the trivial
            # self-pairing of a line item with itself, and it emits each unordered
            # pair {A, B} only once (as A,B, never also B,A), so pairs are not
            # double-counted. COUNT(DISTINCT oi1.order_id) counts *orders*
            # containing both products rather than raw joined rows, which keeps
            # the tally correct even when an order lists the same product on more
            # than one line item. product_a/product_b are added to the ORDER BY
            # after the co-occurrence count so ties resolve the same way on every
            # run. The matcher owns the "bought/purchased together", "market
            # basket", and "product pairs/affinity" phrasings, none of which any
            # other rule uses, so it neither shadows nor is shadowed by the
            # product- or basket-size rules.
            (
                re.compile(
                    r"bought\s+together|purchased\s+together|"
                    r"market[-\s]basket|product\s+(pairs|affinity)|"
                    r"frequently\s+bought",
                    re.I,
                ),
                """
                SELECT p1.name AS product_a,
                       p2.name AS product_b,
                       COUNT(DISTINCT oi1.order_id) AS orders_together
                FROM order_items oi1
                JOIN order_items oi2
                     ON oi2.order_id = oi1.order_id
                    AND oi1.product_id < oi2.product_id
                JOIN products p1 ON p1.id = oi1.product_id
                JOIN products p2 ON p2.id = oi2.product_id
                GROUP BY p1.name, p2.name
                ORDER BY orders_together DESC, product_a, product_b
                LIMIT 5
                """,
            ),
            # Average customer lifespan: the mean number of days between a
            # customer's first and last order. A first CTE collapses the orders
            # table to one row per customer whose active_days is
            # julianday(MAX(order_date)) - julianday(MIN(order_date)) -- julianday
            # converts each date to a fractional day number so the two can be
            # subtracted, which is how date differences are taken in SQLite (there
            # is no DATEDIFF). This is the only rule that does date arithmetic
            # rather than date *formatting* (the strftime rules bucket by month,
            # quarter, or weekday; this one measures an elapsed span). The outer
            # query then AVGs those per-customer spans -- a nested aggregate: MIN
            # and MAX run per customer inside the CTE, AVG runs across customers
            # outside it, which is why the CTE is needed rather than a single flat
            # query. A customer with exactly one order has first == last and so
            # contributes a span of 0, correctly pulling the average toward the
            # behaviour of one-time buyers. The matcher owns the lifespan/tenure
            # phrasings, which no other rule uses, so it neither shadows nor is
            # shadowed by the customer-count or repeat-customer rules.
            (
                re.compile(
                    r"customer\s+(lifespan|tenure|lifetime)|"
                    r"(lifespan|tenure)\s+of\s+(a\s+|the\s+)?customers?|"
                    r"how\s+long\s+.*customers?\s+(stay|remain|are)\s+active|"
                    r"average\s+(active\s+)?(customer\s+)?lifespan|"
                    r"days\s+between\s+(a\s+)?customers?['’]?s?\s+"
                    r"first\s+and\s+last\s+order",
                    re.I,
                ),
                """
                WITH customer_span AS (
                    SELECT customer_id,
                           julianday(MAX(order_date))
                               - julianday(MIN(order_date)) AS active_days
                    FROM orders
                    GROUP BY customer_id
                )
                SELECT ROUND(AVG(active_days), 1) AS avg_customer_lifespan_days
                FROM customer_span
                """,
            ),
            # At-risk (lapsed) customers: those who have bought before but whose
            # most recent order is now old -- the "Recency" lens of RFM analysis.
            # A first CTE collapses the orders table to one row per customer
            # carrying MAX(order_date) (their latest order); the outer query then
            # keeps only customers whose latest order predates a cutoff 90 days
            # before the newest order in the data. Two deliberate design choices,
            # noted so they can be defended:
            #   * The cutoff is anchored to the dataset's own MAX(order_date), not
            #     the wall-clock today(), so the answer is reproducible for anyone
            #     who clones the repo -- the same reasoning the "orders in the last
            #     30 days" rule uses. This rule reuses that date(..., '-N day')
            #     idiom but applies it as a per-customer recency filter rather than
            #     a single global count.
            #   * The JOIN to orders is inner, so only customers with at least one
            #     order are considered. A customer who never ordered is an
            #     acquisition question, not a churn one; "at-risk" implies a prior
            #     relationship that has since gone quiet.
            # The matcher owns the churn/lapsed/inactive/at-risk and "customers
            # haven't ordered" phrasings, none of which other rules use, so it
            # neither shadows nor is shadowed by the repeat-customer or
            # recent-orders rules. It is placed ahead of the broad aggregate rules
            # for the same first-rule-wins reason as the other specific rules.
            (
                re.compile(
                    r"at[-\s]?risk\s+customers?|"
                    r"churn(?:ed|ing)?\s+customers?|"
                    r"lapsed\s+customers?|"
                    r"inactive\s+customers?|"
                    r"customers?\s+(?:who\s+)?"
                    r"(?:haven'?t|hasn'?t|have\s+not|has\s+not|not)\s+"
                    r"(?:placed\s+an?\s+order|ordered)|"
                    r"customers?\b.*\bno\s+orders?\s+in\s+the\s+last",
                    re.I,
                ),
                """
                WITH customer_last_order AS (
                    SELECT c.id AS customer_id,
                           c.name AS name,
                           MAX(o.order_date) AS last_order_date
                    FROM customers c
                    JOIN orders o ON o.customer_id = c.id
                    GROUP BY c.id
                )
                SELECT customer_id, name, last_order_date
                FROM customer_last_order
                WHERE last_order_date
                      < date((SELECT MAX(order_date) FROM orders), '-90 day')
                ORDER BY last_order_date, customer_id
                """,
            ),
            # Revenue split between orders from *new* customers (their very first
            # order) and *returning* customers (every order after the first) --
            # the standard new-vs-returning revenue attribution. A first CTE rolls
            # each order up to its total, carrying the customer and order date; the
            # second labels every order by whether it is that customer's first with
            # ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date, order_id)
            # == 1 -> 'new', else 'returning'. This reuses the same per-customer
            # ROW_NUMBER idiom as the top-spender-per-region rule but to a different
            # end: that rule *keeps* rank 1 to pick a single winner per group, while
            # this one *labels* rank 1 vs the rest and keeps every order, so the two
            # revenue buckets together sum to total revenue. order_id is a
            # deterministic tiebreaker so two orders a customer placed on the same
            # day resolve the same way on every run (only one can be the 'new' one).
            # The matcher owns the new-vs-returning / first-time-vs-repeat contrast
            # phrasings and is registered ahead of the plain "(repeat|returning)
            # customers" count rule below so a question naming "returning customers"
            # in a revenue-split context is not shadowed by that simpler counter.
            (
                re.compile(
                    r"new\s+(?:vs\.?|versus|and|or|&)\s+returning|"
                    r"returning\s+(?:vs\.?|versus|and|or|&)\s+new|"
                    r"first[-\s]?time\s+(?:vs\.?|versus|and|or|&)\s+(?:returning|repeat)|"
                    r"(?:revenue|sales|spend)\s+from\s+new\s+"
                    r"(?:and|vs\.?|versus|&)\s+returning",
                    re.I,
                ),
                """
                WITH order_totals AS (
                    SELECT o.id AS order_id,
                           o.customer_id AS customer_id,
                           o.order_date AS order_date,
                           SUM(oi.quantity * oi.unit_price) AS order_total
                    FROM orders o
                    JOIN order_items oi ON oi.order_id = o.id
                    GROUP BY o.id
                ),
                classified AS (
                    SELECT order_total,
                           CASE
                               WHEN ROW_NUMBER() OVER (
                                        PARTITION BY customer_id
                                        ORDER BY order_date, order_id
                                    ) = 1
                               THEN 'new'
                               ELSE 'returning'
                           END AS customer_type
                    FROM order_totals
                )
                SELECT customer_type,
                       COUNT(*) AS orders,
                       ROUND(SUM(order_total), 2) AS revenue
                FROM classified
                GROUP BY customer_type
                ORDER BY customer_type
                """,
            ),
            (
                re.compile(r"(repeat|returning) customers", re.I),
                """
                SELECT COUNT(*) AS repeat_customers
                FROM (
                    SELECT customer_id
                    FROM orders
                    GROUP BY customer_id
                    HAVING COUNT(*) > 1
                )
                """,
            ),
            (
                re.compile(r"orders.*last (30|thirty) days", re.I),
                """
                SELECT COUNT(*) AS recent_orders
                FROM orders
                WHERE order_date >= date((SELECT MAX(order_date) FROM orders), '-30 day')
                """,
            ),
            (
                re.compile(r"(revenue|sales).*by region", re.I),
                """
                SELECT c.region,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                FROM customers c
                JOIN orders o ON o.customer_id = c.id
                JOIN order_items oi ON oi.order_id = o.id
                GROUP BY c.region
                ORDER BY revenue DESC
                """,
            ),
            (
                re.compile(r"(monthly )?new customers.*(by month|2024)", re.I),
                """
                SELECT strftime('%Y-%m', signup_date) AS month,
                       COUNT(*) AS new_customers
                FROM customers
                WHERE signup_date >= '2024-01-01' AND signup_date < '2025-01-01'
                GROUP BY month
                ORDER BY month
                """,
            ),
            # Unique (active) customers per month -- the "monthly active buyers"
            # metric: how many *distinct* customers placed at least one order in
            # each month, as opposed to the raw order count. COUNT(DISTINCT
            # customer_id) collapses a customer's multiple orders in a month down
            # to one, so a repeat buyer is counted once per month. This is the
            # only rule that uses COUNT(DISTINCT ...). It requires a distinctness
            # word (unique/distinct/active) plus a customer/buyer word, so it does
            # not shadow the plain "how many customers" count or the "new
            # customers by month" signup rule above it (which counts signups, not
            # buyers).
            (
                re.compile(
                    r"(unique|distinct|active)\s+(customers?|buyers?|shoppers?)"
                    r".*(month|2024)|"
                    r"monthly.*(unique|distinct|active)\s+(customers?|buyers?)",
                    re.I,
                ),
                """
                SELECT strftime('%Y-%m', o.order_date) AS month,
                       COUNT(DISTINCT o.customer_id) AS unique_customers
                FROM orders o
                WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'
                GROUP BY month
                ORDER BY month
                """,
            ),
            # Revenue bucketed by calendar quarter for 2024. SQLite has no
            # quarter function, so the quarter number is derived from the month
            # with integer arithmetic: (month + 2) / 3 maps months 1-3 -> 1,
            # 4-6 -> 2, 7-9 -> 3, 10-12 -> 4. Kept with the other time-series
            # rules and ahead of the broad "total revenue" rule below so that
            # quarterly phrasings are not shadowed by it.
            (
                re.compile(
                    r"(revenue|sales).*(by|per).*quarter|quarterly (revenue|sales)",
                    re.I,
                ),
                """
                SELECT '2024-Q' || (
                           (CAST(strftime('%m', o.order_date) AS INTEGER) + 2) / 3
                       ) AS quarter,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                FROM orders o
                JOIN order_items oi ON oi.order_id = o.id
                WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'
                GROUP BY quarter
                ORDER BY quarter
                """,
            ),
            # Revenue split by day of the week (Sunday..Saturday). SQLite's
            # strftime('%w', ...) returns the weekday as a digit 0-6 with
            # Sunday == 0; a CASE expression turns that digit into a readable
            # name. Grouping and ordering use the numeric weekday (not the name)
            # so the rows come back in calendar order rather than alphabetically.
            # Kept with the other time-series rules and ahead of the broad
            # "total revenue" rule so day-of-week phrasings are not shadowed.
            (
                re.compile(
                    r"(revenue|sales).*(day of (the )?week|day[- ]of[- ]week|"
                    r"weekday|days? of the week)|by weekday",
                    re.I,
                ),
                """
                SELECT CASE CAST(strftime('%w', o.order_date) AS INTEGER)
                           WHEN 0 THEN 'Sunday'
                           WHEN 1 THEN 'Monday'
                           WHEN 2 THEN 'Tuesday'
                           WHEN 3 THEN 'Wednesday'
                           WHEN 4 THEN 'Thursday'
                           WHEN 5 THEN 'Friday'
                           WHEN 6 THEN 'Saturday'
                       END AS weekday,
                       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
                FROM orders o
                JOIN order_items oi ON oi.order_id = o.id
                GROUP BY CAST(strftime('%w', o.order_date) AS INTEGER)
                ORDER BY CAST(strftime('%w', o.order_date) AS INTEGER)
                """,
            ),
            # Average time between a customer's consecutive orders (purchase
            # cadence): a repeat-purchase engagement metric. The single CTE lines
            # every order up next to that same customer's previous order using
            # LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date).
            # This is the only rule with a *partitioned* window: the earlier LAG
            # rule (month-over-month) has one unpartitioned series, whereas here
            # PARTITION BY restarts the LAG for each customer so gaps never span
            # two different customers. The order_date difference is taken in days
            # via julianday(), which converts an ISO date to a Julian day number
            # so the subtraction yields whole days. A customer's first order has
            # no prior order, so its LAG is NULL and gap_days is NULL; the outer
            # WHERE drops those rows, leaving one gap per repeat purchase, and AVG
            # is the mean of all such gaps. The ORDER BY breaks ties on o.id so
            # two orders a customer placed on the same day resolve identically on
            # every run (their gap is 0). The matcher owns the "time/days/gap
            # between orders" and "purchase/reorder cadence|frequency" phrasings,
            # none of which any other rule uses, so it neither shadows nor is
            # shadowed by the customer-lifespan or broad order-count rules.
            (
                re.compile(
                    r"(time|days?|gap|interval)\s+between\s+(orders|purchases)|"
                    r"between\s+(consecutive\s+)?(orders|purchases)|"
                    r"(order|purchase|reorder|repurchase)\s+"
                    r"(cadence|frequency|interval)|"
                    r"how\s+(often|frequently)\s+.*(order|purchase)",
                    re.I,
                ),
                """
                WITH order_gaps AS (
                    SELECT o.customer_id,
                           julianday(o.order_date) - julianday(
                               LAG(o.order_date) OVER (
                                   PARTITION BY o.customer_id
                                   ORDER BY o.order_date, o.id
                               )
                           ) AS gap_days
                    FROM orders o
                )
                SELECT ROUND(AVG(gap_days), 1) AS avg_days_between_orders
                FROM order_gaps
                WHERE gap_days IS NOT NULL
                """,
            ),
            # Distribution of orders per customer: a purchase-frequency histogram
            # and the foundation of frequency-based (RFM) segmentation. This is a
            # *nested aggregation* -- the only rule that groups by the output of a
            # prior GROUP BY. The inner CTE collapses the orders table to one row
            # per customer carrying that customer's order count; the outer query
            # then groups those per-customer counts, so each result row reads
            # "this many customers placed exactly this many orders." Ordering by
            # order_count returns the histogram buckets low-to-high. It is a
            # distribution, not the single-number cadence metric owned by the
            # between-orders rule (which this does not use "frequency" wording to
            # avoid), and it is registered ahead of the broad order-count rule
            # below so "orders per customer" phrasings are not shadowed by it.
            (
                re.compile(
                    r"distribution\s+of\s+(the\s+)?(number\s+of\s+)?orders|"
                    r"orders?\s+per\s+customer|"
                    r"(number|count)\s+of\s+orders\s+per\s+customer|"
                    r"how\s+many\s+orders\s+(do|does)\s+(each|every|a)\s+customers?|"
                    r"order[-\s]?count\s+(distribution|histogram|breakdown)|"
                    r"(distribution|histogram|breakdown)\s+of\s+order\s+counts?",
                    re.I,
                ),
                """
                WITH orders_per_customer AS (
                    SELECT customer_id, COUNT(*) AS order_count
                    FROM orders
                    GROUP BY customer_id
                )
                SELECT order_count,
                       COUNT(*) AS customers
                FROM orders_per_customer
                GROUP BY order_count
                ORDER BY order_count
                """,
            ),
            # The rules below are intentionally placed last. Matching is
            # first-rule-wins, so these broad aggregate phrasings ("how many
            # orders") do not shadow the more specific rules above (e.g. orders
            # in the last 30 days), which should take precedence.
            (
                re.compile(r"total revenue|overall revenue|how much revenue", re.I),
                """
                SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue
                FROM order_items oi
                """,
            ),
            (
                re.compile(r"how many orders|number of orders|total orders|order count", re.I),
                "SELECT COUNT(*) AS order_count FROM orders",
            ),
            (
                re.compile(
                    r"how many products|number of products|product count|"
                    r"products in (the )?catalog",
                    re.I,
                ),
                "SELECT COUNT(*) AS product_count FROM products",
            ),
        ]

    def rule_count(self) -> int:
        """Return the number of question patterns registered in the catalog."""
        return len(self._rules)

    def rule_pattern(self, index: int) -> str:
        """Return the regex source of the rule at ``index``.

        Used to name a rule in a test failure message; a pattern is far more
        recognizable than a bare index when a catalog invariant breaks.
        """
        return self._rules[index][0].pattern

    def matching_rule_indexes(self, question: str) -> list[int]:
        """Return the index of every rule whose matcher matches ``question``.

        Resolution is first-rule-wins, so only ``[0]`` of this list decides the
        SQL. The rest is what makes the catalog's ordering auditable: a rule
        that matches but never wins is shadowed by a broader rule registered
        ahead of it, and a rule that never appears first for any question is
        unreachable. ``tests/test_rule_catalog.py`` asserts both properties
        across the gold set.
        """
        return [
            index
            for index, (matcher, _) in enumerate(self._rules)
            if matcher.search(question)
        ]

    def to_sql(self, question: str, schema: str) -> str:  # noqa: ARG002
        # Deliberately reuses ``matching_rule_indexes`` instead of short-circuiting
        # on the first match: routing and the ordering diagnostic then share one
        # implementation and cannot drift apart. Scanning ~40 small regexes is not
        # a meaningful cost next to executing the query.
        matches = self.matching_rule_indexes(question)
        if not matches:
            raise ValueError(
                "Offline backend has no rule for this question. "
                "Set OPENAI_API_KEY and use --llm for open-ended questions."
            )
        _, sql = self._rules[matches[0]]
        return " ".join(sql.split())


# --------------------------------------------------------------------------- #
# LLM backend
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = """You are a careful analytics engineer. Given a SQLite schema
and a question, return a single read-only SQL query that answers it.

Rules:
- Output ONLY the SQL, no prose, no markdown fences.
- Use only SELECT (or WITH ... SELECT). Never modify data.
- Use the exact table and column names from the schema.
- Prefer explicit JOINs and clear column aliases.
"""


class LLMBackend:
    """OpenAI-compatible chat backend. Imports the client lazily."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        from openai import OpenAI  # lazy import; optional dependency

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def to_sql(self, question: str, schema: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
            ],
        )
        sql = resp.choices[0].message.content or ""
        return _strip_fences(sql).strip()


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:sql)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def get_backend(use_llm: bool) -> Backend:
    """Factory: pick the LLM backend when requested, else offline."""
    if use_llm:
        return LLMBackend()
    return OfflineBackend()
