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

    def to_sql(self, question: str, schema: str) -> str:  # noqa: ARG002
        for matcher, sql in self._rules:
            if matcher.search(question):
                return " ".join(sql.split())
        raise ValueError(
            "Offline backend has no rule for this question. "
            "Set OPENAI_API_KEY and use --llm for open-ended questions."
        )


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
