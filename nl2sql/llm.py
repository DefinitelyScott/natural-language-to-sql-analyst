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
