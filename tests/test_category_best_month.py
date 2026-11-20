"""The per-category best-and-worst-month rule: routing, guards and its numbers."""

from __future__ import annotations

import os
import sqlite3

import pytest

from nl2sql import generator
from nl2sql.llm import NoRuleMatchError, OfflineBackend

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")

#: A fragment unique to this rule's SQL, used to tell which rule answered.
_MARKER = "AS high_rank"


def _sql(question: str) -> str:
    return OfflineBackend().to_sql(question, schema="")


@pytest.mark.parametrize(
    "question",
    [
        "What was each category's best and worst month?",
        "Which month was best for each category?",
        "What was the worst month for each category?",
        "Show each category's peak month.",
        "Show the highest revenue month by category.",
    ],
)
def test_per_category_wordings_reach_the_rule(question: str) -> None:
    assert _MARKER in _sql(question)


@pytest.mark.parametrize(
    "question",
    [
        # Picks one category rather than asking for every category.
        "Which category had its best month?",
        # A year the SQL does not filter on.
        "What was each category's best month in 2023?",
        # A different grouping, and a different measure.
        "What was each region's best month?",
        "Which month had the most units sold in each category?",
        "Which month had the highest average order value by category?",
    ],
)
def test_questions_it_cannot_answer_do_not_reach_it(question: str) -> None:
    try:
        sql = _sql(question)
    except NoRuleMatchError:
        return
    assert _MARKER not in sql


def test_company_wide_best_month_is_not_taken_over() -> None:
    """Without a per-category phrase the company-wide rule must still answer."""
    sql = _sql("Which month had the highest revenue?")
    assert "'highest' AS label" in sql
    assert _MARKER not in sql


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_result_matches_an_independent_python_aggregation() -> None:
    """Recompute category-month revenue row by row in Python and pick both ends.

    Ties break on the earlier month, matching the rule's ORDER BY, so the
    comparison is exact rather than "one of the tied months".
    """
    conn = sqlite3.connect(DB)
    totals: dict[str, dict[str, float]] = {}
    for category, month, qty, price in conn.execute(
        "SELECT p.category, strftime('%Y-%m', o.order_date), oi.quantity, oi.unit_price "
        "FROM order_items oi JOIN orders o ON o.id = oi.order_id "
        "JOIN products p ON p.id = oi.product_id"
    ):
        months = totals.setdefault(category, {})
        months[month] = months.get(month, 0.0) + qty * price
    conn.close()

    expected = []
    for category in sorted(totals):
        months = totals[category]
        best = min(months, key=lambda m: (-months[m], m))
        worst = min(months, key=lambda m: (months[m], m))
        expected.append(
            (category, best, round(months[best], 2), worst, round(months[worst], 2))
        )

    ans = generator.answer_question(DB, "What was each category's best and worst month?")
    assert ans.result.columns == [
        "category",
        "best_month",
        "best_revenue",
        "worst_month",
        "worst_revenue",
    ]
    assert [tuple(r) for r in ans.result.rows] == expected
    # The two ends are distinct months for every category on the sample data,
    # so a rule that returned the same month twice would fail here.
    assert all(row[1] != row[3] for row in expected)
