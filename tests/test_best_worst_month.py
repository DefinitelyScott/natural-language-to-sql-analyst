"""The best-and-worst-month rule: routing, guards and the numbers it returns."""

from __future__ import annotations

import os
import sqlite3

import pytest

from nl2sql import generator
from nl2sql.llm import NoRuleMatchError, OfflineBackend

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")


def _sql(question: str) -> str:
    return OfflineBackend().to_sql(question, schema="")


@pytest.mark.parametrize(
    "question",
    [
        "Which month had the highest revenue?",
        "What was our best month for sales?",
        "Which month was the worst for revenue?",
        "Which was the peak sales month?",
        "Which month had the lowest sales?",
    ],
)
def test_either_wording_reaches_the_rule(question: str) -> None:
    assert "'highest' AS label" in _sql(question)


@pytest.mark.parametrize(
    "question",
    [
        "Which month in 2024 had the highest revenue?",
        "Which category had its best month?",
        "Which region had the best month for sales?",
        "Which month had the highest average order value?",
        "What was revenue growth in the best month?",
    ],
)
def test_questions_it_cannot_answer_do_not_reach_it(question: str) -> None:
    try:
        sql = _sql(question)
    except NoRuleMatchError:
        return
    assert "'highest' AS label" not in sql


@pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")
def test_result_matches_an_independent_python_aggregation() -> None:
    """Recompute monthly revenue row by row in Python and compare both ends."""
    conn = sqlite3.connect(DB)
    totals: dict[str, float] = {}
    for month, qty, price in conn.execute(
        "SELECT strftime('%Y-%m', o.order_date), oi.quantity, oi.unit_price "
        "FROM orders o JOIN order_items oi ON oi.order_id = o.id"
    ):
        totals[month] = totals.get(month, 0.0) + qty * price
    conn.close()

    best = min(totals, key=lambda m: (-totals[m], m))
    worst = min(totals, key=lambda m: (totals[m], m))

    ans = generator.answer_question(DB, "Which month had the highest revenue?")
    rows = [tuple(r) for r in ans.result.rows]
    assert rows == [
        ("highest", best, round(totals[best], 2)),
        ("lowest", worst, round(totals[worst], 2)),
    ]
