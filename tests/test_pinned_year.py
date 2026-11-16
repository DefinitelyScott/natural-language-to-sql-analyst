"""Year-pinned rules must not answer a question about a different period.

Ten time-series rules hardcode one calendar year in their SQL
(``WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'``). Their
patterns used to accept any year, so "Show revenue by quarter in 2023." came
back as four rows labelled ``2024-Q1`` .. ``2024-Q4``: a well-formed answer
about the wrong year. ``_PINNED_YEAR_ONLY`` makes those rules decline any other
calendar year and any relative window.

These tests are derived from the catalog rather than from a hand-kept list: a
rule counts as year-pinned if its SQL contains the ``'<year>-01-01'`` literal,
and its own gold questions are rewritten to name another period. A new pinned
rule added without the guard therefore fails here with no edit to this file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nl2sql.llm import NoRuleMatchError, OfflineBackend
from nl2sql.rules import _PINNED_YEAR_ONLY, PINNED_YEAR, build_catalog

GOLD = Path(__file__).resolve().parent.parent / "evals" / "gold.jsonl"

#: Any calendar-year start literal in rule SQL, e.g. ``'2024-01-01'``.
_YEAR_START = re.compile(r"'((?:19|20)\d{2})-01-01'")

OTHER_YEAR = PINNED_YEAR - 1


def _gold_questions() -> list[str]:
    with GOLD.open(encoding="utf-8") as fh:
        return [json.loads(line)["question"] for line in fh if line.strip()]


def _pinned_rule_indexes() -> set[int]:
    return {
        i
        for i, (_, sql) in enumerate(build_catalog())
        if f"'{PINNED_YEAR}-01-01'" in sql
    }


def _rescoped(question: str, period: str) -> str:
    """Rewrite *question* to ask about *period* instead of the pinned year.

    "Show revenue by quarter in 2024." -> "Show revenue by quarter in 2023."
    A question that names no year gets the period appended instead.
    """
    if str(PINNED_YEAR) in question:
        return re.sub(rf"(?:in|of)\s+{PINNED_YEAR}", period, question)
    return question.rstrip("?.") + f" {period}?"


def _pinned_cases() -> list[tuple[int, str]]:
    backend = OfflineBackend()
    pinned = _pinned_rule_indexes()
    cases = []
    for question in _gold_questions():
        indexes = backend.matching_rule_indexes(question)
        if indexes and indexes[0] in pinned:
            cases.append((indexes[0], question))
    return cases


CASES = _pinned_cases()


def test_every_year_literal_in_the_catalog_is_the_pinned_year() -> None:
    # If a rule pinned to some other year appeared, PINNED_YEAR would no longer
    # describe the catalog and the guard would decline that rule's own year.
    # The following year's start also appears, as the exclusive upper bound
    # (``order_date < '2025-01-01'``), never as a range of its own.
    years = {
        int(year)
        for _, sql in build_catalog()
        for year in _YEAR_START.findall(sql)
    }
    assert years == {PINNED_YEAR, PINNED_YEAR + 1}


def test_every_pinned_rule_is_exercised_by_a_gold_question() -> None:
    # Keeps the parametrised test below from passing vacuously: a pinned rule
    # with no gold question would never be rescoped and never checked.
    assert {index for index, _ in CASES} == _pinned_rule_indexes()


@pytest.mark.parametrize(
    "period", [f"in {OTHER_YEAR}", "in the last 6 months", "this year"]
)
@pytest.mark.parametrize(("index", "question"), CASES, ids=[q for _, q in CASES])
def test_pinned_rule_declines_a_different_period(
    index: int, question: str, period: str
) -> None:
    rescoped = _rescoped(question, period)
    assert rescoped != question
    with pytest.raises(NoRuleMatchError):
        OfflineBackend().to_sql(rescoped, schema="")


@pytest.mark.parametrize(
    ("question", "allowed"),
    [
        ("Show revenue by quarter.", True),
        (f"Show revenue by quarter in {PINNED_YEAR}.", True),
        (f"Show revenue by quarter in {OTHER_YEAR}.", False),
        (f"Compare {OTHER_YEAR} and {PINNED_YEAR} revenue by quarter.", False),
        ("Show revenue by quarter for the past two years.", False),
        ("Show revenue by quarter this year.", False),
        # Digits that are not a year must not trip the guard.
        (f"Show the top 5 products by revenue in {PINNED_YEAR}.", True),
        ("Show revenue for order 20231.", True),
    ],
)
def test_guard_declines_only_periods_other_than_the_pinned_year(
    question: str, allowed: bool
) -> None:
    assert bool(re.match(_PINNED_YEAR_ONLY, question, re.I)) is allowed
