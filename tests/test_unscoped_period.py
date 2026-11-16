"""Rules with no date filter must not answer a question about a period.

A rule whose SQL never references a date column computes its answer over every
order (or customer, or product) on record. If its pattern also accepts "in
2023" or "in the last 90 days", the user gets the all-time figure under a
period-scoped question: a plausible, correctly labelled number that silently
ignores the window. ``_UNSCOPED_ONLY`` in ``nl2sql/rules.py`` is the guard for
this, but each rule has to opt in to it, so nothing stops a new unscoped rule
from shipping without it.

This file closes that gap the way ``test_pinned_year.py`` does for the
year-pinned rules: it derives the unscoped rules from their SQL rather than from
a hand-kept list, rewrites each one's gold question to name a period, and runs
the result through the offline backend end to end. Running the whole backend,
rather than the winning rule's regex alone, also catches the fall-through case:
a guarded rule that declines is no help if a broader unguarded rule further
down the catalog then accepts the same question.

Most unscoped rules do not carry the guard yet. They are recorded in
:data:`KNOWN_PERIOD_GAPS` by their gold question, as a ratchet in the same
style as ``ACCEPTED_SHADOWING`` in ``test_shadowing.py``: the test fails if a
question *not* on the list starts accepting a period (a regression, or a new
rule added without the guard), and it also fails if a question *on* the list
starts declining (a fix, which should shrink the list so the gain is recorded).
Unlike the shadowing baseline, every entry here is a real defect queued for a
fix, not an accepted design choice.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nl2sql.llm import NoRuleMatchError, OfflineBackend
from nl2sql.rules import _PERIOD_SCOPED, build_catalog

GOLD = Path(__file__).resolve().parent.parent / "evals" / "gold.jsonl"

#: The only date columns in the sample schema (``orders.order_date`` and
#: ``customers.signup_date``). A rule whose SQL names neither cannot be
#: honouring a period, whatever its pattern accepts.
_DATE_COLUMN = re.compile(r"\b(?:order_date|signup_date)\b")

#: One explicit calendar year and two relative windows: the three shapes
#: ``_PERIOD_SCOPED`` is written to recognise.
PERIODS = ("in 2023", "in the last 90 days", "this year")

#: Gold questions whose answering rule has no date filter *and* still accepts a
#: period-scoped rephrasing, so the user gets an all-time answer to a question
#: about a period. Each is a defect; remove an entry when its rule gains the
#: ``_UNSCOPED_ONLY`` guard (the stale-entry test below will insist on it).
KNOWN_PERIOD_GAPS: frozenset[str] = frozenset(
    {
        # Breakdowns and leaderboards over orders: an all-time split or ranking.
        "Show revenue by region and category.",
        "Show revenue by category with a total row.",
        "Show revenue by price tier.",
        "What percentage of revenue comes from each category?",
        "Which categories have above-average revenue?",
        "How many units were sold in each category?",
        "How many orders were placed in each region?",
        "What is the best-selling product in each category?",
        "What is the best-selling product in each region?",
        "What are the bottom 5 products by revenue?",
        "Who is the top-spending customer in each region?",
        "Show the repeat purchase rate by product.",
        "Which products are most frequently bought together?",
        # Customer-level statistics: lifetime spend and behaviour.
        "How many repeat customers are there?",
        "What is the average revenue per customer?",
        "What is the average revenue per customer in each region?",
        "Segment customers into spend quartiles.",
        "What share of revenue comes from the top 20% of customers?",
        "Show the distribution of orders per customer.",
        "What share of customers bought from each category?",
        "How many different categories does each customer buy from?",
        "How many customers are in each region?",
        # Catalog snapshots over ``products``: no history to scope at all, but
        # accepting "in 2023" still answers a question the data cannot.
        "How many products are in each category?",
        "Which products are priced above the average for their category?",
    }
)


def _gold_questions() -> list[str]:
    with GOLD.open(encoding="utf-8") as fh:
        return [json.loads(line)["question"] for line in fh if line.strip()]


def _unscoped_rule_indexes() -> set[int]:
    return {i for i, (_, sql) in enumerate(build_catalog()) if not _DATE_COLUMN.search(sql)}


def _unscoped_gold_questions() -> list[str]:
    """Gold questions that an unscoped rule answers, in gold-file order."""
    backend = OfflineBackend()
    unscoped = _unscoped_rule_indexes()
    return [
        question
        for question in _gold_questions()
        if (indexes := backend.matching_rule_indexes(question)) and indexes[0] in unscoped
    ]


def _rescoped(question: str, period: str) -> str:
    """Append *period*: "Show revenue by region" -> "Show revenue by region in 2023?"."""
    return question.rstrip("?.") + f" {period}?"


def _accepted_periods(question: str) -> list[str]:
    """The periods from :data:`PERIODS` that the backend answers rather than declines."""
    backend = OfflineBackend()
    accepted = []
    for period in PERIODS:
        try:
            backend.to_sql(_rescoped(question, period), schema="")
        except NoRuleMatchError:
            continue
        accepted.append(period)
    return accepted


CASES = _unscoped_gold_questions()


def test_unscoped_rules_are_found_and_each_has_a_gold_question() -> None:
    # Keeps the parametrised test from passing vacuously: if the SQL heuristic
    # stopped finding rules, or an unscoped rule had no gold question to
    # rescope, nothing would be checked and the suite would stay green.
    backend = OfflineBackend()
    answered = {backend.matching_rule_indexes(q)[0] for q in CASES}
    assert CASES
    assert answered == _unscoped_rule_indexes()


def test_unscoped_gold_questions_name_no_period_of_their_own() -> None:
    # Appending "in 2023" to a question that already says "this year" would
    # produce a contradiction rather than a period-scoped rephrasing.
    scoped = [q for q in CASES if re.search(_PERIOD_SCOPED, q, re.I)]
    assert scoped == []


def test_every_known_gap_is_still_a_gold_question() -> None:
    # An entry whose question was reworded or removed from gold would otherwise
    # sit in the list forever, excusing nothing.
    assert KNOWN_PERIOD_GAPS <= set(CASES)


@pytest.mark.parametrize("question", CASES)
def test_unscoped_rule_declines_a_period_unless_a_known_gap(question: str) -> None:
    accepted = _accepted_periods(question)
    if question in KNOWN_PERIOD_GAPS:
        assert accepted, (
            f"{question!r} now declines every period; remove it from "
            "KNOWN_PERIOD_GAPS so the fix is recorded."
        )
    else:
        assert accepted == [], (
            f"{question!r} is answered by a rule with no date filter, yet "
            f"accepts {accepted}; prefix its pattern with _UNSCOPED_ONLY."
        )


@pytest.mark.parametrize(
    "question",
    [
        # One guarded rule from each family the guard already covers.
        "What is the total revenue?",
        "How many orders do we have?",
        "Show revenue by category",
        "What is the average order value?",
        # The order-level statistics, guarded as one batch.
        "What is the average order value by region?",
        "What is the median order value?",
        "What is the average number of items per order?",
        "Show the distribution of order values.",
        "How many orders contain products from more than one category?",
    ],
)
def test_guarded_rules_decline_every_period_end_to_end(question: str) -> None:
    """The guard holds through fall-through, not only on the winning rule."""
    assert question not in KNOWN_PERIOD_GAPS
    assert _accepted_periods(question) == []
