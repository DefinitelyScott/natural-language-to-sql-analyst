"""Tests for the precision guard — the questions the offline catalog must decline.

Execution accuracy over ``evals/gold.jsonl`` measures recall only: of the
questions the catalog claims to cover, how many it answers correctly. It is
blind to the opposite defect. Rules are matched first-match over an ordered list
of regexes, so broadening one to catch a phrasing variant can also make it swallow
a question it was never meant to answer — and because that question is not in the
gold set, no accuracy number moves when it happens. The failure mode is worse
than a crash: the user gets a well-formed table, correctly labelled, answering a
different question.

``evals/precision.jsonl`` is the counterweight: questions this database has no
answer for, which the backend must therefore route to no rule at all. These
tests cover three things —

1. the loader rejects a malformed guard record rather than skipping it;
2. the checker reports *which* rule matched, not merely that one did;
3. the shipped guard set actually passes against the live catalog.

The third is the regression test. The first two exist so that a failure of the
third is readable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Same import shim as tests/test_evaluate.py: evals/ is a script directory, not
# an installed package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "evals"))

import evaluate  # noqa: E402

from nl2sql.llm import OfflineBackend  # noqa: E402


@pytest.fixture(scope="module")
def backend() -> OfflineBackend:
    return OfflineBackend()


@pytest.fixture(scope="module")
def guards() -> list[dict]:
    return evaluate.load_guard_set(evaluate.PRECISION_PATH)


# --------------------------------------------------------------------------- #
# The shipped guard set
# --------------------------------------------------------------------------- #
def test_no_rule_matches_any_guard_question(
    backend: OfflineBackend, guards: list[dict]
) -> None:
    """The catalog must decline every guard question.

    A failure here names the rule and the reason the question is unanswerable,
    because "some rule matched" is not enough to act on — the fix is always to
    narrow one specific pattern.
    """
    failures = [
        f"{result.question!r} -> rule #{result.matched_rule} "
        f"({result.matched_pattern}); should decline because {result.reason}"
        for result in evaluate.check_precision(backend, guards)
        if not result.passed
    ]
    assert not failures, (
        "offline rules matched questions this database cannot answer: " f"{failures}"
    )


def test_guard_set_is_not_empty(guards: list[dict]) -> None:
    """A guard file emptied by an editing accident would pass silently."""
    assert len(guards) >= 10


def test_guard_questions_are_unique(guards: list[dict]) -> None:
    """Duplicates inflate the guard count without adding coverage."""
    questions = [record["question"] for record in guards]
    assert len(questions) == len(set(questions))


def test_guard_questions_are_not_gold_questions(guards: list[dict]) -> None:
    """The two sets state opposite requirements, so no question may be in both.

    A question in both files asserts that the catalog must answer it correctly
    *and* must not answer it at all. One of the two checks would then always
    fail, and which one is a matter of which file was edited last.
    """
    gold_path = Path(evaluate.PRECISION_PATH).parent / "gold.jsonl"
    gold = {
        json.loads(line)["question"]
        for line in gold_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    overlap = gold & {record["question"] for record in guards}
    assert not overlap, f"questions in both the gold and guard sets: {overlap}"


def test_period_scoped_counts_are_declined(backend: OfflineBackend) -> None:
    """The specific defect the ``_UNSCOPED_ONLY`` guard was added for.

    Each of these once matched a broad, table-wide count rule and was answered
    with the all-time total, silently dropping the period. Asserting them
    directly — rather than relying on their presence in the guard file — pins
    the behaviour even if the guard set is later re-curated.
    """
    for question in (
        "How many orders were placed in the last 7 days?",
        "How many customers churned last month?",
        "What was total revenue in 2023?",
        "How many products were added last month?",
    ):
        assert not backend.matching_rule_indexes(question), (
            f"{question!r} routes to a rule that ignores the period it names"
        )


def test_supported_periods_still_route(backend: OfflineBackend) -> None:
    """The guard must not cost the catalog the windows it does implement.

    These questions carry a period too; they keep working because the rules that
    honour them are registered ahead of the broad ones and win on first match.
    Without this test the guard could be tightened into a regression that only
    the eval harness would notice.
    """
    for question in (
        "How many orders were placed in the last 30 days?",
        "Which customers haven't ordered in the last 90 days?",
        "What were total sales by month in 2024?",
    ):
        assert backend.matching_rule_indexes(question), (
            f"{question!r} no longer matches any rule"
        )


def test_unscoped_phrasings_still_route(backend: OfflineBackend) -> None:
    """The guarded rules must still answer everything they used to."""
    for question, expected in (
        ("How many orders do we have?", "orders"),
        ("How many customers do we have?", "customers"),
        ("How many products are in the catalog?", "products"),
        ("What is the total revenue?", "order_items"),
    ):
        sql = backend.to_sql(question, "")
        assert expected in sql, f"{question!r} produced unexpected SQL: {sql}"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def test_loader_returns_question_and_reason(guards: list[dict]) -> None:
    assert all(record["question"] and record["reason"] for record in guards)


@pytest.mark.parametrize(
    "line",
    [
        '{"reason": "no such data"}',
        '{"question": "Show profit."}',
        '{"question": "Show profit.", "reason": ""}',
        '{"question": "Show profit.", "reason": 7}',
    ],
)
def test_loader_rejects_incomplete_records(tmp_path: Path, line: str) -> None:
    """A guard without a stated reason is not a guard — it is an assertion nobody
    can evaluate later. Rejecting beats skipping: a skipped record is a check
    that stopped running while the file still looks like it covers the case."""
    path = tmp_path / "precision.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="guard record needs"):
        evaluate.load_guard_set(str(path))


def test_loader_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "precision.jsonl"
    path.write_text(
        '\n{"question": "Show profit.", "reason": "no cost data"}\n\n',
        encoding="utf-8",
    )
    assert len(evaluate.load_guard_set(str(path))) == 1


# --------------------------------------------------------------------------- #
# Checking and reporting
# --------------------------------------------------------------------------- #
def test_check_reports_the_matching_rule(backend: OfflineBackend) -> None:
    """A guard question that *does* match must carry the rule that matched it.

    Uses a question the catalog is supposed to answer, so the test exercises the
    failure path without depending on a real defect existing.
    """
    record = {"question": "How many customers do we have?", "reason": "deliberate"}
    (result,) = evaluate.check_precision(backend, [record])

    assert not result.passed
    assert result.matched_rule is not None
    assert result.matched_pattern == backend.rule_pattern(result.matched_rule)


def test_check_reports_the_first_matching_rule(backend: OfflineBackend) -> None:
    """The reported rule is the one that would answer, not just any that match.

    Routing is first-match, so naming a later, shadowed rule would send someone
    to narrow a pattern that never fires.
    """
    question = "How many orders contain products from more than one category?"
    matches = backend.matching_rule_indexes(question)
    assert len(matches) > 1, "fixture question no longer matches multiple rules"

    (result,) = evaluate.check_precision(backend, [{"question": question, "reason": "x"}])
    assert result.matched_rule == matches[0]


def test_format_reports_a_clean_run() -> None:
    """A clean run is one line — the headline, with nothing listed under it."""
    results = [evaluate.GuardResult(question="q", reason="r") for _ in range(3)]
    assert evaluate.format_precision(results) == (
        "Rule precision: 3 guard questions, 0 unexpected matches"
    )


def test_format_names_the_failure() -> None:
    passing = evaluate.GuardResult(question="a question that is declined", reason="x")
    results = [
        passing,
        evaluate.GuardResult(
            question="Show profit by product.",
            reason="no cost data exists",
            matched_rule=12,
            matched_pattern=r"profit|margin",
        ),
    ]
    text = evaluate.format_precision(results)

    assert "2 guard questions, 1 unexpected match" in text
    assert "Show profit by product." in text
    assert "rule #12" in text
    assert "no cost data exists" in text
    # A passing guard must not be listed: the block is a failure report, and
    # printing every clean check would bury the one that is not.
    assert passing.question not in text
