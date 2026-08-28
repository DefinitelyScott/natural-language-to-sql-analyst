"""Catalog-wide ordering invariants for the offline backend.

The offline backend resolves a question by scanning an ordered list of regex
rules and taking the first match. That design is simple and predictable, but it
has one failure mode: a rule registered *after* a broader rule that also matches
its questions can never win. The rule is then dead code — it still looks
implemented and still reads as covered in the README, but nothing routes to it,
and the evaluation harness cannot tell, because the broader rule happens to
return a plausible result for the question.

The existing tests in ``test_offline_backend.py`` guard this pairwise: each time
a pattern was added, a test asserted it did not shadow (or get shadowed by) the
specific neighbours it was likely to collide with. Those tests are precise but
opt-in — they only cover collisions someone anticipated.

These tests close that gap from the other direction, over the whole catalog at
once. They use ``evals/gold.jsonl`` as the set of questions the catalog claims
to answer, and assert that resolving every gold question exercises every rule
exactly once. Any new pattern that lands in an unreachable position fails here
without anyone having to guess which existing rule would swallow it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from nl2sql.llm import OfflineBackend

GOLD_PATH = Path(__file__).resolve().parent.parent / "evals" / "gold.jsonl"


@pytest.fixture(scope="module")
def gold_questions() -> list[str]:
    """Return the question text of every row in the gold evaluation set."""
    lines = GOLD_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["question"] for line in lines if line.strip()]


@pytest.fixture(scope="module")
def backend() -> OfflineBackend:
    return OfflineBackend()


@pytest.fixture(scope="module")
def winners(backend: OfflineBackend, gold_questions: list[str]) -> dict[int, list[str]]:
    """Map each rule index to the gold questions that resolve to it.

    A question with no matching rule is left out of the mapping entirely; that
    case is reported by its own test so the failure names the question rather
    than an absent rule.
    """
    by_rule: dict[int, list[str]] = defaultdict(list)
    for question in gold_questions:
        matches = backend.matching_rule_indexes(question)
        if matches:
            by_rule[matches[0]].append(question)
    return dict(by_rule)


def test_every_gold_question_matches_a_rule(
    backend: OfflineBackend, gold_questions: list[str]
) -> None:
    unmatched = [q for q in gold_questions if not backend.matching_rule_indexes(q)]
    assert not unmatched, (
        "gold questions with no offline rule (the harness would score these as "
        f"errors): {unmatched}"
    )


def test_every_rule_is_reachable_from_a_gold_question(
    backend: OfflineBackend, winners: dict[int, list[str]]
) -> None:
    """No rule may be unreachable — every pattern must win for some question.

    A rule that never wins is either untested by the eval harness or shadowed by
    a broader rule ahead of it. Both are defects: the catalog would be claiming
    coverage it does not actually deliver.
    """
    unreachable = [
        (index, backend.rule_pattern(index))
        for index in range(backend.rule_count())
        if index not in winners
    ]
    assert not unreachable, (
        "offline rules that no gold question resolves to — each is either "
        "shadowed by an earlier, broader rule or missing a gold row: "
        f"{unreachable}"
    )


def test_each_rule_is_exercised_by_exactly_one_gold_question(
    winners: dict[int, list[str]],
) -> None:
    """One gold question per rule keeps eval coverage one-to-one.

    Two gold questions landing on the same rule is the mirror image of an
    unreachable rule: it means a question intended for its own pattern is being
    answered by a neighbour's. Asserting it separately makes the failure name
    the colliding questions instead of the starved rule.
    """
    collisions = {index: qs for index, qs in winners.items() if len(qs) > 1}
    assert not collisions, (
        "gold questions sharing a single offline rule — one of them is being "
        f"answered by a pattern meant for the other: {collisions}"
    )


def test_shadowed_matches_are_reported_but_do_not_change_routing(
    backend: OfflineBackend,
) -> None:
    """``to_sql`` must follow ``matching_rule_indexes``' first entry.

    The diagnostic and the router share an implementation, and this pins that
    contract: a question matched by more than one rule is answered by the
    earliest, and the later matches are visible rather than silent.
    """
    question = "How many orders contain products from more than one category?"
    matches = backend.matching_rule_indexes(question)
    assert len(matches) > 1, (
        "expected this question to match both the multi-category rule and the "
        "broad order-count rule; the fixture question needs updating"
    )
    assert backend.to_sql(question, "") == " ".join(
        backend._rules[matches[0]][1].split()  # noqa: SLF001 - asserting routing
    )
    # The specific rule answers it: the broad rule would return a bare
    # COUNT(*) over orders with no category join at all.
    assert "category" in backend.to_sql(question, "")
