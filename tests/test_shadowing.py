"""Keep the catalog's ordering discipline from decaying silently.

The offline backend routes by first match over an ordered list of regexes. When
a question matches only one rule, order is irrelevant and the routing is a fact
about the pattern. When it matches several, the answer is decided by which of
them was registered first — and every other check in the harness is blind to
that. Execution accuracy compares the winner's output against gold and never
looks at who else was in the running; the precision guard only examines
questions that must match *nothing*; the paraphrase set only asks whether two
phrasings agree with each other, which they do whether or not a third rule is
quietly one line away from taking both.

Eleven of the fifty-nine gold questions are decided this way today, and that is
not in itself a defect. In every one of them the answering rule's pattern is
strictly more demanding than the patterns behind it — "average order value *by
region*" ahead of "average order value", "revenue by category *with a total
row*" ahead of "revenue by category" — which is the ordering the catalog is
supposed to have. Specific before general is the whole reason first-match
resolution works at all.

What is a defect is a *new* one appearing without anyone noticing, because the
new one is exactly as invisible as these eleven were. The usual way it happens is
a rule widened to catch a phrasing variant, which also widens it across some
older rule's question; nothing fails, the older question keeps working only
because of where the two sit, and the dependency is discovered later by someone
reordering the catalog for an unrelated reason. So the current set is recorded
below by name. :func:`test_no_new_shadowing` rejects a claimant that is not on
the record, and :func:`test_accepted_shadowing_has_no_stale_entries` rejects one
left on the record after the overlap is gone — together pinning the record to
the catalog, while keeping each failure message about a single direction.

The record names the losing rules by their own gold questions rather than by
index, so inserting a rule does not invalidate it — an index shifts, a question
does not.

Two of the eleven deserve a second look, and are called out in
:data:`SHADOWING_RATIONALE`: for most of the set a mis-ordering would return a
*coarser* answer to the question that was asked, which is wrong but recognisably
related to it, while for "new vs returning revenue" and "share of customers by
category" the rule behind answers a different question entirely — a customer
count, or a revenue mix — in a well-formed table with plausible column names.
Those are the two where an ordering mistake would be hardest to catch by
reading the output.

The harness reports the same measurement without gating on it — ``evaluate.py``
measures and prints, this file decides what is allowed to change.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "evals"))

import evaluate  # noqa: E402

from nl2sql import llm  # noqa: E402

GOLD_PATH = REPO_ROOT / "evals" / "gold.jsonl"
README_PATH = REPO_ROOT / "README.md"

# "Catalog shadowing: 44/53 gold questions match exactly one rule"
_SHADOWING_RE = re.compile(r"Catalog shadowing: (\d+)/(\d+) gold\s+questions")

#: Gold questions that more than one rule matches, and which rules those are.
#:
#: Each key is a question whose routing depends on catalog order; each value
#: names the rules that also match it, by *their* gold question, in catalog
#: order. Naming them that way rather than by index is what lets a rule be
#: inserted anywhere without editing this record.
#:
#: This is a baseline, not a backlog. Unlike the self-comparing gold rows in
#: ``test_gold_independence.py`` there is no work queued against these entries:
#: a specific rule sitting ahead of the general rule it refines is correct, and
#: "fixing" it would mean narrowing one of the two patterns until they stop
#: overlapping, which costs reach for no gain in correctness. The list exists so
#: that the twelfth entry has to be added deliberately.
ACCEPTED_SHADOWING: dict[str, tuple[str, ...]] = {
    "How has revenue by category changed month over month?": (
        "Show month-over-month revenue growth in 2024.",
        "Show revenue by category",
    ),
    "Show revenue by region and category.": (
        "Show revenue by category",
        "Show revenue by region",
    ),
    "Show revenue by category with a total row.": ("Show revenue by category",),
    "What share of customers bought from each category?": (
        "What percentage of revenue comes from each category?",
    ),
    "What is the average order value by month in 2024?": (
        "What is the average order value?",
    ),
    "What is the average order value by region?": ("What is the average order value?",),
    "What is the best-selling product in each category?": (
        "What is the best selling product?",
    ),
    "What is the best-selling product in each region?": (
        "What is the best selling product?",
    ),
    "Show revenue by customer signup quarter.": ("Show revenue by quarter in 2024.",),
    "How much revenue comes from new vs returning customers?": (
        "How many repeat customers are there?",
        "What is the total revenue?",
    ),
    "How many orders contain products from more than one category?": (
        "How many orders do we have?",
    ),
    # ARPU by region sits ahead of the ungrouped ARPU rule, which also matches
    # "revenue per customer" and would answer with one global figure.
    "What is the average revenue per customer in each region?": (
        "What is the average revenue per customer?",
    ),
}

#: What the question would be answered with if the order were reversed.
#:
#: Kept next to the baseline because "this overlap is the correct ordering, not
#: a bug" is the one claim here a reader has to take on trust, and it is only
#: checkable against what the losing rule would actually have returned. An entry
#: is documentation; the mechanical check is :func:`evaluate.check_shadowing`.
SHADOWING_RATIONALE: dict[str, str] = {
    "How has revenue by category changed month over month?": (
        "the intersection of two rules, and ahead of both: the growth rule "
        "would drop the category dimension and return one series for the "
        "business, and the revenue-by-category rule would drop time and return "
        "one row per category."
    ),
    "Show revenue by region and category.": (
        "a two-dimension breakdown ahead of both single-dimension rules. "
        "Either would return a correctly-computed table with one of the two "
        "dimensions silently collapsed."
    ),
    "Show revenue by customer signup quarter.": (
        "a narrower refinement of revenue-by-quarter: the rule behind buckets "
        "the same revenue by order date instead of signup date, which returns a "
        "plausible quarterly table that answers a different question."
    ),
    "Show revenue by category with a total row.": (
        "the rollup refines the plain breakdown: the rule behind returns the "
        "same rows without the grand-total line, which is the one thing the "
        "question asked for."
    ),
    "What share of customers bought from each category?": (
        "both rules match 'share ... category', and they differ in what is "
        "being shared. The rule behind divides *revenue* by category; this "
        "question asks what fraction of *customers* bought from each. A "
        "mis-ordering here changes the subject of the answer rather than its "
        "granularity, and both tables are one percentage column per category."
    ),
    "What is the average order value by month in 2024?": (
        "the monthly series ahead of the single overall figure."
    ),
    "What is the average order value by region?": (
        "the per-region breakdown ahead of the single overall figure."
    ),
    "What is the best-selling product in each category?": (
        "the per-category winner ahead of the global one, which would return a "
        "single product for the whole catalog."
    ),
    "What is the best-selling product in each region?": (
        "the per-region winner ahead of the global one, which would return a "
        "single product for the whole catalog."
    ),
    "How much revenue comes from new vs returning customers?": (
        "two rules behind it, and neither would return revenue split by "
        "segment. The repeat-customer rule counts customers; the total-revenue "
        "rule returns one grand total, and its unscoped/ungrouped guards do not "
        "decline this phrasing because the split is expressed as 'new vs "
        "returning' rather than as a year or a 'by <dimension>' clause they "
        "know to look for."
    ),
    "How many orders contain products from more than one category?": (
        "the order-count rule's guards exclude 'by category' but not 'from "
        "more than one category', so it matches and would return the count of "
        "all orders rather than of the cross-category subset — a plausible "
        "single number, several times too large."
    ),
    "What is the average revenue per customer in each region?": (
        "the ungrouped ARPU rule keys on 'revenue per customer' and does not "
        "look at 'region', so it would return one global average with the "
        "regional split silently dropped -- correctly labelled, wrong grain."
    ),
}


@pytest.fixture(scope="module")
def gold_records() -> list[dict]:
    return [
        json.loads(line)
        for line in GOLD_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def shadowing(gold_records: list[dict]) -> list[evaluate.ShadowResult]:
    """Run the real check over the real gold set.

    Needs neither a database nor a schema: shadowing is a property of the
    patterns and their order, and nothing downstream of matching can change it.
    """
    return evaluate.check_shadowing(llm.OfflineBackend(), gold_records)


def _contested_by_name(
    results: list[evaluate.ShadowResult],
) -> dict[str, tuple[str, ...]]:
    """Re-express the measured overlaps in the form :data:`ACCEPTED_SHADOWING` uses.

    A rule that no gold question reaches has no name to give, so it falls back
    to its index — which will not match any recorded entry and so surfaces as a
    difference rather than being quietly skipped.
    """
    examples = evaluate.shadow_examples(results)
    return {
        item.question: tuple(
            examples.get(index, f"rule #{index}") for index in item.matched[1:]
        )
        for item in results
        if item.contested
    }


# --------------------------------------------------------------------------- #
# check_shadowing
# --------------------------------------------------------------------------- #
def test_a_question_claimed_by_two_rules_is_contested() -> None:
    """The defect the check exists for: routing decided by position."""
    backend = llm.OfflineBackend()
    question = "What is the average order value by region?"

    (result,) = evaluate.check_shadowing(backend, [{"question": question}])

    assert result.contested
    assert len(result.matched) > 1
    assert result.matched == tuple(backend.matching_rule_indexes(question))


def test_matched_is_in_catalog_order_with_the_answering_rule_first() -> None:
    """``matched[0]`` must be the rule that answers, or the report misnames it."""
    backend = llm.OfflineBackend()
    question = "Show revenue by region and category."

    (result,) = evaluate.check_shadowing(backend, [{"question": question}])

    assert list(result.matched) == sorted(result.matched)
    assert result.matched[0] == backend.matching_rule_indexes(question)[0]


def test_a_question_only_one_rule_claims_is_not_contested() -> None:
    backend = llm.OfflineBackend()
    question = "Show monthly cohort retention."
    assert len(backend.matching_rule_indexes(question)) == 1

    (result,) = evaluate.check_shadowing(backend, [{"question": question}])

    assert not result.contested


def test_a_question_no_rule_claims_is_not_contested() -> None:
    """No rules cannot be an ordering problem; the accuracy run already fails it."""
    (result,) = evaluate.check_shadowing(
        llm.OfflineBackend(), [{"question": "what is the meaning of life?"}]
    )

    assert result.matched == ()
    assert not result.contested


# --------------------------------------------------------------------------- #
# shadow_examples / format_shadowing
# --------------------------------------------------------------------------- #
def test_shadow_examples_names_each_rule_by_the_question_that_wins_it() -> None:
    results = [
        evaluate.ShadowResult("question for rule 3", (3,)),
        evaluate.ShadowResult("question for rule 1", (1, 3)),
    ]

    assert evaluate.shadow_examples(results) == {
        3: "question for rule 3",
        1: "question for rule 1",
    }


def test_shadow_examples_ignores_questions_that_match_nothing() -> None:
    """An unroutable question names no rule and must not claim an index."""
    assert evaluate.shadow_examples([evaluate.ShadowResult("unroutable", ())]) == {}


def test_format_shadowing_names_the_losing_rules_not_just_the_count() -> None:
    """A bare count gives nowhere to look; the losing rule is the thing to see."""
    results = [
        evaluate.ShadowResult("contested question", (2, 9)),
        evaluate.ShadowResult("question for rule 9", (9,)),
        evaluate.ShadowResult("uncontested question", (4,)),
    ]

    rendered = evaluate.format_shadowing(results)

    assert "2/3" in rendered
    assert "contested question" in rendered
    assert "#9: question for rule 9" in rendered
    assert "uncontested question" not in rendered


def test_format_shadowing_falls_back_to_the_index_for_an_unnamed_rule() -> None:
    """A rule no gold question reaches still has to be identifiable."""
    rendered = evaluate.format_shadowing(
        [evaluate.ShadowResult("contested question", (2, 9))]
    )

    assert "#9" in rendered
    assert "no gold question" in rendered


def test_format_shadowing_says_nothing_extra_when_every_question_is_clean() -> None:
    rendered = evaluate.format_shadowing([evaluate.ShadowResult("q", (1,))])

    assert rendered == (
        "Catalog shadowing: 1/1 gold questions match exactly one rule"
    )


def test_report_carries_shadowing_only_when_it_was_checked() -> None:
    """Absent means "not measured", which is not the same as "nothing found"."""
    results = [
        evaluate.ShadowResult("contested", (1, 2)),
        evaluate.ShadowResult("clean", (3,)),
    ]

    assert "shadowing" not in evaluate.build_report([], "llm")
    report = evaluate.build_report([], "offline", None, None, None, results)
    assert report["shadowing"] == {
        "checked": 2,
        "single_match": 1,
        "contested": [{"question": "contested", "matched": (1, 2)}],
    }


# --------------------------------------------------------------------------- #
# The ratchet
# --------------------------------------------------------------------------- #
def test_no_new_shadowing(shadowing: list[evaluate.ShadowResult]) -> None:
    """A rule widened across an older rule's question fails here, not silently.

    Compares claimants rather than only which questions are contested: a
    question already on the list acquiring a *third* claimant is the same
    failure as a clean question acquiring a second, and a check that only
    counted contested questions would see neither.

    Additions only. An overlap that has *gone* is the other test's business, so
    that each failure message can describe what actually happened.
    """
    measured = _contested_by_name(shadowing)
    added = [
        f"{question!r} is now also claimed by {sorted(set(also) - recorded)}"
        for question, also in sorted(measured.items())
        for recorded in [set(ACCEPTED_SHADOWING.get(question, ()))]
        if not set(also) <= recorded
    ]
    assert not added, (
        "these questions are claimed by a rule nobody recorded, so their "
        "answers now depend on catalog order in a way no baseline covers: "
        + "; ".join(added)
        + ". Narrow the widened pattern, or add the claimant to "
        "ACCEPTED_SHADOWING with a rationale saying what that rule would have "
        "answered instead."
    )


def test_accepted_shadowing_has_no_stale_entries(
    shadowing: list[evaluate.ShadowResult], gold_records: list[dict]
) -> None:
    """An overlap that has been narrowed away must leave the record.

    Covers both shapes of staleness: a question no longer contested at all, and
    a question still contested by fewer rules than the record claims. A
    rationale defending an overlap that no longer exists reads as a defence of
    the current catalog and is not one.

    Only questions still in the gold set are considered: an entry whose question
    was deleted is not a stale claim about the catalog, and failing on it would
    turn removing a gold row into an unrelated test failure.
    """
    present = {record["question"] for record in gold_records}
    measured = _contested_by_name(shadowing)
    resolved = [
        (
            f"{question!r} is no longer claimed by "
            f"{sorted(set(recorded) - set(measured.get(question, ())))}"
        )
        for question, recorded in sorted(ACCEPTED_SHADOWING.items())
        if question in present
        and not set(recorded) <= set(measured.get(question, ()))
    ]
    assert not resolved, (
        "these overlaps no longer exist — trim them from ACCEPTED_SHADOWING "
        "(and SHADOWING_RATIONALE, if the whole entry goes) so the baseline "
        "reflects the catalog: " + "; ".join(resolved)
    )


def test_every_accepted_entry_says_why_the_order_is_right() -> None:
    """A baseline without reasons is a list of overlaps nobody has judged."""
    assert set(ACCEPTED_SHADOWING) == set(SHADOWING_RATIONALE), (
        "ACCEPTED_SHADOWING and SHADOWING_RATIONALE must describe the same "
        "questions; unmatched: "
        + "; ".join(sorted(set(ACCEPTED_SHADOWING) ^ set(SHADOWING_RATIONALE)))
    )


def test_accepted_shadowing_names_real_gold_questions(
    gold_records: list[dict],
) -> None:
    """Every name in the record must be a question the gold set still holds.

    Both sides: the contested question and each rule named as also matching it.
    A typo or a renamed question would otherwise sit in the baseline looking
    like a recorded overlap while matching nothing.
    """
    present = {record["question"] for record in gold_records}
    named = set(ACCEPTED_SHADOWING) | {
        question for also in ACCEPTED_SHADOWING.values() for question in also
    }
    unknown = sorted(named - present)
    assert not unknown, (
        "ACCEPTED_SHADOWING names questions that are not in the gold set: "
        + "; ".join(unknown)
    )


def test_no_question_is_recorded_as_shadowing_itself() -> None:
    """A rule cannot lose to itself; such an entry would mask a real claimant."""
    self_referential = sorted(
        question for question, also in ACCEPTED_SHADOWING.items() if question in also
    )
    assert not self_referential, (
        "these entries list their own question among the rules that also match "
        "it: " + "; ".join(self_referential)
    )


def test_readme_shadowing_figures_match_the_harness(
    shadowing: list[evaluate.ShadowResult],
) -> None:
    """The README quotes this ratio, and a stale one misstates the catalog."""
    readme = README_PATH.read_text(encoding="utf-8")
    match = _SHADOWING_RE.search(readme)
    assert match, "README no longer quotes a 'Catalog shadowing: N/M' line"

    clean = sum(1 for item in shadowing if not item.contested)
    assert (int(match.group(1)), int(match.group(2))) == (clean, len(shadowing)), (
        f"README claims {match.group(1)}/{match.group(2)} singly-matched gold "
        f"questions; the harness measures {clean}/{len(shadowing)}"
    )
