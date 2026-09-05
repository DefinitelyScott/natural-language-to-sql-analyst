"""Keep the gold set from silently becoming a copy of the code it measures.

Execution accuracy compares a generated query against a gold query. That
comparison only carries information when the two were written separately: if the
gold SQL for a question is the offline rule's own SQL, the harness runs one query
twice and compares it to itself. The row-for-row match is then guaranteed by
construction — it would still hold if the rule computed revenue by *region* for a
question asking about categories — so the question proves the SQL parses and
executes, and nothing about whether it answers what was asked.

Eight of the current gold rows are still such copies, so this cannot be a check
that simply fails until they are all rewritten: a check that is red on every run
is one people learn to scroll past, and rewriting the whole backlog is not a
change anyone should make in one sitting. It is a *ratchet* instead. The copies
are recorded in :data:`KNOWN_SELF_COMPARING` by name, and the tests below assert
two things about that list: nothing outside it may be a copy (so a rule added with
copy-pasted gold SQL fails immediately), and nothing on it may still be listed
once it has been rewritten (so the backlog can only shrink, and shrinks visibly).

The harness reports the same measurement without gating on it — ``evaluate.py``
measures and prints, this file decides what is allowed to change. That split is
why the console line is safe to read as information rather than as an alarm.
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

# "Gold independence: 37/51 gold queries are written independently..."
_INDEPENDENCE_RE = re.compile(r"Gold independence: (\d+)/(\d+) gold\s+queries")

#: Gold questions whose SQL is currently a copy of the rule they test.
#:
#: This is a backlog, not an approved list. Each entry is a question whose gold
#: query should be rewritten a different way that computes the same answer — a
#: different join order, a subquery where the rule uses a CTE, ``COUNT(DISTINCT
#: ...)`` where the rule groups — so that the two agreeing means something. The
#: list may only shrink; ``test_no_new_self_comparing_gold_rows`` rejects
#: additions and ``test_known_self_comparing_list_has_no_stale_entries`` rejects
#: leaving a fixed one behind.
#:
#: A rewrite only counts if it reaches the answer by a different route. Restating
#: ``COUNT(*) FROM products`` as ``COUNT(DISTINCT id) FROM products`` clears the
#: text comparison without adding a second opinion — the two cannot disagree on a
#: primary key — so it would raise the printed ratio while proving nothing more
#: than before. The whole-table counts left on this list are here for that reason
#: and may outlast the rest of the backlog: for "how many rows are in this table"
#: there is no genuinely independent second formulation to write.
KNOWN_SELF_COMPARING = frozenset(
    {
        "How many customers do we have?",
        "How many new customers signed up by month in 2024?",
        "How many orders do we have?",
        "How many products are in the catalog?",
        "Show revenue by day of week.",
        "Show revenue by price tier.",
        "Which category do customers buy from first?",
        "Which products are most frequently bought together?",
    }
)

#: Why each rewritten gold query counts as a second opinion.
#:
#: Kept here because ``gold.jsonl`` is JSONL and cannot carry a comment, and
#: because "this rewrite is genuinely independent" is the one claim in the whole
#: check that a reader has to take on trust. Each entry names the disagreement
#: the rewrite is now capable of producing — if the rule broke in that specific
#: way, the two queries would return different rows and the gold row would fail.
#: An entry is documentation, not an assertion; the mechanical check is
#: ``check_independence``, which compares the two query texts.
REWRITE_RATIONALE = {
    "What were total sales by month in 2024?": (
        "the rule sums across the orders x order_items fan-out; the gold query "
        "totals each order in a correlated subquery first and then sums those "
        "per month, and filters the year with strftime rather than a half-open "
        "date range. A join that duplicated line items would inflate the rule "
        "and not the gold query."
    ),
    "Which 5 customers spent the most?": (
        "join plus GROUP BY in the rule against a per-customer correlated "
        "subquery in the gold query, which also drops customers with no orders "
        "via NULL rather than via the inner join. Grouping on a non-unique key "
        "-- c.name instead of c.id -- would merge two same-named customers in "
        "the rule only."
    ),
    "What are the 10 largest orders?": (
        "the rule aggregates across a three-table join; the gold query totals "
        "order_items alone in a CTE and joins customers afterwards, so it "
        "cannot be affected by a customer join that multiplies rows."
    ),
    "How many unique customers placed an order each month in 2024?": (
        "COUNT(DISTINCT customer_id) against DISTINCT in a subquery followed by "
        "a plain COUNT(*) -- the two standard formulations of distinctness, "
        "which disagree if the rule counts rows where it should count customers."
    ),
    "How many orders were placed in the last 30 days?": (
        "the rule compares ISO date strings against date(MAX(order_date), '-30 "
        "day'); the gold query differences julianday values instead. They "
        "disagree if the window is off by a day or if string ordering of dates "
        "ever stopped matching chronological ordering."
    ),
    "Show revenue by region and category.": (
        "opposite join orders -- the rule starts from customers and joins "
        "outward, the gold query builds a line-level CTE from order_items and "
        "joins customers last -- so a mis-stated join condition would have to "
        "be wrong identically in both directions to go unnoticed."
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
def independence(gold_records: list[dict]) -> list[evaluate.IndependenceResult]:
    """Run the real check over the real gold set.

    The schema text is empty because the offline backend resolves a question by
    regex and never reads it. Passing a real schema would make the fixture depend
    on a built database for a value that provably cannot affect the outcome.
    """
    return evaluate.check_independence(llm.OfflineBackend(), "", gold_records)


# --------------------------------------------------------------------------- #
# normalize_sql
# --------------------------------------------------------------------------- #
def test_normalize_sql_ignores_layout_semicolons_and_keyword_case() -> None:
    """The three ways a copy-pasted query gets edited without being rewritten."""
    reindented = """
        SELECT   region,
                 SUM(total) AS revenue
        FROM     orders;
    """
    inlined = "select region, sum(total) as revenue from orders"
    assert evaluate.normalize_sql(reindented) == evaluate.normalize_sql(inlined)


def test_normalize_sql_keeps_genuinely_different_queries_apart() -> None:
    """Normalization must not be so aggressive that it manufactures matches."""
    assert evaluate.normalize_sql("SELECT COUNT(*) FROM orders") != evaluate.normalize_sql(
        "SELECT COUNT(DISTINCT customer_id) FROM orders"
    )


# --------------------------------------------------------------------------- #
# check_independence
# --------------------------------------------------------------------------- #
def test_gold_sql_copied_from_the_rule_is_flagged() -> None:
    """The defect the check exists for: gold SQL pasted from the rule."""
    backend = llm.OfflineBackend()
    question = "What is the total revenue?"
    record = {"question": question, "sql": backend.to_sql(question, "")}

    (result,) = evaluate.check_independence(backend, "", [record])

    assert result.independent is False
    assert result.rule is not None


def test_reformatting_a_copy_does_not_make_it_count_as_independent() -> None:
    """Re-indenting pasted SQL is the usual way a copy stops looking like one."""
    backend = llm.OfflineBackend()
    question = "What is the total revenue?"
    pasted = backend.to_sql(question, "")
    reformatted = "\n    ".join(pasted.split()) + "\n;"

    (result,) = evaluate.check_independence(
        backend, "", [{"question": question, "sql": reformatted}]
    )

    assert result.independent is False


def test_independently_written_gold_sql_passes() -> None:
    """A different query for the same answer is what the gold set should hold."""
    backend = llm.OfflineBackend()
    record = {
        "question": "What is the total revenue?",
        "sql": "SELECT ROUND(SUM(quantity * unit_price), 2) FROM order_items",
    }

    (result,) = evaluate.check_independence(backend, "", [record])

    assert result.independent is True


def test_question_matching_no_rule_counts_as_independent() -> None:
    """There is no rule SQL to have copied, and the accuracy run already fails it."""
    record = {"question": "what is the meaning of life?", "sql": "SELECT 42"}

    (result,) = evaluate.check_independence(llm.OfflineBackend(), "", [record])

    assert result.rule is None
    assert result.independent is True


def test_format_independence_names_every_copy() -> None:
    """A count alone gives no starting point; the fix is per-question."""
    results = [
        evaluate.IndependenceResult("copied question", 7, independent=False),
        evaluate.IndependenceResult("original question", 8, independent=True),
    ]

    rendered = evaluate.format_independence(results)

    assert "1/2" in rendered
    assert "copied question" in rendered
    assert "#7" in rendered
    assert "original question" not in rendered


def test_report_carries_independence_only_when_it_was_checked() -> None:
    """Absent means "not measured", which is not the same as "nothing found"."""
    results = [evaluate.IndependenceResult("q", 1, independent=False)]

    assert "independence" not in evaluate.build_report([], "llm")
    report = evaluate.build_report([], "offline", None, None, results)
    assert report["independence"] == {
        "checked": 1,
        "independent": 0,
        "questions": [{"question": "q", "rule": 1, "independent": False}],
    }


# --------------------------------------------------------------------------- #
# The ratchet
# --------------------------------------------------------------------------- #
def test_no_new_self_comparing_gold_rows(
    independence: list[evaluate.IndependenceResult],
) -> None:
    """A rule added with copy-pasted gold SQL fails here, not silently at 100%."""
    copies = {item.question for item in independence if not item.independent}
    new = sorted(copies - KNOWN_SELF_COMPARING)
    assert not new, (
        "these gold queries are copies of the rule they test, so their gold "
        "rows cannot fail: " + "; ".join(new) + ". Rewrite the gold SQL a "
        "different way that computes the same answer, or add the question to "
        "KNOWN_SELF_COMPARING with a reason if it genuinely cannot be."
    )


def test_known_self_comparing_list_has_no_stale_entries(
    independence: list[evaluate.IndependenceResult],
    gold_records: list[dict],
) -> None:
    """A rewritten gold query must leave the backlog, so the list can only shrink.

    Only questions still present in the gold set are considered. An entry whose
    question has been removed outright is not stale in the sense that matters
    here — nothing was fixed to make it independent — and failing on it would
    turn deleting a gold row into an unrelated test failure.
    """
    present = {record["question"] for record in gold_records}
    still_copied = {item.question for item in independence if not item.independent}
    fixed = sorted((KNOWN_SELF_COMPARING & present) - still_copied)
    assert not fixed, (
        "these gold queries no longer copy their rule — remove them from "
        "KNOWN_SELF_COMPARING so the backlog reflects reality: " + "; ".join(fixed)
    )


def test_rewrite_rationale_describes_real_independent_rows(
    independence: list[evaluate.IndependenceResult],
    gold_records: list[dict],
) -> None:
    """Documentation that can drift from the gold set is worse than none.

    A rationale left behind for a question that was deleted, renamed, or has
    quietly gone back to copying its rule would read as a defence of something
    that is no longer true, which is the failure mode this whole check exists to
    prevent one level down.
    """
    present = {record["question"] for record in gold_records}
    copies = {item.question for item in independence if not item.independent}

    missing = sorted(REWRITE_RATIONALE.keys() - present)
    assert not missing, (
        "REWRITE_RATIONALE describes questions that are no longer in the gold "
        "set: " + "; ".join(missing)
    )

    still_copied = sorted(REWRITE_RATIONALE.keys() & copies)
    assert not still_copied, (
        "REWRITE_RATIONALE claims these were rewritten, but they still copy "
        "their rule: " + "; ".join(still_copied)
    )

    overlap = sorted(REWRITE_RATIONALE.keys() & KNOWN_SELF_COMPARING)
    assert not overlap, (
        "a question cannot be both on the backlog and explained as rewritten: "
        + "; ".join(overlap)
    )


def test_readme_independence_figures_match_the_harness(
    independence: list[evaluate.IndependenceResult],
) -> None:
    """The README quotes this ratio, and a stale one flatters the eval set."""
    readme = README_PATH.read_text(encoding="utf-8")
    match = _INDEPENDENCE_RE.search(readme)
    assert match, "README no longer quotes a 'Gold independence: N/M' line"

    independent = sum(1 for item in independence if item.independent)
    assert (int(match.group(1)), int(match.group(2))) == (independent, len(independence)), (
        f"README claims {match.group(1)}/{match.group(2)} independent gold "
        f"queries; the harness measures {independent}/{len(independence)}"
    )
