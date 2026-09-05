"""Tests for the paraphrase set — rephrasings that must keep routing alike.

The gold set carries exactly one phrasing per rule, which makes execution
accuracy blind to a specific regression: inserting a broadly phrased rule ahead
of an older one can quietly capture some of the older rule's phrasings without
touching the one phrasing the gold set happens to use. Accuracy stays at 100%
while the catalog has started answering rephrased questions with the wrong
query.

``evals/paraphrases.jsonl`` pins that down by pairing each canonical gold
question with an alternate phrasing and asserting both route to the same rule.
A record may instead carry a ``known_gap`` — a phrasing the catalog is known not
to reach, recorded so the set measures the matcher's actual reach rather than
only the phrasings that already work.

These tests cover four things:

1. the loader rejects a malformed record rather than skipping it;
2. the checker reports *where* a paraphrase landed, not merely that it differed;
3. the shipped set passes against the live catalog (the regression test);
4. the file stays coherent — canonicals are real gold questions, gaps are
   genuinely still failing, and no pair is duplicated.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

# Same import shim as tests/test_precision_guard.py: evals/ is a script
# directory, not an installed package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "evals"))

import evaluate  # noqa: E402

from nl2sql.llm import OfflineBackend  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def backend() -> OfflineBackend:
    return OfflineBackend()


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return evaluate.load_paraphrase_set(evaluate.PARAPHRASE_PATH)


@pytest.fixture(scope="module")
def checked(backend: OfflineBackend, records: list[dict]) -> list[evaluate.ParaphraseResult]:
    return evaluate.check_paraphrases(backend, records)


@pytest.fixture(scope="module")
def gold_questions() -> set[str]:
    path = REPO_ROOT / "evals" / "gold.jsonl"
    return {
        json.loads(line)["question"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


# --------------------------------------------------------------------------- #
# The shipped paraphrase set
# --------------------------------------------------------------------------- #
def test_every_gating_paraphrase_routes_to_its_canonical_rule(
    checked: list[evaluate.ParaphraseResult],
) -> None:
    """The regression test: no gating pair may drift onto another rule.

    The failure message names both rule indexes, because "landed on rule #18"
    and "landed nowhere" call for opposite fixes — narrowing the rule that stole
    the phrasing versus widening the one that should have caught it.
    """
    misrouted = [
        f"{item.paraphrase!r} -> rule {item.paraphrase_rule} "
        f"(canonical {item.canonical!r} -> rule {item.canonical_rule})"
        for item in checked
        if item.known_gap is None and not item.routed_alike
    ]
    assert not misrouted, "paraphrases no longer route to their canonical rule:\n" + "\n".join(
        misrouted
    )


def test_every_canonical_question_routes_somewhere(
    checked: list[evaluate.ParaphraseResult],
) -> None:
    """A canonical that matches no rule makes its pair meaningless.

    Checked separately from the routing test so the diagnosis is unambiguous:
    this failure is about the canonical question, not about the paraphrase.
    """
    orphaned = [item.canonical for item in checked if item.canonical_rule is None]
    assert not orphaned, f"canonical questions that match no rule: {orphaned}"


def test_known_gaps_are_still_gaps(checked: list[evaluate.ParaphraseResult]) -> None:
    """A documented gap that started routing correctly must be promoted.

    Left in place it would be a stale caveat: the file would claim the catalog
    cannot reach a phrasing it now reaches, and the pair would sit outside the
    gating count where a later regression could break it unnoticed.
    """
    recovered = [item.paraphrase for item in checked if item.recovered]
    assert not recovered, (
        "these known gaps now route correctly — remove their 'known_gap' field "
        f"to make them gating pairs: {recovered}"
    )


def test_canonicals_are_gold_questions(
    records: list[dict], gold_questions: set[str]
) -> None:
    """Every canonical is a question the gold set already verifies.

    The paraphrase set only checks *routing*. Anchoring each canonical to a gold
    question is what makes that meaningful: the gold row proves the rule they
    both reach returns the right answer, so routing alike is enough to conclude
    the paraphrase is answered correctly too.
    """
    unknown = sorted({r["canonical"] for r in records} - gold_questions)
    assert not unknown, f"canonical questions absent from evals/gold.jsonl: {unknown}"


def test_every_rule_has_at_least_one_paraphrase(
    backend: OfflineBackend, records: list[dict], gold_questions: set[str]
) -> None:
    """Every rule the gold set reaches must also carry a rephrasing.

    The robustness ratio is only as wide as the set behind it. Before this guard
    the set covered 31 of the catalog's rules, so "30/30 rephrasings route to the
    canonical rule" was a clean-looking number that had never touched the other
    19 -- and a headline that reports 100% over an unstated fraction of the
    catalog is exactly the kind of figure this repo refuses to print elsewhere.

    Filling the gap once is not enough, because the ratio degrades silently: each
    new pattern that ships without a rephrasing lowers coverage while the printed
    number stays at 100%. This asserts the property instead of the snapshot, so a
    new rule must arrive with a paraphrase or fail here.

    Rules are identified by the gold question that reaches them rather than by
    index, since indexes shift whenever a pattern is inserted mid-catalog.
    """
    covered = {
        matches[0]
        for record in records
        if (matches := backend.matching_rule_indexes(record["canonical"]))
    }
    uncovered = sorted(
        (matches[0], question)
        for question in gold_questions
        if (matches := backend.matching_rule_indexes(question)) and matches[0] not in covered
    )
    assert not uncovered, (
        "these rules have no paraphrase, so nothing measures whether they "
        "survive rephrasing:\n"
        + "\n".join(f"  rule {index}: {question}" for index, question in uncovered)
    )


def test_no_duplicate_paraphrases(records: list[dict]) -> None:
    """One record per phrasing, so a pair cannot be counted twice."""
    seen = [record["paraphrase"] for record in records]
    duplicates = sorted({p for p in seen if seen.count(p) > 1})
    assert not duplicates, f"duplicate paraphrases: {duplicates}"


def test_paraphrases_differ_from_their_canonical(records: list[dict]) -> None:
    """A paraphrase identical to its canonical would test nothing."""
    identical = [
        record["paraphrase"]
        for record in records
        if record["paraphrase"].strip().casefold() == record["canonical"].strip().casefold()
    ]
    assert not identical, f"paraphrase repeats its canonical question: {identical}"


def test_readme_counts_match_the_shipped_set(records: list[dict]) -> None:
    """The two figures quoted in the README are re-measured from the file.

    Same rule as ``tests/test_docs.py`` applies to the gold counts: a figure
    stated once and never re-measured reads as a metric, and is worse than no
    figure at all. Both the prose count and the sample console block are
    checked, since either can be updated without the other.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    gating = sum(1 for record in records if record.get("known_gap") is None)
    gaps = len(records) - gating

    prose = re.search(r"\*\*(\d+) gating pairs and\s+(\d+) known gaps\*\*", readme)
    assert prose, "README no longer quotes the paraphrase set's gating/gap counts"
    assert (int(prose.group(1)), int(prose.group(2))) == (gating, gaps), (
        f"README claims {prose.group(1)} gating pairs and {prose.group(2)} known "
        f"gaps; evals/paraphrases.jsonl has {gating} and {gaps}"
    )

    sample = re.search(
        r"Paraphrase robustness: (\d+)/(\d+) rephrasings.*?"
        r"Known gaps \(not gating\): (\d+)",
        readme,
        re.S,
    )
    assert sample, "README no longer shows a sample paraphrase-robustness block"
    assert [int(group) for group in sample.groups()] == [gating, gating, gaps], (
        "the README sample block disagrees with evals/paraphrases.jsonl "
        f"({gating} gating pairs, all routing, and {gaps} known gaps)"
    )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def test_loader_reads_optional_known_gap(tmp_path: Path) -> None:
    path = tmp_path / "paraphrases.jsonl"
    path.write_text(
        '{"canonical": "a", "paraphrase": "b"}\n'
        '\n'
        '{"canonical": "c", "paraphrase": "d", "known_gap": "why"}\n',
        encoding="utf-8",
    )
    records = evaluate.load_paraphrase_set(str(path))
    assert [r.get("known_gap") for r in records] == [None, "why"]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('{"paraphrase": "b"}', "canonical"),
        ('{"canonical": "a"}', "paraphrase"),
        ('{"canonical": "a", "paraphrase": "   "}', "paraphrase"),
        ('{"canonical": "a", "paraphrase": "b", "known_gap": ""}', "known_gap"),
        ('{"canonical": "a", "paraphrase": "b", "known_gap": 7}', "known_gap"),
    ],
)
def test_loader_rejects_malformed_records(tmp_path: Path, line: str, expected: str) -> None:
    """A malformed record raises, naming the field, instead of being skipped.

    A skipped record is a check that stops running while still reporting a clean
    result — the same failure the gold and guard loaders raise to avoid.
    """
    path = tmp_path / "paraphrases.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        evaluate.load_paraphrase_set(str(path))


# --------------------------------------------------------------------------- #
# Checking and reporting
# --------------------------------------------------------------------------- #
def _result(**kwargs: object) -> evaluate.ParaphraseResult:
    defaults: dict = {
        "canonical": "canonical question",
        "paraphrase": "paraphrased question",
        "canonical_rule": 3,
        "paraphrase_rule": 3,
        "known_gap": None,
    }
    defaults.update(kwargs)
    return evaluate.ParaphraseResult(**defaults)  # type: ignore[arg-type]


def test_checker_records_both_rule_indexes(backend: OfflineBackend) -> None:
    """The reported indexes are the rules the live matcher actually picked."""
    canonical = "How many customers do we have?"
    paraphrase = "How many customers are there?"
    [result] = evaluate.check_paraphrases(
        backend, [{"canonical": canonical, "paraphrase": paraphrase}]
    )
    assert result.canonical_rule == backend.matching_rule_indexes(canonical)[0]
    assert result.paraphrase_rule == backend.matching_rule_indexes(paraphrase)[0]
    assert result.routed_alike


def test_checker_reports_an_unmatched_paraphrase_as_none(backend: OfflineBackend) -> None:
    [result] = evaluate.check_paraphrases(
        backend,
        [
            {
                "canonical": "How many customers do we have?",
                "paraphrase": "zzzz not a question the catalog covers zzzz",
            }
        ],
    )
    assert result.paraphrase_rule is None
    assert not result.routed_alike
    assert not result.passed


def test_known_gap_does_not_fail_the_run() -> None:
    result = _result(paraphrase_rule=None, known_gap="vocabulary not covered")
    assert not result.routed_alike
    assert result.passed


def test_broken_pair_fails_even_when_labelled_a_known_gap() -> None:
    """A canonical that routes nowhere is a defect in the file, not a gap.

    ``known_gap`` excuses a paraphrase the catalog cannot reach. It must not
    also excuse a canonical the catalog cannot reach, or a typo in the canonical
    would silently retire the pair.
    """
    result = _result(canonical_rule=None, paraphrase_rule=None, known_gap="documented")
    assert not result.passed


def test_recovered_gap_is_flagged_but_still_passes() -> None:
    result = _result(known_gap="documented")
    assert result.recovered
    assert result.passed


def test_format_lists_a_misrouted_paraphrase_with_both_rules() -> None:
    text = evaluate.format_paraphrases([_result(paraphrase_rule=9)])
    assert "0/1" in text
    assert "MISROUTED" in text
    assert "rule #9" in text
    assert "#3" in text


def test_format_counts_gating_pairs_only() -> None:
    """Documenting a gap removes a pair from the denominator, not the numerator."""
    text = evaluate.format_paraphrases(
        [_result(), _result(paraphrase="other", paraphrase_rule=None, known_gap="why")]
    )
    assert "1/1" in text
    assert "Known gaps (not gating): 1" in text


def test_format_flags_a_recovered_gap() -> None:
    text = evaluate.format_paraphrases([_result(known_gap="documented")])
    assert "NOW ROUTING" in text


# --------------------------------------------------------------------------- #
# Report payload and exit code
# --------------------------------------------------------------------------- #
def test_report_omits_robustness_when_not_checked() -> None:
    report = evaluate.build_report([], "llm")
    assert "robustness" not in report


def test_report_separates_gating_pairs_from_known_gaps() -> None:
    report = evaluate.build_report(
        [],
        "offline",
        None,
        [
            _result(),
            _result(paraphrase="misrouted", paraphrase_rule=9),
            _result(paraphrase="gap", paraphrase_rule=None, known_gap="why"),
            _result(paraphrase="recovered", known_gap="why"),
        ],
    )
    assert report["robustness"]["gating"] == 2
    assert report["robustness"]["routed"] == 1
    assert report["robustness"]["known_gaps"] == 2
    assert report["robustness"]["recovered_gaps"] == 1
    assert len(report["robustness"]["paraphrases"]) == 4


def test_shipped_set_leaves_the_harness_green(tmp_path: Path) -> None:
    """End to end: the committed paraphrase set must not fail the run.

    Run through ``main`` rather than the checker so the exit-code wiring is
    covered too — a check that reports a failure but returns 0 is not a gate.
    """
    report_path = tmp_path / "report.json"
    assert evaluate.main(["--json", str(report_path)]) == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["robustness"]["routed"] == payload["robustness"]["gating"]


def test_a_misrouted_paraphrase_makes_the_run_exit_nonzero(tmp_path: Path) -> None:
    path = tmp_path / "paraphrases.jsonl"
    path.write_text(
        json.dumps(
            {
                "canonical": "How many customers do we have?",
                "paraphrase": "Show revenue by category",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert evaluate.main(["--paraphrases", str(path)]) == 1
