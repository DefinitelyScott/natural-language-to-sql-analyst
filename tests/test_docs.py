"""Keep the numbers quoted in the README in step with ``evals/gold.jsonl``.

The README states three concrete figures: how many gold questions the evaluation
harness runs, how many of those are order-sensitive, and the ``total``/``passed``
pair in the sample ``--json`` report. All change every time a question pattern is
added, and a stale figure in a README is worse than no figure at all — it reads
as a metric that was reported once and never re-measured.

Rather than rely on remembering to edit prose, these tests parse the claims back
out of the README and compare them to the gold file itself. The regexes are
deliberately anchored to the exact wording used in the README, and each test
fails loudly if that wording disappears, so a rewrite of the surrounding prose
cannot silently disable the check.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"
GOLD_PATH = REPO_ROOT / "evals" / "gold.jsonl"

# "Evaluated 36 questions" — appears in the sample harness output.
_TOTAL_RE = re.compile(r"Evaluated (\d+) questions")
# "execution accuracy: 36/36 (100%)" — the same sample output line.
_ACCURACY_RE = re.compile(r"execution accuracy: (\d+)/(\d+)")
# "23 of the 36 gold questions are order-sensitive."
_ORDERED_RE = re.compile(r"(\d+) of the (\d+) gold questions\s+are order-sensitive")
# `"total": 36,` / `"passed": 36,` — the sample `--json` report body.
_JSON_TOTAL_RE = re.compile(r'"total":\s*(\d+)')
_JSON_PASSED_RE = re.compile(r'"passed":\s*(\d+)')


@pytest.fixture(scope="module")
def readme() -> str:
    return README_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def gold_counts() -> tuple[int, int]:
    """Return ``(total_questions, order_sensitive_questions)`` from the gold file."""
    records = [
        json.loads(line)
        for line in GOLD_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ordered = sum(1 for record in records if record.get("ordered", False))
    return len(records), ordered


def test_readme_question_count_matches_gold(readme: str, gold_counts: tuple[int, int]) -> None:
    total, _ = gold_counts
    claimed = [int(match) for match in _TOTAL_RE.findall(readme)]
    assert claimed, "README no longer quotes an 'Evaluated N questions' line"
    assert all(count == total for count in claimed), (
        f"README claims {claimed} gold questions but evals/gold.jsonl has {total}"
    )


def test_readme_accuracy_line_matches_gold(readme: str, gold_counts: tuple[int, int]) -> None:
    """The sample output shows a passing run, so both figures must be the total.

    The offline backend is expected to score 100% — every catalog pattern has a
    gold row — so a sample line reading anything other than ``N/N`` would either
    misreport the harness or advertise a regression.
    """
    total, _ = gold_counts
    matches = _ACCURACY_RE.findall(readme)
    assert matches, "README no longer quotes an 'execution accuracy: N/M' line"
    for passed, evaluated in matches:
        assert (int(passed), int(evaluated)) == (total, total), (
            f"README shows execution accuracy {passed}/{evaluated}; "
            f"evals/gold.jsonl has {total} questions and the offline backend passes all of them"
        )


def test_readme_json_report_sample_matches_gold(readme: str, gold_counts: tuple[int, int]) -> None:
    """The sample ``--json`` report shows a passing run, so both keys are the total."""
    total, _ = gold_counts
    claimed_total = _JSON_TOTAL_RE.findall(readme)
    claimed_passed = _JSON_PASSED_RE.findall(readme)
    assert claimed_total, 'README no longer shows a \'"total": N\' line in the sample report'
    assert claimed_passed, 'README no longer shows a \'"passed": N\' line in the sample report'
    assert all(int(count) == total for count in claimed_total + claimed_passed), (
        f"README sample report claims total={claimed_total} passed={claimed_passed} "
        f"but evals/gold.jsonl has {total} questions, all of which the offline backend passes"
    )


def test_readme_order_sensitive_count_matches_gold(
    readme: str, gold_counts: tuple[int, int]
) -> None:
    total, ordered = gold_counts
    match = _ORDERED_RE.search(readme)
    assert match, "README no longer quotes an 'N of the M gold questions are order-sensitive' claim"
    assert (int(match.group(1)), int(match.group(2))) == (ordered, total), (
        f"README claims {match.group(1)} of {match.group(2)} gold questions are "
        f"order-sensitive; evals/gold.jsonl has {ordered} of {total}"
    )
