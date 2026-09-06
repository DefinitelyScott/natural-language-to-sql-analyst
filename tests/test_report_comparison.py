"""Tests for the per-question diff between two evaluation reports.

The point of ``--compare`` is to catch what the accuracy headline hides, so the
central case here is two runs with *identical* accuracy that fail a different
set of questions. The rest pin the bucket boundaries: a question present in only
one report is new or dropped rather than fixed or regressed, and a report whose
questions are ambiguous (duplicated) is rejected instead of silently diffed.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "evals"))

import evaluate  # noqa: E402


def _report(backend: str, statuses: dict[str, str]) -> dict:
    """Build a minimal report payload: question -> status."""
    passed = sum(1 for status in statuses.values() if status == evaluate.PASS)
    total = len(statuses)
    return {
        "backend": backend,
        "total": total,
        "passed": passed,
        "execution_accuracy": (passed / total) if total else 0.0,
        "questions": [
            {"question": question, "status": status} for question, status in statuses.items()
        ],
    }


def test_equal_accuracy_can_still_hide_a_regression():
    baseline = _report("offline", {"a": evaluate.PASS, "b": evaluate.MISMATCH})
    current = _report("offline", {"a": evaluate.MISMATCH, "b": evaluate.PASS})

    comparison = evaluate.compare_reports(baseline, current)

    assert comparison.baseline_accuracy == comparison.current_accuracy
    assert comparison.regressed == ["a"]
    assert comparison.fixed == ["b"]
    assert comparison.has_regression


def test_unchanged_run_reports_no_changes():
    report = _report("offline", {"a": evaluate.PASS, "b": evaluate.PASS})

    comparison = evaluate.compare_reports(report, report)

    assert (comparison.regressed, comparison.fixed) == ([], [])
    assert (comparison.added, comparison.removed) == ([], [])
    assert not comparison.has_regression
    assert "no per-question changes" in evaluate.format_comparison(comparison)


def test_persistent_failure_is_neither_regression_nor_fix():
    baseline = _report("offline", {"a": evaluate.ERROR})
    current = _report("offline", {"a": evaluate.MISMATCH})

    comparison = evaluate.compare_reports(baseline, current)

    assert comparison.still_failing == ["a"]
    assert (comparison.regressed, comparison.fixed) == ([], [])


def test_questions_present_in_only_one_report_are_added_or_removed():
    baseline = _report("offline", {"kept": evaluate.PASS, "dropped": evaluate.PASS})
    current = _report("offline", {"kept": evaluate.PASS, "brand new": evaluate.MISMATCH})

    comparison = evaluate.compare_reports(baseline, current)

    assert comparison.added == ["brand new"]
    assert comparison.removed == ["dropped"]
    # A newly added question that fails is not a regression: nothing broke, the
    # gold set simply grew to cover something the backend never handled.
    assert comparison.regressed == []


def test_duplicate_questions_are_rejected():
    report = _report("offline", {"a": evaluate.PASS})
    report["questions"].append({"question": "a", "status": evaluate.MISMATCH})

    with pytest.raises(ValueError, match="duplicate question"):
        evaluate.compare_reports(report, report)


def test_malformed_report_is_rejected():
    with pytest.raises(ValueError, match="no 'questions' list"):
        evaluate.compare_reports({}, _report("offline", {"a": evaluate.PASS}))


def test_backend_mismatch_is_noted_not_hidden():
    baseline = _report("offline", {"a": evaluate.PASS})
    current = _report("llm", {"a": evaluate.PASS})

    text = evaluate.format_comparison(evaluate.compare_reports(baseline, current))

    assert "backends differ" in text
    assert "offline -> llm" in text


def test_formatted_comparison_lists_each_changed_question():
    baseline = _report("offline", {"a": evaluate.PASS, "b": evaluate.MISMATCH})
    current = _report("offline", {"a": evaluate.MISMATCH, "b": evaluate.PASS, "c": evaluate.PASS})

    text = evaluate.format_comparison(evaluate.compare_reports(baseline, current))

    assert "[REGRESSED] a" in text
    assert "[fixed] b" in text
    assert "[new question] c" in text


def test_compare_flag_reads_a_report_written_by_json_flag(tmp_path):
    """A report serialized to disk round-trips back into a comparable payload."""
    report = _report("offline", {"a": evaluate.PASS})
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    loaded = evaluate.load_report(str(path))

    assert evaluate.compare_reports(loaded, report).regressed == []


def test_a_baseline_that_is_not_a_json_object_is_rejected_by_name(tmp_path):
    """Pointing ``--compare`` at the wrong JSON file fails where the mistake is.

    ``evals/`` holds several JSON and JSONL files, so the plausible slip is
    naming one of those instead of a ``--json`` report. ``json.load`` accepts a
    list or a bare string quite happily, and without this check the run would
    fail frames later inside ``_passed_by_question`` complaining about a missing
    key — which reads as a malformed report rather than the wrong file.
    """
    path = tmp_path / "not-a-report.json"
    path.write_text(json.dumps([{"question": "a"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="expected a JSON object"):
        evaluate.load_report(str(path))


def test_missing_baseline_file_exits_with_a_clear_code(tmp_path, capsys, monkeypatch):
    """A baseline path that does not exist is a usage error, not a crash.

    The run itself is stubbed out: this test is about argument handling, so
    building and querying a real database would only slow it down.
    """
    db_path = tmp_path / "store.db"
    db_path.write_bytes(b"")
    monkeypatch.setattr(evaluate, "evaluate", lambda *args, **kwargs: [])

    code = evaluate.main(["--db", str(db_path), "--compare", str(tmp_path / "missing.json")])

    assert code == 2
    assert "Baseline report not found" in capsys.readouterr().err
