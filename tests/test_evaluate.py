"""Tests for the evaluation harness's result-set comparison and reporting.

The harness is the project's headline metric, so the comparison it performs is
worth testing directly: a comparator that is too lenient reports an accuracy
that is not real. The per-question diagnostics are tested for the same reason —
a failure message that points at the wrong row is worse than no message, so the
tests pin down both *that* a difference is detected and *what* it says.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "evals"))

import evaluate  # noqa: E402

from nl2sql import runner  # noqa: E402

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")
GOLD_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evals", "gold.jsonl")


def _result(rows):
    return runner.QueryResult(columns=["a", "b"], rows=rows)


class _FixedBackend:
    """A backend that always returns the same SQL, whatever the question."""

    def __init__(self, sql: str) -> None:
        self._sql = sql

    def to_sql(self, question: str, schema: str) -> str:  # noqa: ARG002 - Backend protocol
        return self._sql


class _RaisingBackend:
    """A backend that fails to generate, standing in for an API/parse failure."""

    def to_sql(self, question: str, schema: str) -> str:  # noqa: ARG002 - Backend protocol
        raise RuntimeError("backend unavailable")


# --------------------------------------------------------------------------- #
# _result_key
# --------------------------------------------------------------------------- #
def test_unordered_comparison_ignores_row_order():
    a = _result([("North", 3), ("South", 1)])
    b = _result([("South", 1), ("North", 3)])
    assert evaluate._result_key(a) == evaluate._result_key(b)


def test_ordered_comparison_detects_row_order():
    """The same rows in a different order must NOT compare equal when ordered."""
    a = _result([("North", 3), ("South", 1)])
    b = _result([("South", 1), ("North", 3)])
    assert evaluate._result_key(a, ordered=True) != evaluate._result_key(b, ordered=True)


def test_ordered_comparison_accepts_identical_order():
    rows = [("North", 3), ("South", 1)]
    assert evaluate._result_key(_result(rows), ordered=True) == evaluate._result_key(
        _result(list(rows)), ordered=True
    )


def test_comparison_defaults_to_unordered():
    """Omitting the flag keeps the historical, order-insensitive behavior."""
    a = _result([("North", 3), ("South", 1)])
    b = _result([("South", 1), ("North", 3)])
    assert evaluate._result_key(a) == evaluate._result_key(b, ordered=False)


def test_comparison_detects_differing_values_either_way():
    a = _result([("North", 3)])
    b = _result([("North", 4)])
    assert evaluate._result_key(a) != evaluate._result_key(b)
    assert evaluate._result_key(a, ordered=True) != evaluate._result_key(b, ordered=True)


def test_comparison_ignores_column_names():
    """Aliasing a column differently is a phrasing difference, not an error."""
    a = runner.QueryResult(columns=["revenue"], rows=[(10,)])
    b = runner.QueryResult(columns=["total_revenue"], rows=[(10,)])
    assert evaluate._result_key(a) == evaluate._result_key(b)


# --------------------------------------------------------------------------- #
# describe_difference
# --------------------------------------------------------------------------- #
def test_identical_keys_report_no_difference():
    key = [("North", "3"), ("South", "1")]
    assert evaluate.describe_difference(key, list(key), ordered=True) is None


def test_row_count_difference_is_reported_on_its_own():
    """A length mismatch is the finding; positional diffs after it are noise."""
    detail = evaluate.describe_difference(
        [("North", "3")], [("North", "3"), ("South", "1")], ordered=True
    )
    assert detail == "row count differs: generated 1, gold 2"


def test_first_differing_row_is_identified_by_index():
    detail = evaluate.describe_difference(
        [("North", "3"), ("South", "9")],
        [("North", "3"), ("South", "1")],
        ordered=True,
    )
    assert "index 1" in detail
    assert "(South, 9)" in detail
    assert "(South, 1)" in detail


def test_difference_message_names_the_comparison_ordering():
    """The index is only readable against the ordering it was computed in."""
    ordered = evaluate.describe_difference([("a",)], [("b",)], ordered=True)
    unordered = evaluate.describe_difference([("a",)], [("b",)], ordered=False)
    assert "as returned" in ordered
    assert "in sorted order" in unordered


def test_wide_rows_are_truncated_in_the_message():
    wide_generated = tuple(str(n) for n in range(10))
    wide_gold = tuple(str(n + 100) for n in range(10))
    detail = evaluate.describe_difference([wide_generated], [wide_gold], ordered=True)
    assert "+4 more columns" in detail
    # The cells past the cap must not be printed.
    assert "109" not in detail


# --------------------------------------------------------------------------- #
# evaluate_question / build_report
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sample_db() -> str:
    if not os.path.exists(DB):
        pytest.skip("sample DB not built")
    return DB


def test_matching_query_passes(sample_db: str):
    gold_sql = "SELECT COUNT(*) AS customer_count FROM customers"
    result = evaluate.evaluate_question(
        sample_db,
        _FixedBackend("SELECT COUNT(*) AS n FROM customers"),
        "",
        {"question": "How many customers?", "sql": gold_sql, "ordered": False},
    )
    assert result.status == evaluate.PASS
    assert result.passed
    assert result.detail is None
    assert result.generated_rows == result.gold_rows == 1


def test_wrong_query_is_a_mismatch_with_a_diagnostic(sample_db: str):
    result = evaluate.evaluate_question(
        sample_db,
        _FixedBackend("SELECT COUNT(*) FROM products"),
        "",
        {
            "question": "How many customers?",
            "sql": "SELECT COUNT(*) FROM customers",
            "ordered": False,
        },
    )
    assert result.status == evaluate.MISMATCH
    assert not result.passed
    assert "first differing row" in result.detail


def test_generation_failure_is_recorded_as_an_error(sample_db: str):
    result = evaluate.evaluate_question(
        sample_db,
        _RaisingBackend(),
        "",
        {"question": "anything", "sql": "SELECT 1", "ordered": False},
    )
    assert result.status == evaluate.ERROR
    assert result.generated_sql is None
    assert "RuntimeError" in result.detail
    # Nothing ran, so no row counts are claimed.
    assert result.generated_rows is None and result.gold_rows is None


def test_unsafe_generated_sql_is_recorded_as_an_error(sample_db: str):
    """The runner's guardrails surface as a failed question, not a crash."""
    result = evaluate.evaluate_question(
        sample_db,
        _FixedBackend("DROP TABLE customers"),
        "",
        {"question": "anything", "sql": "SELECT 1", "ordered": False},
    )
    assert result.status == evaluate.ERROR
    assert result.generated_sql == "DROP TABLE customers"


def test_report_is_json_serializable_and_counts_correctly():
    results = [
        evaluate.QuestionResult(
            question="q1", ordered=False, status=evaluate.PASS, gold_sql="SELECT 1"
        ),
        evaluate.QuestionResult(
            question="q2",
            ordered=True,
            status=evaluate.MISMATCH,
            gold_sql="SELECT 2",
            detail="row count differs: generated 0, gold 1",
        ),
    ]
    report = evaluate.build_report(results, "offline")

    assert report["backend"] == "offline"
    assert (report["total"], report["passed"]) == (2, 1)
    assert report["execution_accuracy"] == 0.5
    # Round-tripping through JSON is the actual contract of --json.
    round_tripped = json.loads(json.dumps(report))
    assert [q["question"] for q in round_tripped["questions"]] == ["q1", "q2"]
    assert round_tripped["questions"][1]["status"] == evaluate.MISMATCH


def test_empty_report_does_not_divide_by_zero():
    report = evaluate.build_report([], "offline")
    assert report["execution_accuracy"] == 0.0


def test_json_flag_writes_a_report(tmp_path, sample_db: str):
    out = tmp_path / "report.json"
    exit_code = evaluate.main(["--db", sample_db, "--json", str(out)])
    assert exit_code == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    gold = evaluate.load_gold(GOLD_PATH)
    assert report["total"] == len(gold)
    assert report["passed"] == len(gold)
    assert report["execution_accuracy"] == 1.0
    assert {q["status"] for q in report["questions"]} == {evaluate.PASS}


# --------------------------------------------------------------------------- #
# gold.jsonl consistency
# --------------------------------------------------------------------------- #
def test_gold_rows_have_required_fields():
    gold = evaluate.load_gold(GOLD_PATH)
    assert gold, "gold.jsonl is empty"
    for item in gold:
        assert item.get("question"), f"missing question: {item}"
        assert item.get("sql"), f"missing sql: {item}"
        assert isinstance(item.get("ordered"), bool), (
            f"'ordered' must be present and boolean: {item['question']}"
        )


def test_ordered_gold_rows_have_an_order_by():
    """An order-sensitive expectation is only meaningful if the gold SQL sorts.

    The converse is not asserted: a gold query may sort purely for readable
    output (e.g. one row per region, listed alphabetically) without the order
    being part of the answer, and those rows are correctly left unordered.
    """
    for item in evaluate.load_gold(GOLD_PATH):
        if item["ordered"]:
            assert "order by" in item["sql"].lower(), (
                f"marked ordered but gold SQL has no ORDER BY: {item['question']}"
            )


def test_ordered_flag_has_teeth_against_the_real_database():
    """Reversing an ordered result must break the comparison.

    This is what proves the flag does real work: it runs each order-sensitive
    gold query against the sample database and checks that a permutation of its
    own rows no longer matches. Single-row results are skipped -- there is no
    permutation to detect.
    """
    if not os.path.exists(DB):
        pytest.skip("sample DB not built")

    checked = 0
    for item in evaluate.load_gold(GOLD_PATH):
        if not item["ordered"]:
            continue
        res = runner.run(DB, item["sql"])
        if len(res.rows) < 2:
            continue
        reversed_res = runner.QueryResult(columns=res.columns, rows=list(reversed(res.rows)))
        assert evaluate._result_key(res, ordered=True) != evaluate._result_key(
            reversed_res, ordered=True
        ), f"ordered comparison did not detect reordering: {item['question']}"
        # ...while the unordered comparison would have accepted it.
        assert evaluate._result_key(res) == evaluate._result_key(reversed_res)
        checked += 1

    assert checked > 0, "no multi-row ordered gold rows were exercised"


def test_gold_file_is_valid_jsonl():
    with open(GOLD_PATH, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                json.loads(line)  # raises on malformed JSON
