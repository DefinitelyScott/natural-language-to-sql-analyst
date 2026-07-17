"""Tests for the evaluation harness's result-set comparison.

The harness is the project's headline metric, so the comparison it performs is
worth testing directly: a comparator that is too lenient reports an accuracy
that is not real.
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
        for lineno, line in enumerate(fh, 1):
            if line.strip():
                json.loads(line)  # raises on malformed JSON
