"""Execution-accuracy evaluation for the text-to-SQL system.

For each (question, gold_sql) pair we generate SQL with the chosen backend,
execute both the generated and the gold query, and compare the *result sets*.
This measures whether the generated query produces the correct answer, which is
the metric that actually matters for text-to-SQL — string-matching the SQL would
penalize correct-but-differently-written queries.

Comparison is order-insensitive by default: two queries that return the same
rows in a different order are the same answer for a question like "how many
customers do we have?". But for a *ranking* ("the top 5 customers by spend") or
a *sequence* ("revenue by month"), the row order is part of the answer — a
result with the right rows in the wrong order is wrong. Each gold row therefore
carries an ``ordered`` flag, and rows are compared as returned when it is set.
Comparing everything order-insensitively, as this harness used to, silently
over-credits the backend on exactly the questions where ordering is the hard
part.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nl2sql import llm, runner, schema  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")
GOLD_PATH = os.path.join(os.path.dirname(__file__), "gold.jsonl")


def _result_key(res: runner.QueryResult, *, ordered: bool = False) -> list[tuple[str, ...]]:
    """Return a comparable form of a result set.

    Values are stringified so that two answers differing only in SQLite's
    dynamic typing do not compare unequal. Column *names* are deliberately not
    compared: a correct query may alias its columns differently from the gold
    query, and penalizing that would measure phrasing rather than correctness.

    When ``ordered`` is False the rows are sorted, so row order is ignored.
    When it is True the rows are compared exactly as returned, because the
    order itself is part of the answer.
    """
    rows = [tuple(str(v) for v in row) for row in res.rows]
    return rows if ordered else sorted(rows)


def load_gold(path: str) -> list[dict]:
    """Load gold (question, sql, ordered) records from a JSONL file."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate(db_path: str, use_llm: bool) -> tuple[int, int, list[str]]:
    gold = load_gold(GOLD_PATH)
    schema_text = schema.schema_context(db_path)
    backend = llm.get_backend(use_llm)

    passed = 0
    failures: list[str] = []
    for item in gold:
        question, gold_sql = item["question"], item["sql"]
        # Absent means unordered: only a question whose answer is a ranking or
        # a sequence opts in to the stricter, order-sensitive comparison.
        ordered = bool(item.get("ordered", False))
        try:
            gen_sql = backend.to_sql(question, schema_text)
            gen = runner.run(db_path, gen_sql)
            ref = runner.run(db_path, gold_sql)
            if _result_key(gen, ordered=ordered) == _result_key(ref, ordered=ordered):
                passed += 1
            else:
                failures.append(question)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{question}  [error: {exc}]")
    return passed, len(gold), failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--llm", action="store_true", help="evaluate the LLM backend")
    args = parser.parse_args(argv)

    if not os.path.exists(args.db):
        print("Database not found. Run: python scripts/build_sample_db.py", file=sys.stderr)
        return 2

    passed, total, failures = evaluate(args.db, args.llm)
    backend_name = "llm" if args.llm else "offline"
    pct = round(100 * passed / total) if total else 0
    print(
        f"Evaluated {total} questions  |  execution accuracy: "
        f"{passed}/{total} ({pct}%)  [{backend_name} backend]"
    )
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
