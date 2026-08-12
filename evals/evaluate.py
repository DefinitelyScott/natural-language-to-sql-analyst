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

Every question produces a :class:`QuestionResult` rather than just a pass/fail
tally. A bare accuracy number tells you *that* a backend regressed; the per
question record — the SQL it generated, whether the run errored or merely
disagreed, and the first row where the two result sets diverge — is what tells
you *why*, without re-running anything by hand. ``--json`` writes those records
to a file so a CI run can archive them, and ``--compare`` diffs the current run
against such a file: two runs can post the same accuracy while failing a
different set of questions, so a per-question diff is the only way to see a
regression and a fix that cancel each other out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nl2sql import llm, runner, schema  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")
GOLD_PATH = os.path.join(os.path.dirname(__file__), "gold.jsonl")

#: A question whose generated result set matched the gold result set.
PASS = "pass"
#: Both queries ran, but their result sets disagree.
MISMATCH = "mismatch"
#: Generation or execution raised — the backend produced nothing comparable.
ERROR = "error"

# How many rows of a differing row to show in the diagnostic, so a wide result
# set does not print an unreadable line.
_MAX_DIFF_CELLS = 6


@dataclass(frozen=True)
class QuestionResult:
    """The outcome of evaluating one gold question.

    ``generated_sql`` is ``None`` only when generation itself raised, and the
    row counts are ``None`` whenever the corresponding query never ran — those
    are the cases where there is genuinely nothing to report, and encoding them
    as ``None`` keeps a failed run from being mistaken for an empty result set.
    """

    question: str
    ordered: bool
    status: str
    gold_sql: str
    generated_sql: str | None = None
    generated_rows: int | None = None
    gold_rows: int | None = None
    detail: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == PASS


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


def _abbreviate(row: tuple[str, ...]) -> str:
    """Render a row for a diagnostic message, truncating very wide rows."""
    cells = list(row[:_MAX_DIFF_CELLS])
    if len(row) > _MAX_DIFF_CELLS:
        cells.append(f"... (+{len(row) - _MAX_DIFF_CELLS} more columns)")
    return "(" + ", ".join(cells) + ")"


def describe_difference(
    generated: list[tuple[str, ...]],
    gold: list[tuple[str, ...]],
    *,
    ordered: bool,
) -> str | None:
    """Explain the first way two comparison keys differ, or ``None`` if equal.

    The keys are the output of :func:`_result_key`, so for an unordered question
    they are already sorted — the "first" differing row is therefore the first
    in sorted order, not the first the query returned. The message says which,
    because reading a positional index against the wrong ordering is exactly the
    kind of thing that sends you debugging the wrong row.

    A row-count difference is reported on its own: when one result set is a
    different length, the first positional disagreement is usually an artifact
    of the misalignment rather than the actual defect.
    """
    if generated == gold:
        return None
    if len(generated) != len(gold):
        return f"row count differs: generated {len(generated)}, gold {len(gold)}"

    order_note = "as returned" if ordered else "in sorted order"
    # strict=True is safe here — the length check above already returned on a
    # mismatch — and documents that the walk assumes aligned result sets.
    for index, (got, want) in enumerate(zip(generated, gold, strict=True)):
        if got != want:
            return (
                f"first differing row ({order_note}, index {index}): "
                f"generated {_abbreviate(got)} vs gold {_abbreviate(want)}"
            )
    # Same length and no differing row means the keys were equal, which the
    # first branch already returned. Reaching here would be a logic error.
    raise AssertionError("result keys differ but no differing row was found")


def load_gold(path: str) -> list[dict]:
    """Load gold (question, sql, ordered) records from a JSONL file."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate_question(
    db_path: str,
    backend: llm.Backend,
    schema_text: str,
    item: dict,
) -> QuestionResult:
    """Generate, execute and compare SQL for one gold record."""
    question, gold_sql = item["question"], item["sql"]
    # Absent means unordered: only a question whose answer is a ranking or a
    # sequence opts in to the stricter, order-sensitive comparison.
    ordered = bool(item.get("ordered", False))

    generated_sql: str | None = None
    try:
        generated_sql = backend.to_sql(question, schema_text)
        gen = runner.run(db_path, generated_sql)
        ref = runner.run(db_path, gold_sql)
    except Exception as exc:  # noqa: BLE001 - any backend/DB failure is a failed question
        return QuestionResult(
            question=question,
            ordered=ordered,
            status=ERROR,
            gold_sql=gold_sql,
            generated_sql=generated_sql,
            detail=f"{type(exc).__name__}: {exc}",
        )

    difference = describe_difference(
        _result_key(gen, ordered=ordered),
        _result_key(ref, ordered=ordered),
        ordered=ordered,
    )
    return QuestionResult(
        question=question,
        ordered=ordered,
        status=PASS if difference is None else MISMATCH,
        gold_sql=gold_sql,
        generated_sql=generated_sql,
        generated_rows=len(gen),
        gold_rows=len(ref),
        detail=difference,
    )


def evaluate(db_path: str, use_llm: bool, *, gold_path: str = GOLD_PATH) -> list[QuestionResult]:
    """Evaluate every gold question and return one result per question."""
    gold = load_gold(gold_path)
    schema_text = schema.schema_context(db_path)
    backend = llm.get_backend(use_llm)
    return [evaluate_question(db_path, backend, schema_text, item) for item in gold]


def build_report(results: list[QuestionResult], backend_name: str) -> dict:
    """Assemble a JSON-serializable report from evaluated questions.

    ``execution_accuracy`` is a fraction rather than a rounded percentage so a
    machine consumer keeps full precision; the CLI does its own rounding for
    display.
    """
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    return {
        "backend": backend_name,
        "total": total,
        "passed": passed,
        "execution_accuracy": (passed / total) if total else 0.0,
        "questions": [asdict(result) for result in results],
    }


@dataclass(frozen=True)
class ReportComparison:
    """How the per-question outcomes of two runs differ.

    Every shared question falls into exactly one of ``regressed``, ``fixed``,
    ``still_failing`` or "still passing" (which is not listed: a question that
    passed in both runs is the uninteresting case and printing it would bury
    the four that matter). ``added`` and ``removed`` cover questions present in
    only one of the two reports, which is what happens when the gold set grows
    between runs.
    """

    baseline_backend: str
    current_backend: str
    baseline_accuracy: float
    current_accuracy: float
    regressed: list[str]
    fixed: list[str]
    still_failing: list[str]
    added: list[str]
    removed: list[str]

    @property
    def has_regression(self) -> bool:
        return bool(self.regressed)


def _passed_by_question(report: dict, label: str) -> dict[str, bool]:
    """Index a report's questions by text, mapping each to whether it passed.

    Raises ``ValueError`` on a duplicate question, because the diff keys on the
    question text: with two records under one key there is no unambiguous
    answer to "did this question pass", and silently keeping the last one would
    make the comparison quietly wrong rather than loudly broken. The gold set
    has no duplicates (``tests/test_rule_catalog.py`` pins one rule per
    question), so this only fires on a hand-edited or merged report.
    """
    questions = report.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"{label} report has no 'questions' list")

    passed: dict[str, bool] = {}
    for record in questions:
        question = record["question"]
        if question in passed:
            raise ValueError(f"{label} report contains duplicate question: {question!r}")
        passed[question] = record["status"] == PASS
    return passed


def compare_reports(baseline: dict, current: dict) -> ReportComparison:
    """Diff two ``build_report`` payloads question by question."""
    before = _passed_by_question(baseline, "baseline")
    after = _passed_by_question(current, "current")
    shared = before.keys() & after.keys()

    return ReportComparison(
        baseline_backend=baseline.get("backend", "unknown"),
        current_backend=current.get("backend", "unknown"),
        baseline_accuracy=float(baseline.get("execution_accuracy", 0.0)),
        current_accuracy=float(current.get("execution_accuracy", 0.0)),
        regressed=sorted(q for q in shared if before[q] and not after[q]),
        fixed=sorted(q for q in shared if not before[q] and after[q]),
        still_failing=sorted(q for q in shared if not before[q] and not after[q]),
        added=sorted(after.keys() - before.keys()),
        removed=sorted(before.keys() - after.keys()),
    )


def format_comparison(comparison: ReportComparison) -> str:
    """Render a comparison as a console block.

    Accuracy alone cannot answer "did anything break": two runs can score the
    same while failing a different set of questions, so a headline that only
    moved from 30/38 to 30/38 hides a regression and a fix cancelling out. The
    per-question buckets are printed for exactly that case.
    """
    lines = [
        f"Comparison vs baseline  |  accuracy "
        f"{comparison.baseline_accuracy:.1%} -> {comparison.current_accuracy:.1%}"
    ]
    if comparison.baseline_backend != comparison.current_backend:
        # Not an error: benchmarking one backend against another is a fair use
        # of this diff. But the two runs then differ by more than the change
        # under test, so the reader is told rather than left to infer it.
        lines.append(
            f"  note: backends differ "
            f"({comparison.baseline_backend} -> {comparison.current_backend})"
        )

    buckets = (
        ("REGRESSED", comparison.regressed),
        ("fixed", comparison.fixed),
        ("still failing", comparison.still_failing),
        ("new question", comparison.added),
        ("dropped question", comparison.removed),
    )
    if not any(questions for _, questions in buckets):
        lines.append("  no per-question changes")
        return "\n".join(lines)

    for label, questions in buckets:
        for question in questions:
            lines.append(f"  [{label}] {question}")
    return "\n".join(lines)


def load_report(path: str) -> dict:
    """Load a report previously written by ``--json``."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--llm", action="store_true", help="evaluate the LLM backend")
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="also write a machine-readable per-question report to PATH",
    )
    parser.add_argument(
        "--compare",
        metavar="PATH",
        help=(
            "diff this run against a report previously written by --json, "
            "listing which questions regressed, were fixed, or are new"
        ),
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.db):
        print("Database not found. Run: python scripts/build_sample_db.py", file=sys.stderr)
        return 2

    results = evaluate(args.db, args.llm)
    backend_name = "llm" if args.llm else "offline"
    report = build_report(results, backend_name)

    total, passed = report["total"], report["passed"]
    pct = round(100 * report["execution_accuracy"])
    print(
        f"Evaluated {total} questions  |  execution accuracy: "
        f"{passed}/{total} ({pct}%)  [{backend_name} backend]"
    )

    failures = [result for result in results if not result.passed]
    if failures:
        print("\nFailures:")
        for result in failures:
            print(f"  - {result.question}  [{result.status}: {result.detail}]")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
        print(f"\nWrote report to {args.json}")

    if args.compare:
        if not os.path.exists(args.compare):
            print(f"Baseline report not found: {args.compare}", file=sys.stderr)
            return 2
        print()
        print(format_comparison(compare_reports(load_report(args.compare), report)))

    # The comparison deliberately does not influence the exit code. A regressed
    # question is by definition failing now, so it already makes ``passed <
    # total``; gating on it again would add a second condition that can never
    # fire on its own. The diff's job is to say *which* questions changed.
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
