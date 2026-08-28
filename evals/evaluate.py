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

Execution accuracy measures only *recall*: of the questions the catalog claims,
how many does it answer correctly. It cannot see the opposite failure. The
offline backend resolves a question by first-match over an ordered list of
regexes, so a rule phrased broadly enough to catch a phrasing variant can also
start answering a question it was never meant to — returning a well-formed,
correctly-labelled table that answers a *different* question. Nothing in a
gold-set score moves when that happens, because the mis-answered question is not
in the gold set.

The precision guard (``evals/precision.jsonl``) is the other half of the
measurement: questions this database cannot answer, which the offline backend
must therefore decline. It is checked at the routing layer — no rule may match —
and any match is reported with the rule that produced it and fails the run.

The paraphrase set (``evals/paraphrases.jsonl``) measures a third property the
other two cannot see: *stability of routing under rephrasing*. The gold set
carries exactly one phrasing per rule, so a rule that only matches its own gold
question still scores 100%. Each paraphrase record pairs a gold question with an
alternate phrasing and asserts the two route to the same rule — which is what
breaks when a newly added, broadly phrased rule is inserted ahead of an older one
and quietly takes over some of its phrasings.

A record may instead carry a ``known_gap`` explaining why that phrasing does
*not* route correctly today. Those are recorded rather than omitted: a set built
only from phrasings that already work would measure nothing about the matcher's
reach. They are reported but do not fail the run, and a known gap that starts
routing correctly is flagged so it can be promoted to a gating pair.

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
PRECISION_PATH = os.path.join(os.path.dirname(__file__), "precision.jsonl")
PARAPHRASE_PATH = os.path.join(os.path.dirname(__file__), "paraphrases.jsonl")

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


def build_report(
    results: list[QuestionResult],
    backend_name: str,
    guards: list[GuardResult] | None = None,
    paraphrases: list[ParaphraseResult] | None = None,
) -> dict:
    """Assemble a JSON-serializable report from evaluated questions.

    ``execution_accuracy`` is a fraction rather than a rounded percentage so a
    machine consumer keeps full precision; the CLI does its own rounding for
    display.

    ``guards`` is optional because the precision check only applies to the
    offline backend, which is the only one with a rule catalog to over-match
    with. When omitted the ``precision`` key is absent rather than zero-filled,
    so a consumer can tell "not checked" from "checked, nothing matched"; the
    counts are named ``checked``/``unexpected_matches`` rather than reusing
    ``total``/``passed``, which belong to the accuracy metric and mean something
    different.

    ``paraphrases`` is optional for the same reason and follows the same naming
    rule. ``gating`` counts only the pairs that can fail the run, so a pair moved
    into ``known_gaps`` leaves the ``routed``/``gating`` ratio rather than
    improving it.
    """
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    report = {
        "backend": backend_name,
        "total": total,
        "passed": passed,
        "execution_accuracy": (passed / total) if total else 0.0,
        "questions": [asdict(result) for result in results],
    }
    if guards is not None:
        report["precision"] = {
            "checked": len(guards),
            "unexpected_matches": sum(1 for guard in guards if not guard.passed),
            "guards": [asdict(guard) for guard in guards],
        }
    if paraphrases is not None:
        gating = [item for item in paraphrases if item.known_gap is None]
        gaps = [item for item in paraphrases if item.known_gap is not None]
        report["robustness"] = {
            "gating": len(gating),
            "routed": sum(1 for item in gating if item.routed_alike),
            "known_gaps": len(gaps),
            "recovered_gaps": sum(1 for item in gaps if item.recovered),
            "paraphrases": [asdict(item) for item in paraphrases],
        }
    return report


# --------------------------------------------------------------------------- #
# Precision guard: questions the offline catalog must decline
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GuardResult:
    """The outcome of checking one guard question against the rule catalog.

    ``matched_rule`` and ``matched_pattern`` are ``None`` on a pass — nothing
    matched, which is the required outcome — and name the offending rule on a
    failure. The pattern is carried alongside the index because the index alone
    does not tell you which rule to go and narrow.
    """

    question: str
    reason: str
    matched_rule: int | None = None
    matched_pattern: str | None = None

    @property
    def passed(self) -> bool:
        return self.matched_rule is None


def load_guard_set(path: str = PRECISION_PATH) -> list[dict]:
    """Load ``(question, reason)`` guard records from a JSONL file.

    Each record must carry both fields. The ``reason`` is not decoration: a
    guard question is only meaningful if it says *why* the answer cannot exist,
    and without one a future reader cannot tell a deliberate out-of-scope
    question from a coverage gap someone meant to fill. A record missing either
    field raises rather than being skipped, for the same reason the gold loader
    raises: a silently dropped guard is a check that stops running while still
    appearing to.
    """
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("question", "reason"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise ValueError(
                        f"{path}:{lineno}: guard record needs a non-empty "
                        f"{field!r} field"
                    )
            records.append(record)
    return records


def check_precision(
    backend: llm.OfflineBackend, guards: list[dict]
) -> list[GuardResult]:
    """Check that no rule in ``backend`` matches any guard question.

    Deliberately inspects routing (``matching_rule_indexes``) rather than
    calling ``to_sql`` and catching ``NoRuleMatchError``. The two agree today —
    ``to_sql`` is implemented on top of the same call — but routing is the
    property under test, and asking for it directly means a failure can name the
    rule that matched instead of only reporting that something did.
    """
    results: list[GuardResult] = []
    for record in guards:
        question = record["question"]
        matches = backend.matching_rule_indexes(question)
        results.append(
            GuardResult(
                question=question,
                reason=record["reason"],
                matched_rule=matches[0] if matches else None,
                matched_pattern=(
                    backend.rule_pattern(matches[0]) if matches else None
                ),
            )
        )
    return results


def format_precision(results: list[GuardResult]) -> str:
    """Render the guard outcome as a console block."""
    failures = [result for result in results if not result.passed]
    noun = "match" if len(failures) == 1 else "matches"
    lines = [
        f"Rule precision: {len(results)} guard questions, "
        f"{len(failures)} unexpected {noun}"
    ]
    for result in failures:
        lines.append(f"  [MATCHED] {result.question}")
        lines.append(f"      rule #{result.matched_rule}: {result.matched_pattern}")
        lines.append(f"      should decline because {result.reason}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Paraphrase robustness: rephrasings must route to their canonical rule
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParaphraseResult:
    """The outcome of routing one paraphrase against its canonical question.

    Both rule indexes are recorded, not just whether they agree: on a failure
    the useful fact is *which* rule captured the paraphrase (or that nothing
    did), and that is the difference between "a broader rule is shadowing this
    one" and "the vocabulary simply is not covered".

    ``known_gap`` documents a phrasing the catalog is known not to reach. It
    holds the reason rather than a bare flag, because a gap with no explanation
    is indistinguishable from one nobody has looked at.
    """

    canonical: str
    paraphrase: str
    canonical_rule: int | None
    paraphrase_rule: int | None
    known_gap: str | None = None

    @property
    def routed_alike(self) -> bool:
        """Whether the paraphrase reached the same rule as its canonical."""
        return self.canonical_rule is not None and self.canonical_rule == self.paraphrase_rule

    @property
    def recovered(self) -> bool:
        """A known gap that now routes correctly and should be promoted."""
        return self.known_gap is not None and self.routed_alike

    @property
    def passed(self) -> bool:
        """Whether this record leaves the run green.

        A known gap passes by construction — it is a documented shortfall, not
        a regression. The one exception is a canonical question that routes
        nowhere: the pair is then measuring nothing at all, so it fails
        regardless of how the paraphrase is labelled.
        """
        if self.canonical_rule is None:
            return False
        return self.routed_alike or self.known_gap is not None


def load_paraphrase_set(path: str = PARAPHRASE_PATH) -> list[dict]:
    """Load ``(canonical, paraphrase[, known_gap])`` records from a JSONL file.

    ``canonical`` and ``paraphrase`` are required and must be non-empty;
    ``known_gap``, when present, must be a non-empty explanation. A malformed
    record raises rather than being skipped, for the same reason the gold and
    guard loaders do: a check that silently stops running still reports a clean
    result.
    """
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("canonical", "paraphrase"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise ValueError(
                        f"{path}:{lineno}: paraphrase record needs a non-empty "
                        f"{field!r} field"
                    )
            gap = record.get("known_gap")
            if gap is not None and (not isinstance(gap, str) or not gap.strip()):
                raise ValueError(
                    f"{path}:{lineno}: 'known_gap' must be a non-empty reason when present"
                )
            records.append(record)
    return records


def check_paraphrases(
    backend: llm.OfflineBackend, records: list[dict]
) -> list[ParaphraseResult]:
    """Route every canonical/paraphrase pair and record where each landed.

    Like :func:`check_precision`, this inspects routing directly rather than
    comparing the SQL two questions produce. Two rules can emit identical SQL
    today and diverge tomorrow, so equal SQL would let a paraphrase drift onto
    the wrong rule without the check noticing; the rule index is the property
    that actually has to hold.
    """
    results: list[ParaphraseResult] = []
    for record in records:
        canonical, paraphrase = record["canonical"], record["paraphrase"]
        canonical_matches = backend.matching_rule_indexes(canonical)
        paraphrase_matches = backend.matching_rule_indexes(paraphrase)
        results.append(
            ParaphraseResult(
                canonical=canonical,
                paraphrase=paraphrase,
                canonical_rule=canonical_matches[0] if canonical_matches else None,
                paraphrase_rule=paraphrase_matches[0] if paraphrase_matches else None,
                known_gap=record.get("known_gap"),
            )
        )
    return results


def _describe_routing(result: ParaphraseResult) -> str:
    """One line saying where a paraphrase went instead of its canonical rule."""
    if result.paraphrase_rule is None:
        return f"no rule matched (canonical routes to #{result.canonical_rule})"
    return (
        f"routed to rule #{result.paraphrase_rule}, "
        f"canonical routes to #{result.canonical_rule}"
    )


def format_paraphrases(results: list[ParaphraseResult]) -> str:
    """Render the paraphrase outcome as a console block.

    Gating pairs and known gaps are counted separately so the headline cannot be
    inflated by documenting a failure: adding a ``known_gap`` moves a pair out of
    the gating count instead of into its numerator.
    """
    gating = [result for result in results if result.known_gap is None]
    gaps = [result for result in results if result.known_gap is not None]
    routed = sum(1 for result in gating if result.routed_alike)

    lines = [
        f"Paraphrase robustness: {routed}/{len(gating)} rephrasings route to "
        f"the canonical rule"
    ]
    for result in gating:
        if not result.passed:
            lines.append(f"  [MISROUTED] {result.paraphrase}")
            lines.append(f"      {_describe_routing(result)}")
            lines.append(f"      canonical: {result.canonical}")

    if gaps:
        recovered = [result for result in gaps if result.recovered]
        lines.append(f"  Known gaps (not gating): {len(gaps)}")
        for result in recovered:
            lines.append(
                f"    [NOW ROUTING] {result.paraphrase} — drop its 'known_gap' "
                f"to make it a gating pair"
            )
        for result in gaps:
            if not result.passed:
                lines.append(f"    [BROKEN PAIR] {result.paraphrase}")
                lines.append(f"        canonical question matches no rule: {result.canonical}")
    return "\n".join(lines)


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
        "--precision",
        metavar="PATH",
        default=PRECISION_PATH,
        help=(
            "JSONL file of questions the offline backend must decline "
            "(default: evals/precision.jsonl); ignored with --llm"
        ),
    )
    parser.add_argument(
        "--paraphrases",
        metavar="PATH",
        default=PARAPHRASE_PATH,
        help=(
            "JSONL file of rephrasings that must route to the same offline rule "
            "as their canonical question (default: evals/paraphrases.jsonl); "
            "ignored with --llm"
        ),
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

    # The guard checks a rule catalog, so it only means anything for the offline
    # backend. An LLM always emits *something*, and whether it should have
    # refused is a judgement about the SQL rather than about routing — a
    # different measurement that would need a different harness.
    # Both offline-only checks read the same rule catalog, so they share one
    # backend instance rather than each building its own.
    offline = None if args.llm else llm.OfflineBackend()
    guards = (
        None
        if offline is None
        else check_precision(offline, load_guard_set(args.precision))
    )
    # Routing stability is a property of a fixed rule catalog. An LLM re-reads
    # the question every time and has no rule to route to, so "did these two
    # phrasings reach the same rule" has no meaning for it.
    paraphrases = (
        None
        if offline is None
        else check_paraphrases(offline, load_paraphrase_set(args.paraphrases))
    )
    report = build_report(results, backend_name, guards, paraphrases)

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

    if guards is not None:
        print()
        print(format_precision(guards))

    if paraphrases is not None:
        print()
        print(format_paraphrases(paraphrases))

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
    #
    # The precision guard *does* gate, and independently: a rule that started
    # answering an out-of-scope question moves no accuracy number, so without
    # its own condition here the run would exit 0 on exactly the regression the
    # guard exists to catch.
    guard_failed = guards is not None and any(not guard.passed for guard in guards)
    # Gates for the same reason the precision guard does: a paraphrase that
    # drifts onto another rule still returns a well-formed table, so no accuracy
    # number moves when it breaks.
    routing_failed = paraphrases is not None and any(
        not item.passed for item in paraphrases
    )
    return 0 if passed == total and not guard_failed and not routing_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
