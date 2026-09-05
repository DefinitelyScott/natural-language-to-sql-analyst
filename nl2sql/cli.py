"""Command-line interface.

Commands:
    ask "<question>"      answer a natural-language question with SQL + results
    explain "<question>"  show the SQL a question resolves to, without running it
    rules [--search]      list the questions the offline backend can answer
    schema [--counts]     print the introspected database schema
"""

from __future__ import annotations

import argparse
import os
import sys

from . import catalog, generator, llm, output, runner, schema
from .runner import QueryTimeoutError, UnsafeQueryError

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
_DEFAULT_DB = os.path.join(_REPO_ROOT, "data", "store.db")
# Example questions for `rules` come from the evaluation gold set rather than a
# separate list — see nl2sql/catalog.py for why.
_DEFAULT_GOLD = os.path.join(_REPO_ROOT, "evals", "gold.jsonl")

# How many rows to show in the human-readable table preview before truncating.
# csv/json output is never truncated — an export should be complete.
_TABLE_PREVIEW_ROWS = 20

# Commands that read the database. `rules` inspects only the in-process rule
# catalog, so requiring a built database to list it would be a false dependency.
_NEEDS_DB = frozenset({"ask", "explain", "schema"})

# Shared help for --no-cache. Caching is on by default here even though the
# library defaults it off: the cache file belongs to the person running the
# command, and paying for a model call they have already paid for is not a
# sensible default for a tool. It is a no-op without --llm, since the offline
# backend is never cached.
_NO_CACHE_HELP = (
    "do not read or write the local SQL cache; regenerate the query even if an "
    "identical question, schema, model and prompt were answered before "
    "(--llm only — offline SQL is never cached)"
)


#: How many nearest catalog questions to offer when the offline backend has no
#: rule for the question asked. Three is enough to cover a near-miss in phrasing
#: without turning the error into a menu the user has to read past.
_SUGGESTION_LIMIT = 3


def _print_no_rule_help(question: str, gold_path: str) -> None:
    """Print nearest answerable questions for an unmatched question, on stderr.

    Suggestions are drawn from the rule catalog rather than straight from the
    gold file, so every question offered is one the live matcher routes to a
    rule — a suggestion that would fail the same way the user's question just
    did would be worse than none.

    A missing or malformed gold file costs the suggestions and nothing else; the
    pointer to ``nl2sql rules`` is still printed. It degrades silently on
    purpose: the user's problem here is an unrecognized question, and a second
    warning about a file they did not mention would bury the message that
    matters. ``nl2sql rules`` reports the same fault explicitly if they follow
    the pointer.
    """
    try:
        examples = catalog.load_example_questions(gold_path)
    except (OSError, ValueError):
        examples = []

    entries = catalog.build_catalog(llm.OfflineBackend(), examples)
    suggestions = catalog.suggest_questions(
        question, catalog.answerable_questions(entries), limit=_SUGGESTION_LIMIT
    )

    if suggestions:
        print("\nDid you mean:", file=sys.stderr)
        for suggestion in suggestions:
            print(f"  {suggestion}", file=sys.stderr)
    print(
        "\nRun `nl2sql rules` to list every question the offline backend "
        "answers, or `nl2sql rules --search <text>` to filter it.",
        file=sys.stderr,
    )


def _print_explanation(exp: generator.Explanation) -> int:
    """Print a dry-run report. Return the process exit code.

    An unsafe query exits 1 even though the command itself succeeded: the
    linter convention (report findings, exit non-zero) is more useful here than
    the "the command ran, so 0" one, because it lets a script gate on
    ``explain`` before ever letting the SQL near the database.
    """
    print(f"\nQuestion: {exp.question}")
    print(f"Backend:  {exp.backend}")
    if exp.cached:
        # Only printed on a hit. A dry run is what you reach for after editing
        # a prompt, and "the model was never called" is the one fact that would
        # otherwise make the output impossible to interpret.
        print("Source:   local cache (pass --no-cache to regenerate)")

    if exp.matched_rule is not None:
        print(f"\nMatched offline rule #{exp.matched_rule}: {exp.matched_pattern}")
        if exp.shadowed_rules:
            # Later rules that also match are inert — first-rule-wins. Showing
            # them is how you tell a deliberate ordering from an accidental one
            # when adding a pattern to the catalog.
            print("Also matched (shadowed, in catalog order):")
            for index, pattern in exp.shadowed_rules:
                print(f"  #{index}: {pattern}")

    print("\nSQL (not executed):")
    print("  " + exp.sql)

    if exp.is_safe:
        print("\nSafety: passes the read-only validator.\n")
        return 0
    print(f"\nSafety: REJECTED — {exp.safety_error}\n", file=sys.stderr)
    return 1


def _print_rules(
    entries: list[catalog.CatalogEntry], *, as_json: bool, searched: bool
) -> int:
    """Print the offline rule catalog. Return the process exit code.

    An empty result from a ``--search`` exits 1, following grep: a script can
    then test whether the catalog covers a topic without parsing the output. An
    empty catalog with no search would mean the backend has no rules at all,
    which is a different failure and is not this function's to diagnose.
    """
    if searched and not entries:
        print("No offline rules match that search.", file=sys.stderr)
        return 1

    columns = ("rule", "example", "pattern")
    rows = [
        (entry.index, entry.example or "(no example)", entry.pattern)
        for entry in entries
    ]

    if as_json:
        print(output.format_json(columns, rows))
        return 0

    print(f"\n{len(entries)} offline rule(s), in matching order:\n")
    # No row limit: the catalog is the answer here, not a preview of one, and
    # truncating it would defeat the point of asking what the backend covers.
    print(output.format_table(columns, rows))
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nl2sql", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer a natural-language question")
    ask.add_argument("question", help="the question, in plain English")
    ask.add_argument("--db", default=_DEFAULT_DB, help="path to the SQLite database")
    ask.add_argument("--llm", action="store_true", help="use the LLM backend")
    ask.add_argument("--no-cache", action="store_true", help=_NO_CACHE_HELP)
    ask.add_argument("--max-rows", type=int, default=1000)
    ask.add_argument(
        "--timeout-ms",
        type=int,
        default=runner.DEFAULT_TIMEOUT_MS,
        help=(
            "cancel the query if it is still running after this many "
            f"milliseconds (default: {runner.DEFAULT_TIMEOUT_MS}). Use 0 to "
            "run with no deadline. --max-rows bounds how much comes back; "
            "this bounds how long the database works to produce it."
        ),
    )
    ask.add_argument(
        "--format",
        choices=("table", "csv", "json"),
        default="table",
        help=(
            "output format (default: table). With csv/json, only the data is "
            "written to stdout so it can be redirected to a file; the generated "
            "SQL and any truncation warning go to stderr. csv/json write every "
            "row the --max-rows cap allowed through, with no preview limit of "
            "their own."
        ),
    )

    why = sub.add_parser(
        "explain",
        help="show the SQL a question resolves to, without executing it",
        description=(
            "Dry run: generate the SQL for a question and report how it was "
            "produced — which offline rule matched, which other rules matched "
            "but were shadowed by it, and whether the SQL passes the read-only "
            "safety validator. The query is never executed, so this is safe to "
            "point at SQL you do not yet trust. Exits 1 when the SQL would be "
            "rejected as unsafe, so it can be used as a pre-flight check in a "
            "script."
        ),
    )
    why.add_argument("question", help="the question, in plain English")
    why.add_argument("--db", default=_DEFAULT_DB, help="path to the SQLite database")
    why.add_argument("--llm", action="store_true", help="use the LLM backend")
    why.add_argument("--no-cache", action="store_true", help=_NO_CACHE_HELP)

    rules = sub.add_parser(
        "rules",
        help="list the questions the offline backend can answer",
        description=(
            "List the offline backend's rule catalog in matching order, with an "
            "example question for each rule. This is the discoverability "
            "counterpart to `explain`: `explain` tells you how one question you "
            "already have resolves, while `rules` tells you which questions "
            "exist to ask. Rule numbers are the same ones `explain` reports. "
            "Needs no database — it inspects the rule catalog only. Exits 1 "
            "when a --search matches nothing."
        ),
    )
    rules.add_argument(
        "--search",
        metavar="TEXT",
        help=(
            "show only rules whose example question or pattern contains TEXT "
            "(case-insensitive)"
        ),
    )
    rules.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="output format (default: table)",
    )
    rules.add_argument(
        "--gold",
        default=_DEFAULT_GOLD,
        help=(
            "JSONL file to draw example questions from "
            "(default: the evaluation gold set)"
        ),
    )

    show = sub.add_parser(
        "schema",
        help="print the database schema (tables, columns, keys)",
        description=(
            "Print the introspected schema exactly as it is rendered for the "
            "LLM prompt — useful for seeing what context the model works from, "
            "and for orienting yourself in an unfamiliar database. Text columns "
            "holding only a handful of distinct values are annotated with those "
            "values, so you can see the exact literals a WHERE clause has to "
            "match."
        ),
    )
    show.add_argument("--db", default=_DEFAULT_DB, help="path to the SQLite database")
    show.add_argument(
        "--counts",
        action="store_true",
        help="also show the number of rows in each table",
    )
    show.add_argument(
        "--no-values",
        action="store_true",
        help=(
            "omit the categorical value hints and print structure only "
            f"(hints cover text columns with at most "
            f"{schema.DEFAULT_MAX_DISTINCT} distinct values)"
        ),
    )

    args = parser.parse_args(argv)

    if args.command in _NEEDS_DB and not os.path.exists(args.db):
        print(
            f"Database not found at {args.db}. "
            "Run: python scripts/build_sample_db.py",
            file=sys.stderr,
        )
        return 2

    if args.command == "rules":
        # A missing or malformed gold file costs the listing its examples, not
        # the listing itself: the rules are the answer and they come from the
        # backend. The reason is written to stderr so the degradation is
        # visible rather than looking like a catalog with no coverage.
        try:
            examples = catalog.load_example_questions(args.gold)
        except (OSError, ValueError) as exc:
            print(f"Warning: no example questions ({exc})", file=sys.stderr)
            examples = []

        entries = catalog.build_catalog(llm.OfflineBackend(), examples)
        if args.search:
            entries = catalog.filter_catalog(entries, args.search)
        return _print_rules(
            entries, as_json=args.format == "json", searched=bool(args.search)
        )

    if args.command == "schema":
        max_distinct = 0 if args.no_values else schema.DEFAULT_MAX_DISTINCT
        print(schema.schema_context(args.db, max_distinct=max_distinct))
        if args.counts:
            counts = schema.table_row_counts(args.db)
            width = max((len(name) for name, _ in counts), default=0)
            print("\nRow counts:")
            for name, count in counts:
                print(f"  {name:<{width}}  {count:,}")
        return 0

    if args.command == "explain":
        try:
            exp = generator.explain_question(
                args.db,
                args.question,
                use_llm=args.llm,
                use_cache=not args.no_cache,
            )
        except llm.NoRuleMatchError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            _print_no_rule_help(args.question, _DEFAULT_GOLD)
            return 1
        except (ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return _print_explanation(exp)

    try:
        ans = generator.answer_question(
            args.db,
            args.question,
            use_llm=args.llm,
            use_cache=not args.no_cache,
            max_rows=args.max_rows,
            # argparse carries "no deadline" as 0, because a flag is easier to
            # write than a second one; the library spells it None and rejects
            # 0, so the intent has to be stated here rather than fall out of a
            # falsy value someone passed by accident.
            timeout_ms=args.timeout_ms or None,
        )
    except llm.NoRuleMatchError as exc:
        # Handled ahead of the general ValueError clause below, which it
        # subclasses: an unrecognized question is the one failure the catalog
        # can help with, and it is by far the most common way `ask` fails.
        print(f"Error: {exc}", file=sys.stderr)
        _print_no_rule_help(args.question, _DEFAULT_GOLD)
        return 1
    except QueryTimeoutError as exc:
        # Also ahead of the general clause, which catches its RuntimeError
        # base. A timeout is the one failure here the user can resolve by
        # changing a flag rather than the question, so it is worth the hint.
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "Raise --timeout-ms (or pass --timeout-ms 0 to remove the "
            "deadline) if the query is legitimately this expensive.",
            file=sys.stderr,
        )
        return 1
    except (ValueError, UnsafeQueryError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    columns, rows = ans.result.columns, ans.result.rows

    # Disclosed for the same reason a repair is: the answer is sound either
    # way, but "no model was called for this" changes how you read it — most
    # sharply when you have just changed a prompt and are checking the effect.
    if ans.cached:
        print(
            "Note: SQL replayed from the local cache (no model call); "
            "pass --no-cache to regenerate.",
            file=sys.stderr,
        )

    # A repaired answer is still an answer, but it is a weaker one: the backend
    # got it wrong once. Reporting each failed attempt on stderr keeps the
    # result usable in a pipeline while making the retry visible, so nobody
    # reads a repaired run as a clean one.
    for number, attempt in enumerate(ans.repairs, start=1):
        print(
            f"Note: generated SQL failed (attempt {number}) and was repaired "
            f"— {attempt.error}",
            file=sys.stderr,
        )

    # Warn on stderr in every format. It is a diagnostic, not data, so it must
    # not land in a redirected csv/json file — and in table mode the preview's
    # "... (N more rows)" line would otherwise read as the full remainder when
    # the row cap has already discarded rows behind it.
    if ans.result.truncated:
        print(
            f"Warning: result truncated to {args.max_rows} rows by the "
            "--max-rows cap; re-run with a larger --max-rows for the "
            "complete result.",
            file=sys.stderr,
        )

    if args.format == "table":
        print(f"\nQuestion: {ans.question}\n")
        print("SQL:")
        print("  " + ans.sql)
        print(f"\nResults ({len(ans.result)} rows):")
        print(output.format_table(columns, rows, limit=_TABLE_PREVIEW_ROWS))
        print()
        return 0

    # Machine-readable output: keep stdout a clean data stream for redirection
    # (e.g. `nl2sql ask "..." --format csv > sales.csv`) and send the SQL to
    # stderr so it is still visible without corrupting the file.
    print(f"SQL: {ans.sql}", file=sys.stderr)
    if args.format == "csv":
        print(output.format_csv(columns, rows))
    else:
        print(output.format_json(columns, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
