"""Command-line interface.

Commands:
    ask "<question>"   answer a natural-language question with SQL + results
    schema [--counts]  print the introspected database schema
"""

from __future__ import annotations

import argparse
import os
import sys

from . import generator, output, schema
from .runner import UnsafeQueryError

_DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")

# How many rows to show in the human-readable table preview before truncating.
# csv/json output is never truncated — an export should be complete.
_TABLE_PREVIEW_ROWS = 20


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nl2sql", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer a natural-language question")
    ask.add_argument("question", help="the question, in plain English")
    ask.add_argument("--db", default=_DEFAULT_DB, help="path to the SQLite database")
    ask.add_argument("--llm", action="store_true", help="use the LLM backend")
    ask.add_argument("--max-rows", type=int, default=1000)
    ask.add_argument(
        "--format",
        choices=("table", "csv", "json"),
        default="table",
        help=(
            "output format (default: table). With csv/json, only the data is "
            "written to stdout so it can be redirected to a file; the generated "
            "SQL is written to stderr."
        ),
    )

    show = sub.add_parser(
        "schema",
        help="print the database schema (tables, columns, keys)",
        description=(
            "Print the introspected schema exactly as it is rendered for the "
            "LLM prompt — useful for seeing what context the model works from, "
            "and for orienting yourself in an unfamiliar database."
        ),
    )
    show.add_argument("--db", default=_DEFAULT_DB, help="path to the SQLite database")
    show.add_argument(
        "--counts",
        action="store_true",
        help="also show the number of rows in each table",
    )

    args = parser.parse_args(argv)

    if not os.path.exists(args.db):
        print(
            f"Database not found at {args.db}. "
            "Run: python scripts/build_sample_db.py",
            file=sys.stderr,
        )
        return 2

    if args.command == "schema":
        print(schema.schema_context(args.db))
        if args.counts:
            counts = schema.table_row_counts(args.db)
            width = max((len(name) for name, _ in counts), default=0)
            print("\nRow counts:")
            for name, count in counts:
                print(f"  {name:<{width}}  {count:,}")
        return 0

    try:
        ans = generator.answer_question(
            args.db, args.question, use_llm=args.llm, max_rows=args.max_rows
        )
    except (ValueError, UnsafeQueryError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    columns, rows = ans.result.columns, ans.result.rows

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
