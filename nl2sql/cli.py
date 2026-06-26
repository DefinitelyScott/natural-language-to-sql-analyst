"""Command-line interface: `python -m nl2sql.cli ask "<question>"`."""

from __future__ import annotations

import argparse
import os
import sys

from . import generator
from .runner import UnsafeQueryError

_DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")


def _format_table(columns: list[str], rows: list[tuple], limit: int = 20) -> str:
    if not columns:
        return "(no columns)"
    widths = [len(c) for c in columns]
    shown = rows[:limit]
    for row in shown:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
    out = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(columns))]
    for row in shown:
        out.append("  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))
    if len(rows) > limit:
        out.append(f"... ({len(rows) - limit} more rows)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nl2sql", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer a natural-language question")
    ask.add_argument("question", help="the question, in plain English")
    ask.add_argument("--db", default=_DEFAULT_DB, help="path to the SQLite database")
    ask.add_argument("--llm", action="store_true", help="use the LLM backend")
    ask.add_argument("--max-rows", type=int, default=1000)

    args = parser.parse_args(argv)

    if not os.path.exists(args.db):
        print(
            f"Database not found at {args.db}. "
            "Run: python scripts/build_sample_db.py",
            file=sys.stderr,
        )
        return 2

    try:
        ans = generator.answer_question(
            args.db, args.question, use_llm=args.llm, max_rows=args.max_rows
        )
    except (ValueError, UnsafeQueryError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nQuestion: {ans.question}\n")
    print("SQL:")
    print("  " + ans.sql)
    print(f"\nResults ({len(ans.result)} rows):")
    print(_format_table(ans.result.columns, ans.result.rows))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
