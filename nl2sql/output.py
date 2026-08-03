"""Render query results in human- and machine-readable formats.

Three formatters share a ``(columns, rows)`` signature so the CLI can select one
by name:

* ``format_table`` — an aligned text table for humans. Supports an optional row
  ``limit`` so the terminal preview stays readable on large result sets.
* ``format_csv`` — RFC-4180 CSV via the stdlib ``csv`` writer (handles quoting
  of values that contain commas, quotes, or newlines).
* ``format_json`` — a JSON array of row objects keyed by column name.

``format_csv`` and ``format_json`` apply no limit of their own: an export should
contain every row it is handed. They cannot promise a *complete* export, though
— ``runner.run`` caps how many rows reach them (``--max-rows``), and the CLI
warns when that cap actually bit.

Keeping these as pure functions (no I/O, no globals) makes them trivial to unit
test and to reuse outside the CLI.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Sequence

Row = Sequence[Any]


def format_table(
    columns: Sequence[str],
    rows: Sequence[Row],
    *,
    limit: int | None = None,
) -> str:
    """Render results as a column-aligned text table.

    When ``limit`` is given and there are more rows than that, only the first
    ``limit`` rows are shown followed by a "... (N more rows)" line. ``None``
    values render as empty cells.
    """
    if not columns:
        return "(no columns)"

    shown = list(rows) if limit is None else list(rows)[:limit]
    str_rows = [["" if v is None else str(v) for v in row] for row in shown]

    widths = [len(c) for c in columns]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    lines = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(columns))]
    lines.extend(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        for row in str_rows
    )
    if limit is not None and len(rows) > limit:
        lines.append(f"... ({len(rows) - limit} more rows)")
    return "\n".join(lines)


def format_csv(columns: Sequence[str], rows: Sequence[Row]) -> str:
    """Render results as CSV with a header row. Writes every row given."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    writer.writerows(rows)
    return buffer.getvalue().rstrip("\r\n")


def format_json(columns: Sequence[str], rows: Sequence[Row]) -> str:
    """Render results as a JSON array of row objects. Writes every row given.

    ``default=str`` keeps the call from failing on any value type SQLite can
    return that is not natively JSON-serializable (e.g. ``bytes``).
    """
    records = [dict(zip(columns, row)) for row in rows]
    return json.dumps(records, indent=2, default=str)
