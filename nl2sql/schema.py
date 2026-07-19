"""Introspect a SQLite database and render its schema as prompt context.

The rendered schema is what we hand to an LLM so it can write correct SQL. It is
deliberately compact: table name, columns with types, and primary/foreign keys.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass
class Column:
    name: str
    type: str
    pk: bool


@dataclass
class Table:
    name: str
    columns: list[Column]
    foreign_keys: list[tuple[str, str, str]]  # (column, ref_table, ref_column)


def introspect(conn: sqlite3.Connection) -> list[Table]:
    """Return the list of user tables in the database with column + FK info."""
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    tables: list[Table] = []
    for (table_name,) in cur.fetchall():
        cols = [
            Column(name=row[1], type=row[2] or "", pk=bool(row[5]))
            for row in cur.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        ]
        fks = [
            (row[3], row[2], row[4])  # from, table, to
            for row in cur.execute(f"PRAGMA foreign_key_list('{table_name}')").fetchall()
        ]
        tables.append(Table(name=table_name, columns=cols, foreign_keys=fks))
    return tables


def render(tables: list[Table]) -> str:
    """Render tables as a readable schema block for an LLM prompt."""
    lines: list[str] = []
    for t in tables:
        col_parts = []
        for c in t.columns:
            tag = " PRIMARY KEY" if c.pk else ""
            col_parts.append(f"  {c.name} {c.type}{tag}".rstrip())
        lines.append(f"TABLE {t.name} (")
        lines.append(",\n".join(col_parts))
        for col, ref_table, ref_col in t.foreign_keys:
            lines.append(f"  FOREIGN KEY ({col}) REFERENCES {ref_table}({ref_col})")
        lines.append(")")
        lines.append("")
    return "\n".join(lines).strip()


def schema_context(db_path: str) -> str:
    """Convenience: open a DB read-only and return its rendered schema."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return render(introspect(conn))
    finally:
        conn.close()


def table_row_counts(db_path: str) -> list[tuple[str, int]]:
    """Return ``(table_name, row_count)`` for every user table.

    Table names come from ``sqlite_master`` (via :func:`introspect`), not from
    user input, so interpolating them into ``COUNT(*)`` queries is safe; they
    are still double-quoted to handle any unusual-but-legal table names.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        counts: list[tuple[str, int]] = []
        for table in introspect(conn):
            cur = conn.execute(f'SELECT COUNT(*) FROM "{table.name}"')
            counts.append((table.name, cur.fetchone()[0]))
        return counts
    finally:
        conn.close()
