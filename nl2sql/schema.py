"""Introspect a SQLite database and render its schema as prompt context.

The rendered schema is what we hand to an LLM so it can write correct SQL. It is
deliberately compact: table name, columns with types, primary/foreign keys, and
— for columns that behave like categories — the distinct values they hold.

Why the values matter. Structure alone tells a model that ``customers.region``
exists and holds text; it does not tell it that the four strings in there are
``North``, ``South``, ``East`` and ``West``. Asked to filter on a region or a
product category, a model with only the structure has to guess the literal, and
guesses wrong in the ordinary ways: wrong case (``'north'``), a plural
(``'Electronics items'``), or a category that simply is not in the data. The
query runs, returns zero rows, and looks like a correct answer to an empty
question — the failure mode the read-only validator and the repair loop both
miss, because nothing errored. Listing the values closes that gap for the
columns where it is cheap to do so.

Sampling is bounded on purpose (see :func:`column_values`): a column with more
distinct values than the cap is free-form text, not a category, and dumping it
would bloat the prompt and copy row data — customer names, order dates — into
it for nothing.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: Most distinct values a column may hold and still be rendered as a category.
#: Twelve is chosen to sit just above the widest genuine dimension in the sample
#: schema (12 product names) and far below anything free-form, so the cap is
#: doing the classifying rather than a hand-maintained column allowlist.
DEFAULT_MAX_DISTINCT = 12

#: A column holding any value longer than this is treated as free text even if
#: it has few distinct values. The alternative — truncating a long value for
#: display — is worse than omitting it: a model that copies a truncated literal
#: into a ``WHERE`` clause silently matches nothing.
_MAX_VALUE_LENGTH = 40

#: Declared-type substrings that give a column TEXT affinity, per SQLite's
#: documented affinity rules. Only text columns are sampled: a small set of
#: distinct integers or reals is usually a measure that happens to be sparse
#: (a quantity, a price), not a category worth spelling out for the model.
_TEXT_AFFINITY_MARKERS = ("CHAR", "CLOB", "TEXT")


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


def _quote(identifier: str) -> str:
    """Double-quote a SQLite identifier, escaping any embedded double quote.

    Every identifier passed here comes from ``sqlite_master`` or a ``PRAGMA``,
    never from user input, so this is about correctness on unusual-but-legal
    names rather than injection defence.
    """
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _has_text_affinity(declared_type: str) -> bool:
    """True when SQLite would give ``declared_type`` TEXT affinity."""
    upper = declared_type.upper()
    return any(marker in upper for marker in _TEXT_AFFINITY_MARKERS)


def column_values(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    *,
    max_distinct: int = DEFAULT_MAX_DISTINCT,
) -> list[str] | None:
    """Return ``column``'s distinct values, or ``None`` if it is not categorical.

    ``None`` means "do not describe this column by its contents": either it has
    more than ``max_distinct`` distinct non-null values, or one of them is
    longer than :data:`_MAX_VALUE_LENGTH`. Both are the same judgement — the
    column is free-form text — reached by two different signals, and both are
    reported the same way so the caller has one case to handle.

    An empty list is a different answer: the column *is* categorical and simply
    has no non-null values yet (an empty table). The caller can render that as
    nothing without conflating it with "too many to list".

    The query fetches at most ``max_distinct + 1`` rows, so a column with a
    million distinct values costs one extra row rather than a full scan of them
    all. That one extra row is the whole test: getting it back means the column
    is over the cap. Nulls are excluded because ``NULL`` is not a value a
    ``WHERE`` clause can be written against with ``=``.

    No ``ORDER BY`` is issued, so *which* rows come back when the column is over
    the cap is up to SQLite — which does not matter, because in that case the
    values are discarded and only the count is used. When the column is under
    the cap every distinct value comes back regardless of order, and the result
    is sorted here, so the returned list is deterministic.
    """
    if max_distinct < 1:
        return None

    rows = conn.execute(
        f"SELECT DISTINCT {_quote(column)} FROM {_quote(table)} "
        f"WHERE {_quote(column)} IS NOT NULL LIMIT ?",
        (max_distinct + 1,),
    ).fetchall()

    if len(rows) > max_distinct:
        return None

    values = [str(row[0]) for row in rows]
    if any(len(value) > _MAX_VALUE_LENGTH for value in values):
        return None
    return sorted(values)


def sample_categorical_values(
    conn: sqlite3.Connection,
    tables: Sequence[Table],
    *,
    max_distinct: int = DEFAULT_MAX_DISTINCT,
) -> dict[tuple[str, str], list[str]]:
    """Map ``(table, column)`` to its distinct values, for categorical columns.

    Only text-affinity columns that are neither a primary key nor a foreign key
    are considered. Keys are excluded by kind rather than by cardinality: they
    identify rows, and a small lookup table whose ids happen to fit under the
    cap would still be listing identifiers no one filters on by literal.

    Columns that are not categorical (:func:`column_values` returned ``None``)
    and columns with no values are simply absent from the mapping, so a caller
    can treat "in the mapping" as "there is something worth printing".
    """
    sampled: dict[tuple[str, str], list[str]] = {}
    for table in tables:
        fk_columns = {column for column, _, _ in table.foreign_keys}
        for column in table.columns:
            if column.pk or column.name in fk_columns:
                continue
            if not _has_text_affinity(column.type):
                continue
            values = column_values(
                conn, table.name, column.name, max_distinct=max_distinct
            )
            if values:
                sampled[(table.name, column.name)] = values
    return sampled


def _format_values(values: Sequence[str]) -> str:
    """Render values as a SQL-comment hint: ``one of: 'East', 'North'``.

    Values are shown as SQL string literals — single-quoted, with embedded
    quotes doubled — so they can be pasted into a ``WHERE`` clause as they
    appear rather than re-quoted by whoever reads them.
    """
    literals = ", ".join("'" + value.replace("'", "''") + "'" for value in values)
    return f"one of: {literals}"


def render(
    tables: Sequence[Table],
    values: Mapping[tuple[str, str], list[str]] | None = None,
) -> str:
    """Render tables as a readable schema block for an LLM prompt.

    ``values`` is the mapping from :func:`sample_categorical_values`; when it is
    omitted the output is structure only, exactly as before. Value hints are
    attached as trailing SQL line comments so the block stays readable as DDL
    and a model has no reason to mistake a hint for a column.
    """
    values = values or {}
    lines: list[str] = []
    for t in tables:
        lines.append(f"TABLE {t.name} (")
        last = len(t.columns) - 1
        for i, c in enumerate(t.columns):
            tag = " PRIMARY KEY" if c.pk else ""
            # The comma belongs to the declaration, not the line, so it has to
            # be placed before the comment rather than after it.
            declaration = f"  {c.name} {c.type}{tag}".rstrip()
            if i < last:
                declaration += ","
            hint = values.get((t.name, c.name))
            lines.append(
                f"{declaration}  -- {_format_values(hint)}" if hint else declaration
            )
        for col, ref_table, ref_col in t.foreign_keys:
            lines.append(f"  FOREIGN KEY ({col}) REFERENCES {ref_table}({ref_col})")
        lines.append(")")
        lines.append("")
    return "\n".join(lines).strip()


def schema_context(
    db_path: str, *, max_distinct: int = DEFAULT_MAX_DISTINCT
) -> str:
    """Open a DB read-only and return its rendered schema.

    Pass ``max_distinct=0`` for structure only. The sampling costs one small
    ``LIMIT``-ed query per text column, which is why it is on by default: the
    schema is rendered once per question, and the value hints are worth more to
    a model than the handful of queries costs.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = introspect(conn)
        values = sample_categorical_values(conn, tables, max_distinct=max_distinct)
        return render(tables, values)
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
            cur = conn.execute(f"SELECT COUNT(*) FROM {_quote(table.name)}")
            counts.append((table.name, cur.fetchone()[0]))
        return counts
    finally:
        conn.close()
