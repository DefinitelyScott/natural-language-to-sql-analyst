"""Orchestrate: question -> schema context -> SQL -> executed result."""

from __future__ import annotations

from dataclasses import dataclass

from . import llm, runner, schema


@dataclass
class Answer:
    question: str
    sql: str
    result: runner.QueryResult


def answer_question(
    db_path: str,
    question: str,
    *,
    use_llm: bool = False,
    max_rows: int = 1000,
) -> Answer:
    """Generate SQL for ``question`` and execute it against ``db_path``."""
    schema_text = schema.schema_context(db_path)
    backend = llm.get_backend(use_llm)
    sql = backend.to_sql(question, schema_text)
    result = runner.run(db_path, sql, max_rows=max_rows)
    return Answer(question=question, sql=sql, result=result)
