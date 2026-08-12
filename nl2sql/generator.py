"""Orchestrate: question -> schema context -> SQL -> executed result.

Two entry points, deliberately separated:

* :func:`answer_question` generates SQL *and executes it* — what ``nl2sql ask``
  does.
* :func:`explain_question` generates SQL and inspects it *without touching the
  data* — what ``nl2sql explain`` does. It reports which offline rule the
  question routed to, which other rules also matched but were shadowed by it,
  and whether the SQL would survive ``runner.validate``.

Splitting them keeps the dry run honest: an explanation cannot accidentally run
a query, so it is safe to point at SQL you do not yet trust (an LLM's, say).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import llm, runner, schema


@dataclass
class Answer:
    question: str
    sql: str
    result: runner.QueryResult


@dataclass
class Explanation:
    """A dry-run report: the SQL a question resolves to, and how it got there.

    ``matched_rule`` / ``shadowed_rules`` are populated only for the offline
    backend, which routes by an ordered regex catalog and can therefore say
    *why* a question produced this SQL. The LLM backend has no such structure,
    so both stay empty and only ``sql`` and the safety verdict are meaningful.

    ``safety_error`` is the message ``runner.validate`` rejected the SQL with,
    or ``None`` when it passed. It is captured rather than raised because
    reporting an unsafe query is the whole point of the command — the caller
    wants to see the offending SQL alongside the reason it was rejected.
    """

    question: str
    backend: str
    sql: str
    matched_rule: int | None = None
    matched_pattern: str | None = None
    shadowed_rules: list[tuple[int, str]] = field(default_factory=list)
    safety_error: str | None = None

    @property
    def is_safe(self) -> bool:
        """True when the generated SQL passed ``runner.validate``."""
        return self.safety_error is None


def _resolve(db_path: str, question: str, *, use_llm: bool) -> tuple[llm.Backend, str]:
    """Return the backend that answered ``question`` and the SQL it produced.

    The backend is handed back alongside the SQL so a caller that wants to
    introspect the routing (:func:`explain_question`) does not have to construct
    a second one — which for the LLM backend would mean a second API client.
    """
    schema_text = schema.schema_context(db_path)
    backend = llm.get_backend(use_llm)
    return backend, backend.to_sql(question, schema_text)


def generate_sql(db_path: str, question: str, *, use_llm: bool = False) -> str:
    """Generate SQL for ``question`` without executing it."""
    _, sql = _resolve(db_path, question, use_llm=use_llm)
    return sql


def answer_question(
    db_path: str,
    question: str,
    *,
    use_llm: bool = False,
    max_rows: int = 1000,
) -> Answer:
    """Generate SQL for ``question`` and execute it against ``db_path``."""
    _, sql = _resolve(db_path, question, use_llm=use_llm)
    result = runner.run(db_path, sql, max_rows=max_rows)
    return Answer(question=question, sql=sql, result=result)


def explain_question(
    db_path: str, question: str, *, use_llm: bool = False
) -> Explanation:
    """Generate SQL for ``question`` and describe it, without executing it."""
    backend, sql = _resolve(db_path, question, use_llm=use_llm)

    explanation = Explanation(
        question=question,
        backend="llm" if use_llm else "offline",
        sql=sql,
    )

    if isinstance(backend, llm.OfflineBackend):
        matches = backend.matching_rule_indexes(question)
        # ``to_sql`` raises when nothing matches, so reaching this line
        # guarantees at least one index: the first is the winner, the rest are
        # shadowed by it.
        explanation.matched_rule = matches[0]
        explanation.matched_pattern = backend.rule_pattern(matches[0])
        explanation.shadowed_rules = [
            (index, backend.rule_pattern(index)) for index in matches[1:]
        ]

    try:
        runner.validate(sql)
    except runner.UnsafeQueryError as exc:
        explanation.safety_error = str(exc)

    return explanation
