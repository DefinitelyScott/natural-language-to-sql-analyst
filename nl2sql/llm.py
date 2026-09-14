"""Resolve a natural-language question to SQL.

Two backends:

* ``OfflineBackend`` — a deterministic, rule-based matcher over a catalog of
  known analytical question patterns. Requires no network or API key, so the
  test suite and CI use it. It is intentionally small and transparent.
* ``LLMBackend`` — sends the schema + question to an OpenAI-compatible chat
  model and returns the SQL it produces. Used when ``OPENAI_API_KEY`` is set.

Both return a raw SQL string; validation and execution happen in ``runner``.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol, runtime_checkable

from .rules import build_catalog


class Backend(Protocol):
    def to_sql(self, question: str, schema: str) -> str: ...


@runtime_checkable
class RepairingBackend(Backend, Protocol):
    """A backend that can revise SQL it wrote, given the error it raised.

    Optional: a backend is repairable only if implementing ``repair`` can
    actually change the outcome. ``LLMBackend`` qualifies — a model that
    hallucinated a column name will often fix it when shown the error.
    ``OfflineBackend`` deliberately does not: its SQL is hand-written and
    keyed to a fixed rule, so re-asking the same question returns the same
    string and a retry could only burn time. ``generator.answer_question``
    checks for this protocol at runtime (``isinstance``, which for a
    ``runtime_checkable`` Protocol tests only that the methods exist) and
    skips the repair step entirely when a backend does not implement it.

    It extends :class:`Backend` rather than standing alone because repairing is
    a capability *added to* generating, never a substitute for it: every holder
    of one of these calls ``to_sql`` on it too — ``cache.CachingBackend`` stores
    a ``RepairingBackend`` and immediately generates through it. Declaring only
    ``repair`` made that a type error the checker was right to flag, and the fix
    is to state the real contract rather than to widen the annotation. The
    runtime check tightens with it: ``isinstance`` now requires both methods, so
    a class with ``repair`` but no ``to_sql`` no longer passes as repairable.
    """

    def repair(self, question: str, schema: str, sql: str, error: str) -> str: ...


class NoRuleMatchError(ValueError):
    """No rule in the offline catalog matches the question.

    Subclasses :class:`ValueError` so existing callers that catch ``ValueError``
    around ``to_sql`` keep working unchanged. It exists as its own type so the
    CLI can distinguish "the catalog does not cover this question" — the one
    failure a nearest-question suggestion can help with — from every other
    ``ValueError`` a backend might raise, without matching on message text.
    """


# --------------------------------------------------------------------------- #
# Offline rule-based backend
# --------------------------------------------------------------------------- #
class OfflineBackend:
    """Map a question to SQL via lightweight keyword rules.

    This is not meant to be a general NL parser. It recognizes a fixed catalog
    of common analytics questions so the project is runnable and verifiable
    offline. Each rule is a (matcher, sql) pair, and resolution is
    first-rule-wins, so catalog order is matching priority.

    The catalog itself lives in :mod:`nl2sql.rules`; this class is only the
    scan over it and the diagnostics that report on that scan.
    """

    def __init__(self) -> None:
        self._rules: list[tuple[re.Pattern[str], str]] = build_catalog()

    def rule_count(self) -> int:
        """Return the number of question patterns registered in the catalog."""
        return len(self._rules)

    def rule_pattern(self, index: int) -> str:
        """Return the regex source of the rule at ``index``.

        Used to name a rule in a test failure message; a pattern is far more
        recognizable than a bare index when a catalog invariant breaks.
        """
        return self._rules[index][0].pattern

    def matching_rule_indexes(self, question: str) -> list[int]:
        """Return the index of every rule whose matcher matches ``question``.

        Resolution is first-rule-wins, so only ``[0]`` of this list decides the
        SQL. The rest is what makes the catalog's ordering auditable: a rule
        that matches but never wins is shadowed by a broader rule registered
        ahead of it, and a rule that never appears first for any question is
        unreachable. ``tests/test_rule_catalog.py`` asserts both properties
        across the gold set.
        """
        return [
            index
            for index, (matcher, _) in enumerate(self._rules)
            if matcher.search(question)
        ]

    def to_sql(self, question: str, schema: str) -> str:  # noqa: ARG002
        # Deliberately reuses ``matching_rule_indexes`` instead of short-circuiting
        # on the first match: routing and the ordering diagnostic then share one
        # implementation and cannot drift apart. Scanning ~40 small regexes is not
        # a meaningful cost next to executing the query.
        matches = self.matching_rule_indexes(question)
        if not matches:
            raise NoRuleMatchError(
                "Offline backend has no rule for this question. "
                "Set OPENAI_API_KEY and use --llm for open-ended questions."
            )
        _, sql = self._rules[matches[0]]
        return " ".join(sql.split())


# --------------------------------------------------------------------------- #
# LLM backend
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = """You are a careful analytics engineer. Given a SQLite schema
and a question, return a single read-only SQL query that answers it.

Rules:
- Output ONLY the SQL, no prose, no markdown fences.
- Use only SELECT (or WITH ... SELECT). Never modify data.
- Use the exact table and column names from the schema.
- Prefer explicit JOINs and clear column aliases.
"""

_REPAIR_PROMPT = """You are a careful analytics engineer. A SQLite query you
wrote for a question failed. Rewrite it so that it runs and still answers the
same question.

Rules:
- Output ONLY the corrected SQL, no prose, no markdown fences.
- Use only SELECT (or WITH ... SELECT). Never modify data.
- Use only the exact table and column names from the schema.
- Fix the reported error. Do not change what the query is trying to measure.
"""


def _prompt_fingerprint(prompt: str) -> str:
    """Return a short, stable digest of a prompt.

    Used to make the prompt part of a cache key without storing the prompt
    itself in every cache entry. Truncated to 12 hex characters: the digest
    only has to distinguish successive revisions of one file, not resist an
    adversary, and a short one keeps ``cache_identity`` readable when it turns
    up in a cache file someone is inspecting by hand.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


class LLMBackend:
    """OpenAI-compatible chat backend. Imports the client lazily."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        from openai import OpenAI  # lazy import; optional dependency

        self._client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def cache_identity(self) -> str:
        """Everything about this backend's configuration that shapes its SQL.

        The model name and a fingerprint of the system prompt, which together
        with the question and schema determine ``to_sql``'s output at
        ``temperature=0``. Satisfies :class:`nl2sql.cache.CacheableBackend`, and
        exists so that ``cache`` never has to reach into this class to discover
        how it is configured.

        The *repair* prompt is deliberately excluded: repairs are not cached, so
        including it would invalidate every stored entry on an edit that cannot
        change any of them.
        """
        return f"{self._model}/{_prompt_fingerprint(_SYSTEM_PROMPT)}"

    def to_sql(self, question: str, schema: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
            ],
        )
        sql = resp.choices[0].message.content or ""
        return _strip_fences(sql).strip()

    def repair(self, question: str, schema: str, sql: str, error: str) -> str:
        """Return a rewritten query, given the SQL that failed and its error.

        The failed SQL and the error text are the whole point: without them the
        model is just being asked the same question again at temperature 0 and
        would return the same query. With them, the common LLM text-to-SQL
        failure modes — a column that does not exist, a table joined on the
        wrong key, a function SQLite does not have — become directly
        correctable, because the engine has already named what is wrong.

        The original question and schema are re-sent rather than relying on a
        conversation history so this call is stateless: one repair is
        independent of any other, which keeps it cheap to reason about and
        makes the backend safe to reuse across questions.

        This method has no authority of its own. Whatever it returns goes back
        through the same validator and read-only connection as the first
        attempt, so a repair cannot widen what the system is willing to run.
        """
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": _REPAIR_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Schema:\n{schema}\n\n"
                        f"Question: {question}\n\n"
                        f"SQL that failed:\n{sql}\n\n"
                        f"Error:\n{error}"
                    ),
                },
            ],
        )
        return _strip_fences(resp.choices[0].message.content or "").strip()


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:sql)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def get_backend(use_llm: bool) -> Backend:
    """Factory: pick the LLM backend when requested, else offline."""
    if use_llm:
        return LLMBackend()
    return OfflineBackend()
