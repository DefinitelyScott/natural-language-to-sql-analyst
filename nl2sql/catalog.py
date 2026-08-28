"""Enumerate the offline backend's question catalog, and search it.

Two jobs, both discoverability: listing the catalog (what ``nl2sql rules``
prints) and finding the entries nearest to a question the catalog could not
answer (the "Did you mean" line ``ask`` and ``explain`` print on a miss). They
live together because both answer "what can I ask?" from the same pairing of
rules to example questions, and the suggestion is only trustworthy because the
pairing is recomputed from the live matcher.


``OfflineBackend`` answers a question by scanning an ordered list of regex rules
and taking the first match, but nothing outside its source file tells a user
which questions those rules actually cover. ``ask`` only reports success or
failure on one question at a time, and ``explain`` needs you to already know the
question you want. This module is the part of ``nl2sql rules`` that closes that
gap and has no I/O or formatting of its own: it pairs each rule with an example
question that routes to it.

The examples are read out of the evaluation gold set rather than kept in a
second hand-written list. Two reasons:

* ``tests/test_rule_catalog.py`` already pins the gold set to exactly one
  question per rule, so it is the repo's canonical example for each pattern.
* The pairing is recomputed from the live matcher on every call, so an example
  cannot go stale. If a new, broader rule starts shadowing an older one, the
  older rule loses its example here immediately instead of continuing to
  advertise a question that no longer reaches it.

A rule with no example is listed with ``example=None`` rather than dropped. An
entry that no question reaches is precisely the shadowed-or-untested rule the
catalog tests exist to catch, so hiding it from the listing would hide the
defect it is evidence of.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from .llm import OfflineBackend


@dataclass(frozen=True)
class CatalogEntry:
    """One offline rule, as presented to a user.

    ``index`` is the rule's position in the catalog, which is also its matching
    priority — the same number ``explain`` reports as the matched rule — so the
    listing and the dry-run diagnostic can be read against each other.
    """

    index: int
    pattern: str
    example: str | None


def load_example_questions(path: str | os.PathLike[str]) -> list[str]:
    """Return the ``question`` field of every record in a JSONL gold file.

    Blank lines are skipped. A line that is not valid JSON, or a record whose
    ``question`` is missing or not a string, raises :class:`ValueError` naming
    the offending line — a silently skipped record would show up only as a rule
    that mysteriously lost its example, which is the same symptom as a genuine
    shadowing bug and would send a reader looking in the wrong place.
    """
    questions: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({exc})") from exc
            question = record.get("question") if isinstance(record, dict) else None
            if not isinstance(question, str):
                raise ValueError(
                    f"{path}:{lineno}: record has no string 'question' field"
                )
            questions.append(question)
    return questions


def build_catalog(
    backend: OfflineBackend, examples: Sequence[str] = ()
) -> list[CatalogEntry]:
    """Pair every rule in ``backend`` with an example question that routes to it.

    Returned in catalog order, one entry per rule, whether or not an example was
    found. A question is credited to the rule that would actually answer it —
    the first of its matches — not to every rule it happens to match, so the
    example shown for a rule is one you can type and get that rule's SQL back.

    When two examples resolve to the same rule the earlier one wins and the
    later is dropped, since a rule has room for a single example. That case is a
    defect in its own right (it means one of the two questions is being answered
    by a pattern meant for the other) and ``test_rule_catalog.py`` fails on it;
    this function only has to stay deterministic when it happens.
    """
    by_rule: dict[int, str] = {}
    for question in examples:
        matches = backend.matching_rule_indexes(question)
        if matches:
            by_rule.setdefault(matches[0], question)

    return [
        CatalogEntry(
            index=index,
            pattern=backend.rule_pattern(index),
            example=by_rule.get(index),
        )
        for index in range(backend.rule_count())
    ]


def answerable_questions(entries: Iterable[CatalogEntry]) -> list[str]:
    """Return the example question of every catalog entry that has one.

    Every string returned is one the offline backend provably answers:
    :func:`build_catalog` credits an example to a rule only after the live
    matcher routed it there. Suggesting from this list rather than from the raw
    gold file is what keeps a suggestion from pointing at a question that has
    since been shadowed and would now fail the same way the user's did.
    """
    return [entry.example for entry in entries if entry.example is not None]


#: Words dropped before comparing a question against the catalog. Deliberately
#: only closed-class filler — question words, articles, auxiliaries and the
#: imperatives the catalog phrases examples with. Nothing that carries analytical
#: meaning is listed, so "revenue", "month" and "customers" always count.
_STOPWORDS = frozenset(
    """
    a an and are as at be by can do does for from get give had has have how i in
    into is it list many me much of on or our please show that the there to us
    was we were what whats when which who whom whose will with
    """.split()
)

_WORD = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> frozenset[str]:
    """Lowercase ``text`` and return its meaning-carrying word tokens.

    Punctuation is discarded rather than split on, so "month-over-month" and
    "month over month" tokenize identically — the catalog's examples and a
    user's phrasing differ that way often enough to matter.

    A one-letter alphabetic token is dropped as well. Those are not words: they
    are the fragments splitting on punctuation leaves behind ("haven't" ->
    "haven", "t"), and matching two questions on a shared "t" would be noise.
    A one-character *digit* is kept, because "top 5 products" means something by
    the 5.
    """
    tokens = {
        token
        for token in _WORD.findall(text.lower())
        if len(token) > 1 or token.isdigit()
    }
    return frozenset(tokens) - _STOPWORDS


def _similarity(question: str, candidate: str) -> tuple[float, float]:
    """Score ``candidate`` against ``question``: (token overlap, character ratio).

    The primary score is Jaccard overlap of content words, which is what makes
    the ranking readable — a suggestion is offered because it talks about the
    same *things*, and anyone can verify that by eye. Word order is ignored
    because "revenue by region" and "region revenue" are the same request.

    The character-level ratio is a tiebreaker only. Token overlap is coarse and
    ties are common once the catalog has forty-odd entries phrased from the same
    small vocabulary; without a second key the winner would come down to
    alphabetical order, which carries no information. It is not used as the
    primary score because it rewards incidental shared characters — a long
    candidate can beat a short exact-topic match on raw string similarity.
    """
    question_tokens = _content_tokens(question)
    candidate_tokens = _content_tokens(candidate)
    union = question_tokens | candidate_tokens
    overlap = len(question_tokens & candidate_tokens) / len(union) if union else 0.0
    ratio = SequenceMatcher(None, question.lower(), candidate.lower()).ratio()
    return overlap, ratio


def suggest_questions(
    question: str, candidates: Iterable[str], *, limit: int = 3
) -> list[str]:
    """Return up to ``limit`` catalog questions closest to ``question``.

    Candidates sharing no content word with ``question`` are dropped rather than
    ranked last. A suggestion list padded out to ``limit`` with unrelated
    questions is worse than a short one: it reads as "the tool guessed", and the
    user has to check each entry to discover none of them apply. When nothing
    overlaps, the honest output is nothing — the caller can fall back to
    pointing at the full catalog.

    Duplicate candidates are collapsed, keeping first occurrence. Ordering is
    fully deterministic: score, then tiebreak ratio, then the question text.
    """
    seen: set[str] = set()
    scored: list[tuple[float, float, str]] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        overlap, ratio = _similarity(question, candidate)
        if overlap > 0.0:
            scored.append((overlap, ratio, candidate))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [candidate for _, _, candidate in scored[:limit]]


def filter_catalog(entries: Iterable[CatalogEntry], needle: str) -> list[CatalogEntry]:
    """Return the entries whose example question or pattern contains ``needle``.

    Case-insensitive substring matching, deliberately: the point is to let
    someone type ``revenue`` and see what the catalog can answer about revenue,
    not to make them write a second regex over the first one.
    """
    lowered = needle.lower()
    return [
        entry
        for entry in entries
        if lowered in entry.pattern.lower()
        or (entry.example is not None and lowered in entry.example.lower())
    ]
