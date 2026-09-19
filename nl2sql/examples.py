"""Example questions: loading them, and ranking them against a question.

Two parts of this project need the same thing — a list of example questions and
a way to find the ones nearest to a question someone just asked:

* :mod:`nl2sql.catalog` ranks the offline catalog's example questions to build
  the "Did you mean" line ``ask`` prints when no rule matches.
* :class:`nl2sql.llm.LLMBackend` ranks a pool of solved ``(question, sql)``
  pairs to pick the few-shot examples it shows the model.

The scoring lives here so those two cannot drift apart: a phrasing the suggester
treats as close is the same phrasing the prompt builder treats as close, and
there is one implementation to reason about rather than two that agree by
coincidence.

Few-shot selection, specifically
--------------------------------

The LLM backend is zero-shot by default — the model sees the rendered schema and
the question and nothing else. A handful of solved questions from the same
database is the cheapest way to communicate the conventions the schema text
cannot state: that revenue is ``quantity * unit_price`` off ``order_items``,
that a month is ``strftime('%Y-%m', o.order_date)``, that totals are rounded to
two places. Examples are chosen *per question* rather than fixed, so a question
about categories is shown category queries instead of whatever happens to sit at
the top of the file.

Two rules keep the feature honest:

* **Never show the answer to the question being asked.**
  :func:`select_examples` drops any pool entry whose question is the asked one,
  so pointing the pool at a file that happens to contain the question cannot
  quietly turn a generation into a lookup.
* **A pool drawn from an evaluation gold set contaminates that evaluation.**
  The exclusion above removes the exact row; it cannot remove the near-neighbour
  rows, and those are precisely what makes few-shot work and precisely what
  makes the resulting score meaningless. Nothing here can detect that — it is a
  choice the caller makes — so ``evals/evaluate.py`` passes no pool at all, and
  the ``--examples`` help text says so where someone is most likely to read it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

#: How many examples go into a prompt when a caller does not say otherwise.
#: Three is enough to establish a house style across two or three different
#: shapes of query, and small enough that the examples never crowd out the
#: schema they are meant to be read alongside.
DEFAULT_LIMIT = 3

#: The instruction that introduces the examples in the prompt. A module-level
#: constant rather than an inline string because it is *prompt text*, of the
#: same standing as ``nl2sql.llm._SYSTEM_PROMPT``: editing it changes what every
#: few-shot prompt says, so :func:`pool_fingerprint` digests it and the SQL
#: cache invalidates on an edit. Inline, it would have been an edit no cache key
#: could see.
_EXAMPLES_HEADER = (
    "Examples of questions already answered against this schema. Follow their "
    "join and aggregation conventions; do not assume one of them answers the "
    "question below:"
)

#: Words dropped before comparing two questions. Deliberately only closed-class
#: filler — question words, articles, auxiliaries and the imperatives the
#: catalog phrases examples with. Nothing that carries analytical meaning is
#: listed, so "revenue", "month" and "customers" always count.
_STOPWORDS = frozenset(
    """
    a an and are as at be by can do does for from get give had has have how i in
    into is it list many me much of on or our please show that the there to us
    was we were what whats when which who whom whose will with
    """.split()
)

_WORD = re.compile(r"[a-z0-9]+")
_WHITESPACE = re.compile(r"\s+")


def _content_tokens(text: str) -> frozenset[str]:
    """Lowercase ``text`` and return its meaning-carrying word tokens.

    Punctuation is discarded rather than split on, so "month-over-month" and
    "month over month" tokenize identically — a user's phrasing and a stored
    example differ that way often enough to matter.

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


def similarity(question: str, candidate: str) -> tuple[float, float]:
    """Score ``candidate`` against ``question``: (token overlap, character ratio).

    The primary score is Jaccard overlap of content words, which is what makes
    the ranking readable — a candidate ranks highly because it talks about the
    same *things*, and anyone can verify that by eye. Word order is ignored
    because "revenue by region" and "region revenue" are the same request.

    The character-level ratio is a tiebreaker only. Token overlap is coarse and
    ties are common once a pool has forty-odd entries phrased from the same
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


@dataclass(frozen=True)
class Example:
    """A solved question: the English, and the SQL that answers it."""

    question: str
    sql: str


def load_examples(path: str | os.PathLike[str]) -> list[Example]:
    """Return every ``{"question": ..., "sql": ...}`` record in a JSONL file.

    Blank lines are skipped. Any other malformed line raises :class:`ValueError`
    naming it, for the same reason :func:`nl2sql.catalog.load_example_questions`
    does: a silently dropped record would surface only as a prompt quietly
    missing an example, which looks exactly like the selector deciding nothing
    was relevant. One of those is a broken file and the other is normal
    operation, so they must not produce the same symptom.
    """
    loaded: list[Example] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({exc})") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{lineno}: record is not a JSON object")
            question, sql = record.get("question"), record.get("sql")
            if not isinstance(question, str) or not isinstance(sql, str):
                raise ValueError(
                    f"{path}:{lineno}: record needs string 'question' and 'sql' fields"
                )
            loaded.append(Example(question=question, sql=sql))
    return loaded


def _normalize_question(question: str) -> str:
    """Return ``question`` in the form used to test two questions for sameness.

    Case, surrounding whitespace, internal runs of whitespace and trailing
    ``?``/``.`` punctuation are all discarded. Used only for the self-exclusion
    and duplicate checks in :func:`select_examples`, where the goal is to catch a
    pool entry that *is* the asked question however it happened to be typed.

    Deliberately not used for scoring. The model receives the exact string the
    caller passed, so ranking a normalized form would rank something the prompt
    never contains.
    """
    return _WHITESPACE.sub(" ", question.strip().lower()).rstrip("?.").strip()


def select_examples(
    question: str, pool: Iterable[Example], *, limit: int = DEFAULT_LIMIT
) -> list[Example]:
    """Return up to ``limit`` pool entries most similar to ``question``.

    Three filters apply, in order:

    * An entry whose question is the asked one is dropped — see the module
      docstring for why that matters more than it looks like it should.
    * A repeated question keeps its first occurrence, so a pool assembled from
      two overlapping files cannot spend the whole budget on one example.
    * An entry sharing no content word with the question is dropped rather than
      ranked last. An unrelated example is not a weak hint, it is a misleading
      one: the model is being shown a query and implicitly told to write
      something like it.

    Ordering is fully deterministic — overlap, then the tiebreak ratio, then the
    question text. That is what lets :func:`pool_fingerprint` cover the selection
    by digesting only the pool it was drawn from.
    """
    asked = _normalize_question(question)
    seen: set[str] = set()
    scored: list[tuple[float, float, str, Example]] = []

    for example in pool:
        key = _normalize_question(example.question)
        if key == asked or key in seen:
            continue
        seen.add(key)
        overlap, ratio = similarity(question, example.question)
        if overlap > 0.0:
            scored.append((overlap, ratio, example.question, example))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [example for _, _, _, example in scored[:limit]]


def render_examples(examples: Sequence[Example]) -> str:
    """Return the prompt block for ``examples``, or ``""`` when there are none.

    One ``Q:``/``SQL:`` pair per block, blank line between blocks — the plainest
    shape that survives being read by a person debugging a prompt and by a model
    reading it as a pattern.

    An empty pool renders to the empty string — no header over nothing — which
    is what lets :func:`nl2sql.llm.build_user_message` reproduce the zero-shot
    prompt byte for byte and keep SQL cached before this module existed
    reachable.
    """
    if not examples:
        return ""
    blocks = "\n\n".join(
        f"Q: {example.question}\nSQL: {example.sql}" for example in examples
    )
    return f"{_EXAMPLES_HEADER}\n\n{blocks}"


def pool_fingerprint(pool: Sequence[Example]) -> str:
    """Return a short, stable digest of a few-shot pool and its prompt header.

    :func:`nl2sql.llm.build_cache_identity` folds this into the SQL cache key so
    a cached answer cannot be replayed under different examples, or under a
    reworded instruction about how to read them, than the ones that produced it.

    What is covered, and what is not
    --------------------------------

    Digested: the pool's contents, and :data:`_EXAMPLES_HEADER`. Those are the
    two *inputs* to the prompt that can change without any code changing — a
    caller swaps files, or someone edits the instruction — and a key blind to
    either would replay an answer written under a different prompt.

    Not digested: the selection *code* — :data:`DEFAULT_LIMIT`,
    :func:`similarity`, :data:`_STOPWORDS`. Retuning the scorer can change which
    examples a question selects from an unchanged pool, and no runtime digest
    can see that; it is a code change, and the remedy is the one any code change
    gets, namely deleting ``data/sql_cache.json`` or passing ``--no-cache``. The
    same limit already applies to every other behaviour in this project that a
    cache key cannot observe, so stating the boundary is more useful than
    pretending it is closed. Note that ``pool`` and the header *can* change on a
    machine that is not being edited at all, which is why those two are here.

    The pool is sorted before hashing, because selection is order-independent: a
    pool whose lines were reordered on disk picks the same examples, and hashing
    it differently would discard cache entries that are still correct. Entries
    are serialized as JSON so a newline or a NUL inside an example's SQL cannot
    shift the boundary between two records and alias two different pools onto
    one digest.

    Truncated to 12 hex characters for the same reason the prompt fingerprint is
    — it has to distinguish a handful of pools, not resist an adversary, and a
    short digest keeps a cache file readable when someone opens it by hand.
    """
    payload = json.dumps(
        [_EXAMPLES_HEADER, sorted([example.question, example.sql] for example in pool)],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
