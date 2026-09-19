"""Cover the few-shot example pool: loading, selection, rendering, fingerprint.

Three properties matter more than the rest, and each has a test that fails
loudly if it breaks:

* **A zero-shot prompt is unchanged.** ``build_user_message`` with no pool must
  return exactly the two-part message this project sent before few-shot existed,
  and ``build_cache_identity`` must return exactly the identity it used to. Both
  are what keep SQL cached by an earlier version reachable rather than orphaned.
* **The asked question is never shown as an example.** A pool that contains the
  question would otherwise turn a generation into a lookup, silently.
* **The cache key covers the selection.** Selection is a deterministic function
  of (question, pool, limit); the question is already in the key, so digesting
  the pool and limit is what makes that sound. The digest must therefore move
  when the pool's *content* moves and stay put when only its *order* does.

Selection is exercised through the public functions rather than the scorer:
what callers depend on is the ranking, and pinning individual similarity floats
would fail on any harmless retuning of the scorer while telling nobody whether
the ranking still holds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nl2sql import cli, llm
from nl2sql import examples as examples_module
from nl2sql.examples import (
    DEFAULT_LIMIT,
    Example,
    load_examples,
    pool_fingerprint,
    render_examples,
    select_examples,
    similarity,
)

# A small pool spanning three unrelated topics, so "the selector picked the
# right one" is a statement about meaning rather than about luck.
POOL = [
    Example("What is the total revenue by category?", "SELECT category FROM a"),
    Example("How many customers are in each region?", "SELECT region FROM b"),
    Example("Show revenue by category and month.", "SELECT month FROM c"),
    Example("What is the average order value?", "SELECT avg FROM d"),
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _write_jsonl(path: Path, records: list[object]) -> Path:
    lines = [
        record if isinstance(record, str) else json.dumps(record)
        for record in records
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_examples_reads_question_and_sql(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "pool.jsonl",
        [
            {"question": "Total revenue?", "sql": "SELECT 1"},
            {"question": "Order count?", "sql": "SELECT 2"},
        ],
    )
    assert load_examples(path) == [
        Example("Total revenue?", "SELECT 1"),
        Example("Order count?", "SELECT 2"),
    ]


def test_load_examples_ignores_extra_fields_and_blank_lines(tmp_path: Path) -> None:
    """The gold file carries an ``ordered`` flag this loader has no use for.

    Tolerating unknown keys is what lets one file serve both the eval harness
    and a few-shot pool without a second copy that can drift from the first.
    """
    path = _write_jsonl(
        tmp_path / "pool.jsonl",
        [
            {"question": "Total revenue?", "sql": "SELECT 1", "ordered": True},
            "",
            "   ",
            {"question": "Order count?", "sql": "SELECT 2"},
        ],
    )
    assert len(load_examples(path)) == 2


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ("{not json", "not valid JSON"),
        ('["question", "sql"]', "not a JSON object"),
        ('{"question": "Q?"}', "string 'question' and 'sql'"),
        ('{"question": "Q?", "sql": 7}', "string 'question' and 'sql'"),
    ],
)
def test_load_examples_rejects_a_malformed_record(
    tmp_path: Path, record: str, expected: str
) -> None:
    """A bad line raises, naming the file and line number.

    Skipping it would surface only as a prompt missing an example, which is
    indistinguishable from the selector finding nothing relevant — a broken file
    and normal operation must not look the same.
    """
    path = _write_jsonl(tmp_path / "pool.jsonl", [record])
    with pytest.raises(ValueError, match=expected) as excinfo:
        load_examples(path)
    assert "pool.jsonl:1" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def test_selection_prefers_examples_about_the_same_things() -> None:
    """The nearest example shares content words, not just a topic in general.

    "Break down total revenue by category" shares all three of *total*,
    *revenue* and *category* with the first pool entry and only two of four with
    the month one, so the ranking is decided by overlap rather than by the
    character-ratio tiebreaker.
    """
    chosen = select_examples("Break down total revenue by category", POOL, limit=1)
    assert [example.question for example in chosen] == [
        "What is the total revenue by category?"
    ]


def test_selection_drops_examples_with_no_shared_content_word() -> None:
    """An unrelated example is a misleading hint, not a weak one.

    The limit here is the whole pool, so there is room for all four entries and
    the three that share no content word are dropped on their merits rather than
    squeezed out by the budget. Without the overlap filter this returns four.
    """
    chosen = select_examples("Which region has the most customers?", POOL, limit=4)
    assert [example.question for example in chosen] == [
        "How many customers are in each region?"
    ]


def test_selection_returns_nothing_when_nothing_overlaps() -> None:
    assert select_examples("How is the weather in Lisbon?", POOL) == []


@pytest.mark.parametrize(
    "asked",
    [
        "What is the average order value?",
        "what is the average order value",
        "  What is the average ORDER value?  ",
        "What is the average   order value?",
    ],
)
def test_the_asked_question_is_never_offered_as_its_own_example(asked: str) -> None:
    """Case, spacing and trailing punctuation must not defeat the exclusion.

    This is the guard that stops a pool containing the question from turning
    generation into a lookup, so it has to hold for a question typed the way a
    person actually types one.
    """
    chosen = select_examples(asked, POOL, limit=len(POOL))
    assert all(
        example.question != "What is the average order value?" for example in chosen
    )


def test_a_repeated_question_does_not_spend_the_budget_twice() -> None:
    duplicated = [*POOL, Example("What is the total revenue by category?", "SELECT z")]
    chosen = select_examples("revenue by category", duplicated, limit=len(duplicated))
    questions = [example.question for example in chosen]
    assert len(questions) == len(set(questions))
    # First occurrence wins, so the duplicate's SQL is the one dropped.
    assert all(example.sql != "SELECT z" for example in chosen)


def test_selection_respects_the_limit() -> None:
    assert len(select_examples("revenue by category and region", POOL, limit=2)) == 2
    assert select_examples("revenue by category", POOL, limit=0) == []


def test_selection_is_deterministic() -> None:
    first = select_examples("revenue by category", POOL)
    second = select_examples("revenue by category", list(reversed(POOL)))
    assert first == second


def test_similarity_scores_overlap_ahead_of_string_resemblance() -> None:
    """Content overlap must outrank raw character resemblance, not merely agree.

    These two candidates are chosen because the character ratio prefers the
    *wrong* one: "reverse the catalogue" looks more like "revenue by category"
    letter by letter than "category revenue" does, while sharing none of its
    words. A scorer that led on string similarity would rank them the other way
    round, so this is the pair that tells the two designs apart.
    """
    on_topic_overlap, on_topic_ratio = similarity("revenue by category", "category revenue")
    off_topic_overlap, off_topic_ratio = similarity(
        "revenue by category", "reverse the catalogue"
    )

    assert off_topic_ratio > on_topic_ratio, "the pair no longer tests what it was picked for"
    assert on_topic_overlap > off_topic_overlap


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_an_empty_pool_renders_to_nothing() -> None:
    assert render_examples([]) == ""


def test_rendered_block_carries_every_question_and_its_sql() -> None:
    block = render_examples(POOL[:2])
    for example in POOL[:2]:
        assert f"Q: {example.question}" in block
        assert f"SQL: {example.sql}" in block


# --------------------------------------------------------------------------- #
# Cache identity
# --------------------------------------------------------------------------- #
def test_fingerprint_moves_with_the_pool_contents() -> None:
    baseline = pool_fingerprint(POOL)
    assert baseline == pool_fingerprint(POOL), "digest is not stable across calls"
    assert baseline != pool_fingerprint(POOL[:2]), "dropping an entry went unnoticed"
    assert baseline != pool_fingerprint(
        [*POOL[:3], Example(POOL[3].question, "SELECT different")]
    ), "rewriting an example's SQL went unnoticed"


def test_fingerprint_moves_when_the_prompt_header_is_reworded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The header is prompt text, so editing it must invalidate the cache.

    Exactly the reason the system prompt is fingerprinted: an edit changes what
    every few-shot prompt says, and a key that could not see it would go on
    serving SQL written under the old wording.
    """
    before = pool_fingerprint(POOL)
    monkeypatch.setattr(examples_module, "_EXAMPLES_HEADER", "Some other instruction:")
    assert pool_fingerprint(POOL) != before


def test_fingerprint_cannot_be_aliased_by_punctuation_inside_an_example() -> None:
    """Two different pools must not collide because of a separator character.

    A naive digest joining ``question`` and ``sql`` with a delimiter aliases a
    one-entry pool whose SQL contains that delimiter onto a two-entry pool — the
    classic way a hand-rolled key confuses two genuinely different inputs.
    """
    assert pool_fingerprint([Example("a", "x\nb\x00y")]) != pool_fingerprint(
        [Example("a", "x"), Example("b", "y")]
    )


def test_fingerprint_ignores_pool_order() -> None:
    """Reordering a pool selects the same examples, so it must not miss.

    The digest exists to stop a stale hit, not to maximise misses: a pool whose
    lines were sorted on disk produces identical prompts, and hashing it
    differently would throw away cache entries that are still correct.
    """
    assert pool_fingerprint(POOL) == pool_fingerprint(list(reversed(POOL)))


def test_zero_shot_cache_identity_is_unchanged_by_this_feature() -> None:
    """Entries cached before few-shot existed must stay reachable."""
    expected = f"gpt-4o-mini/{llm._prompt_fingerprint(llm._SYSTEM_PROMPT)}"
    assert llm.build_cache_identity("gpt-4o-mini") == expected
    assert "fewshot" not in llm.build_cache_identity("gpt-4o-mini")


def test_a_pool_changes_the_cache_identity() -> None:
    with_pool = llm.build_cache_identity("gpt-4o-mini", POOL)
    assert with_pool != llm.build_cache_identity("gpt-4o-mini")
    assert with_pool != llm.build_cache_identity("gpt-4o-mini", POOL[:2])
    assert with_pool.startswith(llm.build_cache_identity("gpt-4o-mini"))


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #
SCHEMA = "TABLE orders(id INTEGER)"


def test_zero_shot_prompt_is_byte_identical_to_the_original() -> None:
    """Pins the exact string, because the cache is keyed on the prompt."""
    assert (
        llm.build_user_message("Total revenue?", SCHEMA)
        == f"Schema:\n{SCHEMA}\n\nQuestion: Total revenue?"
    )


def test_examples_sit_between_the_schema_and_the_question() -> None:
    message = llm.build_user_message("Show revenue by category", SCHEMA, POOL)
    assert message.index("Schema:") < message.index("Q: ")
    assert message.index("Q: ") < message.index("Question: Show revenue by category")


def test_the_prompt_never_contains_the_answer_to_its_own_question() -> None:
    """End-to-end form of the self-exclusion guard, at the prompt boundary."""
    asked = "What is the average order value?"
    message = llm.build_user_message(asked, SCHEMA, POOL)
    assert "SELECT avg FROM d" not in message
    assert message.count(asked) == 1  # only as the question, never as an example


def test_the_prompt_holds_at_most_the_default_number_of_examples() -> None:
    crowded = [
        Example(f"What was revenue by category in 202{n}?", f"SELECT {n}")
        for n in range(8)
    ]
    message = llm.build_user_message("revenue by category", SCHEMA, crowded)
    assert message.count("\nQ: ") == DEFAULT_LIMIT


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #
def test_examples_without_llm_is_refused_rather_than_ignored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Silently ignoring it would answer the question the user did not ask.

    The offline backend reads no prompt, so a pool cannot reach it. Accepting
    the flag anyway would return a perfectly good answer with nothing to say the
    examples were never used.
    """
    pool = _write_jsonl(tmp_path / "pool.jsonl", [{"question": "Q?", "sql": "SELECT 1"}])
    assert cli.main(["ask", "total revenue", "--examples", str(pool)]) == 2
    assert "--examples" in capsys.readouterr().err


def test_an_unreadable_examples_file_is_fatal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unlike the gold file `rules` degrades over, this one cannot be skipped.

    There the examples decorate a listing that stands without them; here they
    are the change to the prompt the user asked for, so continuing would quietly
    send a different prompt than the one requested.
    """
    assert cli.main(["ask", "q", "--llm", "--examples", "/nonexistent/pool.jsonl"]) == 2
    assert "could not read --examples" in capsys.readouterr().err
