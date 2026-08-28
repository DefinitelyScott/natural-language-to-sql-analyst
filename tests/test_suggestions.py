"""Tests for nearest-question suggestions on an unrecognized question.

Two layers are covered: the ranking itself (`catalog.suggest_questions`) and the
CLI path that reaches for it (`nl2sql ask` / `nl2sql explain` on a question no
rule matches). The ranking tests assert the properties that make a suggestion
trustworthy — it is relevant, it is answerable, and it is not invented to pad a
list — rather than pinning an exact ordering of an evolving catalog.
"""

import os

import pytest

from nl2sql import catalog, cli, llm

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DB = os.path.join(REPO_ROOT, "data", "store.db")
GOLD = os.path.join(REPO_ROOT, "evals", "gold.jsonl")

needs_db = pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")

CANDIDATES = [
    "Show revenue by region",
    "Show revenue by category",
    "How many customers do we have?",
    "What is the average order value?",
]


def test_suggests_the_candidate_sharing_the_most_content_words():
    assert catalog.suggest_questions("revenue by region please", CANDIDATES, limit=1) == [
        "Show revenue by region"
    ]


def test_returns_nothing_when_no_content_word_overlaps():
    """A padded list would read as a guess; an empty one is the honest answer."""
    assert catalog.suggest_questions("what is the meaning of life?", CANDIDATES) == []


def test_stopwords_alone_do_not_make_a_question_relevant():
    """"How many ..." shares only filler with the count question, so it must not match.

    This is what the stopword list buys: without it, every question phrased as a
    request would look similar to every other one.
    """
    assert catalog.suggest_questions("how do we show that to me", CANDIDATES) == []


def test_respects_the_limit_and_ranks_the_best_first():
    suggestions = catalog.suggest_questions("revenue by category", CANDIDATES, limit=2)
    assert len(suggestions) == 2
    assert suggestions[0] == "Show revenue by category"


def test_punctuation_does_not_change_tokenization():
    """"month-over-month" and "month over month" must score identically."""
    hyphenated = catalog.suggest_questions(
        "month-over-month revenue", ["Show month over month revenue growth in 2024."]
    )
    spaced = catalog.suggest_questions(
        "month over month revenue", ["Show month over month revenue growth in 2024."]
    )
    assert hyphenated == spaced != []


def test_duplicate_candidates_are_offered_once():
    duplicated = ["Show revenue by region", "Show revenue by region"]
    assert catalog.suggest_questions("revenue by region", duplicated) == [
        "Show revenue by region"
    ]


def test_ranking_is_deterministic_regardless_of_candidate_order():
    forward = catalog.suggest_questions("revenue", CANDIDATES)
    backward = catalog.suggest_questions("revenue", list(reversed(CANDIDATES)))
    assert forward == backward


def test_answerable_questions_drops_entries_without_an_example():
    entries = [
        catalog.CatalogEntry(index=0, pattern="a", example="Show revenue by region"),
        catalog.CatalogEntry(index=1, pattern="b", example=None),
    ]
    assert catalog.answerable_questions(entries) == ["Show revenue by region"]


def test_every_suggestion_from_the_real_catalog_is_answerable():
    """A suggestion the backend cannot answer would fail exactly as the user's did."""
    backend = llm.OfflineBackend()
    entries = catalog.build_catalog(backend, catalog.load_example_questions(GOLD))
    answerable = catalog.answerable_questions(entries)

    suggestions = catalog.suggest_questions("revenue by store", answerable)
    assert suggestions, "expected 'revenue' to overlap the catalog"
    for suggestion in suggestions:
        # Raises NoRuleMatchError if the catalog cannot route it.
        assert backend.to_sql(suggestion, "").upper().lstrip().startswith(
            ("SELECT", "WITH")
        )


def test_offline_backend_raises_the_specific_no_match_error():
    """The CLI branches on this type, so a plain ValueError would silently
    disable the suggestion path."""
    with pytest.raises(llm.NoRuleMatchError):
        llm.OfflineBackend().to_sql("what is the meaning of life?", "")


@needs_db
def test_ask_suggests_nearest_questions_on_a_miss(capsys):
    assert cli.main(["ask", "revenue by store", "--db", DB]) == 1
    err = capsys.readouterr().err
    assert "Did you mean:" in err
    assert "revenue" in err.lower()
    assert "nl2sql rules" in err


@needs_db
def test_explain_suggests_nearest_questions_on_a_miss(capsys):
    assert cli.main(["explain", "revenue by store", "--db", DB]) == 1
    err = capsys.readouterr().err
    assert "Did you mean:" in err
    assert "nl2sql rules" in err


@needs_db
def test_miss_with_no_overlap_still_points_at_the_catalog(capsys):
    """No relevant suggestion is not the same as no help."""
    assert cli.main(["ask", "what is the meaning of life?", "--db", DB]) == 1
    err = capsys.readouterr().err
    assert "Did you mean:" not in err
    assert "nl2sql rules" in err


@needs_db
def test_suggestions_go_to_stderr_only(capsys):
    """They are a diagnostic; stdout stays a clean data stream."""
    cli.main(["ask", "revenue by store", "--db", DB, "--format", "csv"])
    captured = capsys.readouterr()
    assert "Did you mean:" not in captured.out
