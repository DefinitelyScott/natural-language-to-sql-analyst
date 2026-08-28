"""Tests for the rule catalog listing (`nl2sql.catalog`) and the `rules` command.

Two things are worth pinning here beyond "it prints something".

First, the example question shown for a rule must be one that *routes to that
rule*, not merely one that matches its pattern. Those differ whenever a question
matches several rules, which is common in a first-rule-wins catalog, and getting
it wrong would print an example that produces a different rule's SQL — the worst
kind of documentation, because it looks verified.

Second, the listing must degrade rather than fail when the gold file it draws
examples from is missing or malformed. The rules themselves come from the
backend, so they are still answerable; losing the examples is a downgrade, not
an error.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from nl2sql import catalog, cli
from nl2sql.llm import OfflineBackend

GOLD = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evals", "gold.jsonl")


class _StubBackend:
    """A three-rule catalog with a deliberate overlap, standing in for the real one.

    Rules 0 and 1 both match "revenue by region", with 0 winning; rule 2 matches
    nothing any test question says. That is the exact shape the real catalog has
    (specific patterns ahead of broad ones, plus the possibility of an
    unreachable rule) reduced to something a test can assert on exhaustively.
    """

    _PATTERNS = (r"revenue by region", r"revenue", r"unreachable pattern")

    def rule_count(self) -> int:
        return len(self._PATTERNS)

    def rule_pattern(self, index: int) -> str:
        return self._PATTERNS[index]

    def matching_rule_indexes(self, question: str) -> list[int]:
        return [
            index
            for index, pattern in enumerate(self._PATTERNS)
            if re.search(pattern, question, re.I)
        ]


# --------------------------------------------------------------------------- #
# build_catalog
# --------------------------------------------------------------------------- #
def test_build_catalog_returns_one_entry_per_rule_in_order():
    entries = catalog.build_catalog(_StubBackend())
    assert [entry.index for entry in entries] == [0, 1, 2]
    assert [entry.pattern for entry in entries] == list(_StubBackend._PATTERNS)


def test_example_is_credited_to_the_rule_that_would_answer_it():
    """A question matching two rules must only appear against the winner.

    "Show revenue by region" matches both rule 0 and rule 1, but rule 0 answers
    it. Showing it against rule 1 would advertise an example that returns rule
    0's SQL.
    """
    entries = catalog.build_catalog(_StubBackend(), ["Show revenue by region"])
    assert entries[0].example == "Show revenue by region"
    assert entries[1].example is None


def test_rule_no_question_reaches_is_listed_with_no_example():
    """An unreachable rule stays in the listing — it is evidence, not noise."""
    entries = catalog.build_catalog(_StubBackend(), ["Show revenue by region"])
    assert entries[2].example is None
    assert entries[2].pattern == "unreachable pattern"


def test_first_example_wins_when_two_resolve_to_the_same_rule():
    """Collisions are a defect elsewhere; here they only have to be deterministic."""
    entries = catalog.build_catalog(
        _StubBackend(), ["Show revenue by region", "revenue by region please"]
    )
    assert entries[0].example == "Show revenue by region"


def test_every_real_rule_has_an_example_from_the_gold_set():
    """The shipped listing must be fully populated.

    ``test_rule_catalog.py`` asserts the same invariant from the routing side;
    asserting it here too is what guarantees the user-facing command never
    prints "(no example)" for the repo as shipped.
    """
    entries = catalog.build_catalog(
        OfflineBackend(), catalog.load_example_questions(GOLD)
    )
    missing = [entry.index for entry in entries if entry.example is None]
    assert not missing, f"offline rules with no gold example question: {missing}"


# --------------------------------------------------------------------------- #
# load_example_questions
# --------------------------------------------------------------------------- #
def test_load_example_questions_reads_questions_and_skips_blank_lines(tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text(
        json.dumps({"question": "one", "sql": "SELECT 1"})
        + "\n\n"
        + json.dumps({"question": "two", "sql": "SELECT 2"})
        + "\n",
        encoding="utf-8",
    )
    assert catalog.load_example_questions(path) == ["one", "two"]


@pytest.mark.parametrize(
    "line",
    [
        "{not json",
        json.dumps({"sql": "SELECT 1"}),  # no question field
        json.dumps({"question": 7}),  # question is not a string
        json.dumps(["question"]),  # record is not an object
    ],
)
def test_load_example_questions_rejects_malformed_records(tmp_path, line):
    """A bad record raises rather than being skipped.

    Skipping it would surface later as a rule with no example, which is
    indistinguishable from a shadowing bug and would send a reader hunting in
    the wrong file.
    """
    path = tmp_path / "gold.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="1:"):
        catalog.load_example_questions(path)


# --------------------------------------------------------------------------- #
# filter_catalog
# --------------------------------------------------------------------------- #
def test_filter_catalog_matches_example_and_pattern_case_insensitively():
    entries = catalog.build_catalog(_StubBackend(), ["Show revenue by region"])

    by_example = catalog.filter_catalog(entries, "SHOW REVENUE")
    assert [entry.index for entry in by_example] == [0]

    # Rule 2 has no example, so only its pattern can match.
    by_pattern = catalog.filter_catalog(entries, "unreachable")
    assert [entry.index for entry in by_pattern] == [2]

    assert catalog.filter_catalog(entries, "no such topic") == []


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_rules_command_lists_the_catalog(capsys):
    assert cli.main(["rules"]) == 0
    out = capsys.readouterr().out
    assert f"{OfflineBackend().rule_count()} offline rule(s)" in out
    assert "How many customers do we have?" in out
    assert "(no example)" not in out


def test_rules_command_needs_no_database(capsys, monkeypatch):
    """`rules` reads only the in-process catalog, so a missing DB is irrelevant.

    Every other command exits 2 when the database is absent. Pointing the
    default at a path that does not exist proves `rules` is exempt rather than
    merely lucky that the sample DB happens to be built.
    """
    monkeypatch.setattr(cli, "_DEFAULT_DB", "/nonexistent/nope.db")
    assert cli.main(["rules"]) == 0
    assert "Database not found" not in capsys.readouterr().err
    # ...and the same absent database still stops a command that needs one.
    assert cli.main(["schema", "--db", "/nonexistent/nope.db"]) == 2


def test_rules_command_json_output_is_parseable(capsys):
    assert cli.main(["rules", "--format", "json"]) == 0
    records = json.loads(capsys.readouterr().out)
    assert len(records) == OfflineBackend().rule_count()
    assert [record["rule"] for record in records] == list(range(len(records)))
    assert set(records[0]) == {"rule", "example", "pattern"}


def test_rules_command_search_filters_the_listing(capsys):
    assert cli.main(["rules", "--search", "region", "--format", "json"]) == 0
    records = json.loads(capsys.readouterr().out)
    assert records, "expected at least one region rule in the catalog"
    assert len(records) < OfflineBackend().rule_count()
    for record in records:
        assert "region" in (record["example"] + record["pattern"]).lower()


def test_rules_command_search_with_no_match_exits_1(capsys):
    """grep convention: a script can gate on whether the catalog covers a topic."""
    assert cli.main(["rules", "--search", "zzz-no-such-topic"]) == 1
    captured = capsys.readouterr()
    assert "No offline rules match" in captured.err
    assert captured.out == ""


def test_rules_command_survives_a_missing_gold_file(capsys):
    """Examples are a nice-to-have; the rules themselves still list."""
    assert cli.main(["rules", "--gold", "/nonexistent/gold.jsonl"]) == 0
    captured = capsys.readouterr()
    assert "Warning: no example questions" in captured.err
    assert f"{OfflineBackend().rule_count()} offline rule(s)" in captured.out
    assert "(no example)" in captured.out


def test_rules_command_survives_a_malformed_gold_file(capsys, tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text("{not json\n", encoding="utf-8")
    assert cli.main(["rules", "--gold", str(path)]) == 0
    captured = capsys.readouterr()
    assert "Warning: no example questions" in captured.err
    assert "(no example)" in captured.out
