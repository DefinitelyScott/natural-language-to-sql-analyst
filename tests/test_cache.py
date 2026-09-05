"""Cover the on-disk SQL cache: key correctness, durability, and transparency.

The cache exists to avoid model calls, so the property that matters most is not
"a hit returns a string" but *when* a hit is allowed to happen. Every input that
changes what the model would write — the question, the schema, the model name,
the system prompt — must produce a different key, or the cache will confidently
replay an answer to a question nobody asked. Most of the tests below are that
one property, checked one input at a time.

A counting stub stands in for the LLM backend throughout. It records how many
times it was asked, which is what makes a hit observable: the visible difference
between a hit and a miss is not the SQL (identical by construction) but whether
the wrapped backend was called at all.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nl2sql import cache, cli, generator, llm

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")

needs_db = pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")


class CountingBackend:
    """A repairing backend that counts calls and returns a scripted answer."""

    def __init__(self, sql: str = "SELECT 1") -> None:
        self.sql = sql
        self.calls = 0
        self.repairs = 0

    def to_sql(self, question: str, schema: str) -> str:
        self.calls += 1
        return self.sql

    def repair(self, question: str, schema: str, sql: str, error: str) -> str:
        self.repairs += 1
        return f"{self.sql} /* repaired */"


class NonRepairingBackend:
    """Satisfies ``Backend`` but not ``RepairingBackend``."""

    def to_sql(self, question: str, schema: str) -> str:
        return "SELECT 1"


class ExplodingBackend:
    """Raises instead of answering, to prove failures are not cached."""

    def __init__(self) -> None:
        self.calls = 0

    def to_sql(self, question: str, schema: str) -> str:
        self.calls += 1
        raise RuntimeError("model unavailable")

    def repair(self, question: str, schema: str, sql: str, error: str) -> str:
        raise RuntimeError("model unavailable")


@pytest.fixture
def store(tmp_path: Path) -> cache.SqlCache:
    return cache.SqlCache(tmp_path / "sql_cache.json")


# --------------------------------------------------------------------------
# Key construction
# --------------------------------------------------------------------------


def test_identical_inputs_produce_the_same_key() -> None:
    assert cache.cache_key("m", "q", "s") == cache.cache_key("m", "q", "s")


@pytest.mark.parametrize(
    ("identity", "question", "schema"),
    [
        ("other-model", "q", "s"),
        ("m", "a different question", "s"),
        ("m", "q", "a different schema"),
    ],
    ids=["identity", "question", "schema"],
)
def test_changing_any_component_changes_the_key(
    identity: str, question: str, schema: str
) -> None:
    """Every keyed input must be able to invalidate on its own.

    These are the three ways a cached answer goes stale — a new model or prompt,
    a different question, a rebuilt database — and a key insensitive to any one
    of them would serve an answer written under assumptions that no longer hold.
    """
    assert cache.cache_key(identity, question, schema) != cache.cache_key("m", "q", "s")


def test_key_components_cannot_be_confused_by_concatenation() -> None:
    """Shifting a character across a component boundary must change the key.

    Without a separator, ``("ab", "c", "s")`` and ``("a", "bc", "s")`` would
    hash the same joined string — the classic way a naive cache key aliases two
    genuinely different requests onto one entry.
    """
    assert cache.cache_key("ab", "c", "s") != cache.cache_key("a", "bc", "s")


def test_question_case_and_spacing_are_significant() -> None:
    """The model sees the exact string, so the exact string is the key.

    Normalizing would raise the hit rate by asserting the model treats these as
    the same prompt, which the cache is in no position to know.
    """
    assert cache.cache_key("m", "Total revenue?", "s") != cache.cache_key(
        "m", "total revenue?", "s"
    )
    assert cache.cache_key("m", "total  revenue", "s") != cache.cache_key(
        "m", "total revenue", "s"
    )


# --------------------------------------------------------------------------
# SqlCache storage
# --------------------------------------------------------------------------


def test_get_returns_none_when_the_file_does_not_exist(store: cache.SqlCache) -> None:
    assert store.get("missing") is None


def test_put_then_get_round_trips(store: cache.SqlCache) -> None:
    store.put("k", "SELECT 1", question="q", identity="m")
    assert store.get("k") == "SELECT 1"


def test_a_second_cache_object_reads_what_the_first_wrote(tmp_path: Path) -> None:
    """The cache has to survive process exit, which is the whole point."""
    path = tmp_path / "sql_cache.json"
    cache.SqlCache(path).put("k", "SELECT 1", question="q", identity="m")
    assert cache.SqlCache(path).get("k") == "SELECT 1"


def test_stored_file_is_readable_json_naming_the_question(store: cache.SqlCache) -> None:
    """The file is meant to be inspectable by hand, not just by this module."""
    store.put("k", "SELECT 1", question="How much revenue?", identity="gpt/abc")
    payload = json.loads(Path(store.path).read_text(encoding="utf-8"))

    assert payload["entries"]["k"] == {
        "question": "How much revenue?",
        "identity": "gpt/abc",
        "sql": "SELECT 1",
    }


def test_put_overwrites_an_existing_key(store: cache.SqlCache) -> None:
    store.put("k", "SELECT 1", question="q", identity="m")
    store.put("k", "SELECT 2", question="q", identity="m")
    assert store.get("k") == "SELECT 2"


@pytest.mark.parametrize(
    "contents",
    ["not json at all", "[]", '{"version": 999, "entries": {"k": {"sql": "SELECT 1"}}}'],
    ids=["corrupt", "wrong-shape", "future-version"],
)
def test_an_unusable_file_reads_as_a_miss(tmp_path: Path, contents: str) -> None:
    """A broken cache must degrade to "no cache", never to an exception.

    Raising here would turn a disposable optimization into an outage: the user
    asked a question, and a stale artifact in ``data/`` is not a reason to
    refuse to answer it.
    """
    path = tmp_path / "sql_cache.json"
    path.write_text(contents, encoding="utf-8")
    assert cache.SqlCache(path).get("k") is None


def test_writing_over_an_unusable_file_recovers_it(tmp_path: Path) -> None:
    path = tmp_path / "sql_cache.json"
    path.write_text("not json at all", encoding="utf-8")

    store = cache.SqlCache(path)
    store.put("k", "SELECT 1", question="q", identity="m")
    assert store.get("k") == "SELECT 1"


def test_put_survives_an_unwritable_location(tmp_path: Path) -> None:
    """An impossible path costs the caching and nothing else."""
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")

    store = cache.SqlCache(blocker / "nested" / "sql_cache.json")
    store.put("k", "SELECT 1", question="q", identity="m")  # must not raise
    assert store.get("k") is None


def test_a_failed_write_leaves_no_temporary_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cache that cannot save must not litter the directory it failed in.

    The write is staged through a temporary file in the destination directory so
    that the swap is atomic; a failure after that file exists has to clean up
    after itself, or every attempt would leave another orphan behind.
    """

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(cache.os, "replace", explode)

    store = cache.SqlCache(tmp_path / "sql_cache.json")
    store.put("k", "SELECT 1", question="q", identity="m")

    assert list(tmp_path.iterdir()) == []


def test_entries_are_capped_and_evicted_oldest_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cache, "MAX_ENTRIES", 3)
    store = cache.SqlCache(tmp_path / "sql_cache.json")

    for index in range(5):
        store.put(f"k{index}", f"SELECT {index}", question=f"q{index}", identity="m")

    assert [store.get(f"k{index}") for index in range(5)] == [
        None,
        None,
        "SELECT 2",
        "SELECT 3",
        "SELECT 4",
    ]


def test_default_path_is_resolved_at_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Patching the module constant has to be enough to redirect the cache.

    If the default were bound in the signature it would freeze at import time,
    and a test that thought it was writing to ``tmp_path`` would quietly write
    to the developer's real cache instead.
    """
    redirected = tmp_path / "elsewhere.json"
    monkeypatch.setattr(cache, "DEFAULT_CACHE_PATH", str(redirected))
    assert cache.SqlCache().path == str(redirected)


# --------------------------------------------------------------------------
# CachedBackend
# --------------------------------------------------------------------------


def test_first_call_misses_and_second_call_hits(store: cache.SqlCache) -> None:
    backend = CountingBackend("SELECT 42")
    wrapped = cache.CachedBackend(backend, store, "gpt/abc")

    assert wrapped.lookup("q", "schema") == ("SELECT 42", False)
    assert wrapped.lookup("q", "schema") == ("SELECT 42", True)
    assert backend.calls == 1


def test_a_hit_is_reached_from_a_fresh_wrapper(store: cache.SqlCache) -> None:
    """The cache is shared state on disk, not memoization inside one object."""
    first = CountingBackend("SELECT 42")
    cache.CachedBackend(first, store, "gpt/abc").lookup("q", "schema")

    second = CountingBackend("SELECT 99")
    sql, hit = cache.CachedBackend(second, store, "gpt/abc").lookup("q", "schema")

    assert (sql, hit) == ("SELECT 42", True)
    assert second.calls == 0


@pytest.mark.parametrize(
    ("identity", "question", "schema"),
    [
        ("gpt/xyz", "q", "schema"),
        ("gpt/abc", "another question", "schema"),
        ("gpt/abc", "q", "another schema"),
    ],
    ids=["identity", "question", "schema"],
)
def test_a_changed_input_forces_regeneration(
    store: cache.SqlCache, identity: str, question: str, schema: str
) -> None:
    """The key tests above, exercised end-to-end through the wrapper."""
    cache.CachedBackend(CountingBackend(), store, "gpt/abc").lookup("q", "schema")

    backend = CountingBackend()
    _, hit = cache.CachedBackend(backend, store, identity).lookup(question, schema)

    assert not hit
    assert backend.calls == 1


def test_to_sql_matches_lookup_and_still_populates_the_cache(
    store: cache.SqlCache,
) -> None:
    """The wrapper has to remain a drop-in ``Backend`` for the rest of the pipeline."""
    backend = CountingBackend("SELECT 7")
    wrapped = cache.CachedBackend(backend, store, "gpt/abc")

    assert wrapped.to_sql("q", "schema") == "SELECT 7"
    assert wrapped.to_sql("q", "schema") == "SELECT 7"
    assert backend.calls == 1


def test_a_failed_generation_is_not_cached(store: cache.SqlCache) -> None:
    """A replayed failure would be permanent; a real retry is the honest behavior."""
    backend = ExplodingBackend()
    wrapped = cache.CachedBackend(backend, store, "gpt/abc")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="model unavailable"):
            wrapped.lookup("q", "schema")

    assert backend.calls == 2


def test_repairs_delegate_and_are_never_cached(store: cache.SqlCache) -> None:
    backend = CountingBackend("SELECT 1")
    wrapped = cache.CachedBackend(backend, store, "gpt/abc")

    first = wrapped.repair("q", "schema", "SELECT bad", "no such column")
    second = wrapped.repair("q", "schema", "SELECT bad", "no such column")

    assert first == second == "SELECT 1 /* repaired */"
    assert backend.repairs == 2


def test_wrapping_a_non_repairing_backend_is_refused(store: cache.SqlCache) -> None:
    """Better a clear constructor error than an AttributeError inside the repair loop.

    ``generator.answer_question`` decides whether to attempt a repair by testing
    ``isinstance(backend, llm.RepairingBackend)``, and the wrapper defines
    ``repair`` unconditionally — so wrapping a backend that cannot repair would
    advertise a capability it does not have.
    """
    with pytest.raises(TypeError, match="does not implement repair"):
        cache.CachedBackend(NonRepairingBackend(), store, "gpt/abc")


# --------------------------------------------------------------------------
# Backend capability protocols
# --------------------------------------------------------------------------


def test_offline_backend_is_not_cacheable() -> None:
    """Caching a regex scan would add a file read to the cheapest path there is.

    ``generator._resolve`` gates the wrap on this protocol, so this is what
    keeps offline runs — the whole test suite and CI — off the cache entirely.
    """
    assert not isinstance(llm.OfflineBackend(), cache.CacheableBackend)


def test_a_backend_reporting_an_identity_is_cacheable() -> None:
    class Identified(CountingBackend):
        @property
        def cache_identity(self) -> str:
            return "stub/000"

    assert isinstance(Identified(), cache.CacheableBackend)


def test_llm_backend_identity_covers_model_and_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Editing the system prompt must invalidate everything cached under it.

    ``LLMBackend.__init__`` builds an OpenAI client, so the identity property is
    read off an uninitialised instance with the one field it actually depends on
    set by hand — the alternative is a network dependency in a unit test.
    """
    backend = llm.LLMBackend.__new__(llm.LLMBackend)
    backend._model = "gpt-4o-mini"

    before = backend.cache_identity
    assert before.startswith("gpt-4o-mini/")

    monkeypatch.setattr(llm, "_SYSTEM_PROMPT", "a different system prompt")
    assert backend.cache_identity != before

    backend._model = "some-other-model"
    assert not backend.cache_identity.startswith("gpt-4o-mini/")


# --------------------------------------------------------------------------
# Wiring through the generator
# --------------------------------------------------------------------------


class IdentifiedBackend(CountingBackend):
    """A cacheable, repairing stub — what ``get_backend`` returns under ``--llm``."""

    @property
    def cache_identity(self) -> str:
        return "stub/000"


@pytest.fixture
def redirected_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the default cache at ``tmp_path`` for tests that go via the generator."""
    monkeypatch.setattr(cache, "DEFAULT_CACHE_PATH", str(tmp_path / "sql_cache.json"))


@needs_db
def test_generator_reports_a_cache_hit_on_the_explanation(
    monkeypatch: pytest.MonkeyPatch, redirected_cache: None
) -> None:
    backend = IdentifiedBackend("SELECT 1")
    monkeypatch.setattr(llm, "get_backend", lambda use_llm: backend)  # noqa: ARG005

    first = generator.explain_question(DB, "q", use_llm=True, use_cache=True)
    second = generator.explain_question(DB, "q", use_llm=True, use_cache=True)

    assert (first.cached, second.cached) == (False, True)
    assert backend.calls == 1


@needs_db
def test_no_cache_bypasses_both_the_read_and_the_write(
    monkeypatch: pytest.MonkeyPatch, redirected_cache: None
) -> None:
    """``use_cache=False`` must not merely skip the lookup — it must not populate.

    A run that writes while refusing to read would make ``--no-cache`` seed the
    very entries the next default run replays, which is the opposite of what a
    user disabling the cache is asking for.
    """
    backend = IdentifiedBackend("SELECT 1")
    monkeypatch.setattr(llm, "get_backend", lambda use_llm: backend)  # noqa: ARG005

    generator.explain_question(DB, "q", use_llm=True, use_cache=False)
    assert not os.path.exists(cache.DEFAULT_CACHE_PATH)

    assert not generator.explain_question(DB, "q", use_llm=True, use_cache=True).cached
    assert backend.calls == 2


@needs_db
def test_offline_answers_are_never_cached(redirected_cache: None) -> None:
    """The default path writes no cache file at all, so CI never depends on one."""
    answer = generator.answer_question(DB, "How many customers do we have?", use_cache=True)

    assert not answer.cached
    assert not os.path.exists(cache.DEFAULT_CACHE_PATH)


# --------------------------------------------------------------------------
# CLI disclosure
# --------------------------------------------------------------------------


@needs_db
def test_ask_discloses_a_replay_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    redirected_cache: None,
) -> None:
    """A replayed answer must be distinguishable from a fresh one.

    On stderr rather than stdout, like the truncation and repair notices: it is
    a diagnostic about the run, so it must not land in a redirected
    ``--format csv`` file.
    """
    backend = IdentifiedBackend("SELECT COUNT(*) AS n FROM orders")
    monkeypatch.setattr(llm, "get_backend", lambda use_llm: backend)  # noqa: ARG005
    command = ["ask", "how many orders", "--db", DB, "--llm"]

    assert cli.main(command) == 0
    assert "replayed from the local cache" not in capsys.readouterr().err

    assert cli.main(command) == 0
    first_replay = capsys.readouterr()
    assert "replayed from the local cache" in first_replay.err
    assert "replayed from the local cache" not in first_replay.out

    assert cli.main([*command, "--no-cache"]) == 0
    assert "replayed from the local cache" not in capsys.readouterr().err
    assert backend.calls == 2


@needs_db
def test_explain_names_the_cache_as_the_source(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    redirected_cache: None,
) -> None:
    backend = IdentifiedBackend("SELECT COUNT(*) AS n FROM orders")
    monkeypatch.setattr(llm, "get_backend", lambda use_llm: backend)  # noqa: ARG005
    command = ["explain", "how many orders", "--db", DB, "--llm"]

    assert cli.main(command) == 0
    assert "Source:" not in capsys.readouterr().out

    assert cli.main(command) == 0
    assert "Source:   local cache" in capsys.readouterr().out


@needs_db
def test_offline_ask_never_mentions_the_cache(capsys: pytest.CaptureFixture[str]) -> None:
    """The default path is uncached, so the note would be a lie there."""
    for _ in range(2):
        assert cli.main(["ask", "How many customers do we have?", "--db", DB]) == 0
        assert "cache" not in capsys.readouterr().err
