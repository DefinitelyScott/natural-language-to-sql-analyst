"""Tests for ``LLMBackend`` — the one class in the package that calls a model.

Everything else in ``nl2sql`` was reachable from a test without much ceremony,
so everything else had tests. This class did not: its constructor reads
``OPENAI_API_KEY`` and lazily imports ``openai``, and CI installs neither. The
gap that left was not cosmetic. ``to_sql`` and ``repair`` assemble the exact
request sent to the model — the system prompt, ``temperature=0``, the message
order, which of the two prompts is used, whether few-shot examples are attached
— and every one of those was asserted only by ``tests/test_repair.py`` stubs
that *stand in for* this class rather than exercise it. A stub agreeing with
itself proves nothing about the real request.

The fix is to install a fake ``openai`` module in ``sys.modules`` before the
backend is constructed, so the lazy ``from openai import OpenAI`` inside
``__init__`` resolves to a recorder. Two reasons to do it this way rather than
by injecting a client through the constructor:

* It runs in CI unchanged. ``openai`` is deliberately absent there (see the
  mypy override in ``pyproject.toml``), and a test that needed the real package
  installed would be a test that never ran on the offline path this project is
  built around.
* It leaves the production signature alone. Adding a ``client`` parameter would
  put a seam in shipped code that exists only for tests; the lazy import is
  already a seam, and ``monkeypatch.setitem`` restores ``sys.modules`` after
  each test.

What is *not* claimed here: none of this checks that a real model writes good
SQL. That is what ``evals/`` measures. These tests check that the request we
believe we are sending is the request we send, and that the reply is unwrapped
the way the rest of the pipeline expects.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field

import pytest

from nl2sql import llm, runner
from nl2sql.examples import Example

QUESTION = "What was revenue by category?"
SCHEMA = "TABLE order_items(id, order_id, product_id, quantity, unit_price)"

#: The one pool entry ``select_examples`` keeps for ``QUESTION``. The other
#: shares no content word with it and is dropped rather than ranked last, so it
#: is the wrong string to look for when checking that a pool reached a prompt —
#: its absence would mean the selector is working, not that the pool was lost.
SELECTED_EXAMPLE_SQL = "SELECT ROUND(SUM(quantity * unit_price), 2) FROM order_items"
POOL = (
    Example("How many customers do we have?", "SELECT COUNT(*) FROM customers"),
    Example("What was total revenue?", SELECTED_EXAMPLE_SQL),
)


# --------------------------------------------------------------------------- #
# A fake OpenAI client
# --------------------------------------------------------------------------- #
# Shaped to the one access path ``LLMBackend`` uses —
# ``client.chat.completions.create(...).choices[0].message.content`` — and no
# more. Mirroring more of the SDK's surface would only create more ways for
# this double to drift from it.
@dataclass
class _Message:
    content: str | None


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]


@dataclass
class _Transcript:
    """Everything the backend said to the client, and what it was told back."""

    replies: list[str | None]
    calls: list[dict[str, object]] = field(default_factory=list)
    api_key: str | None = None

    @property
    def last_call(self) -> dict[str, object]:
        assert self.calls, "the backend never called the model"
        return self.calls[-1]

    def messages(self) -> list[dict[str, str]]:
        messages = self.last_call["messages"]
        assert isinstance(messages, list)
        return messages

    def system_prompt(self) -> str:
        return self.messages()[0]["content"]

    def user_message(self) -> str:
        return self.messages()[1]["content"]


class _Completions:
    def __init__(self, transcript: _Transcript) -> None:
        self._transcript = transcript

    def create(self, **kwargs: object) -> _Response:
        self._transcript.calls.append(kwargs)
        if not self._transcript.replies:
            raise AssertionError(
                f"the backend made {len(self._transcript.calls)} call(s) but only "
                f"{len(self._transcript.calls) - 1} repl(y/ies) were scripted"
            )
        return _Response([_Choice(_Message(self._transcript.replies.pop(0)))])


class _Chat:
    def __init__(self, transcript: _Transcript) -> None:
        self.completions = _Completions(transcript)


@pytest.fixture
def fake_openai(monkeypatch):
    """Return a helper that installs a fake ``openai`` and returns the transcript.

    Call it with the replies the model should give, in order. The returned
    ``_Transcript`` records the API key the client was built with and every
    request that reached it.
    """

    def install(*replies: str | None) -> _Transcript:
        transcript = _Transcript(replies=list(replies))

        class _FakeOpenAI:
            def __init__(self, api_key: str) -> None:
                transcript.api_key = api_key
                self.chat = _Chat(transcript)

        module = types.ModuleType("openai")
        module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "openai", module)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        return transcript

    return install


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def test_a_missing_api_key_is_reported_before_the_import(monkeypatch):
    """The key check must come first, or the error names the wrong problem.

    Someone running ``--llm`` without a key and without the optional ``openai``
    extra installed has two things missing. Only one of them is the thing to
    tell them about first, and the order of the two statements in ``__init__``
    is what decides which error they see. If the import ran first they would get
    ``ModuleNotFoundError: No module named 'openai'`` and go install a package
    that would then fail for the original reason.

    No fake module is installed here, so this also holds on a machine where
    ``openai`` genuinely is not available — which is the machine the message is
    written for.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delitem(sys.modules, "openai", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        llm.LLMBackend()


def test_the_client_is_built_with_the_environment_key(fake_openai):
    transcript = fake_openai()

    llm.LLMBackend()

    assert transcript.api_key == "test-key"


def test_a_constructed_backend_satisfies_the_repairing_protocol(fake_openai):
    """``tests/test_repair.py`` checks the class has the method; this checks the object.

    ``generator.answer_question`` decides whether to retry with
    ``isinstance(backend, RepairingBackend)`` on the instance it was handed, so
    the instance is where the property has to hold.
    """
    fake_openai()

    assert isinstance(llm.LLMBackend(), llm.RepairingBackend)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def test_generation_sends_the_system_prompt_and_the_assembled_user_message(fake_openai):
    """The whole request, pinned field by field.

    ``temperature=0`` is the load-bearing one: the cache keys entries on the
    model and prompt alone, so a sampling temperature above zero would make two
    cache hits for one key legitimately disagree.
    """
    transcript = fake_openai("SELECT 1")

    llm.LLMBackend(model="gpt-4o-mini").to_sql(QUESTION, SCHEMA)

    call = transcript.last_call
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == 0
    assert [message["role"] for message in transcript.messages()] == ["system", "user"]
    assert transcript.system_prompt() == llm._SYSTEM_PROMPT
    # Compared against the builder rather than against a literal: the exact
    # bytes of this message are what the cache is keyed on, and
    # ``tests/test_examples.py`` already pins the builder's output. Asserting
    # equality here is what proves the backend actually routes through it.
    assert transcript.user_message() == llm.build_user_message(QUESTION, SCHEMA)


def test_generation_strips_a_markdown_fence_from_the_reply(fake_openai):
    """Models fence their output despite being told not to; the runner needs bare SQL."""
    fake_openai("```sql\nSELECT 1\n```")

    assert llm.LLMBackend().to_sql(QUESTION, SCHEMA) == "SELECT 1"


def test_an_empty_reply_becomes_an_empty_string(fake_openai):
    """``message.content`` is optional in the SDK's types, so ``None`` is reachable.

    Returning ``""`` hands the failure to ``runner.validate``, which rejects an
    empty query by name. Letting the ``None`` through would surface as an
    ``AttributeError`` several frames away instead, so the second assertion is
    what makes the first one worth making.
    """
    fake_openai(None)

    sql = llm.LLMBackend().to_sql(QUESTION, SCHEMA)

    assert sql == ""
    with pytest.raises(runner.UnsafeQueryError, match="empty query"):
        runner.validate(sql)


def test_few_shot_examples_reach_the_generation_prompt(fake_openai):
    transcript = fake_openai("SELECT 1")

    llm.LLMBackend(examples=POOL).to_sql(QUESTION, SCHEMA)

    assert transcript.user_message() == llm.build_user_message(QUESTION, SCHEMA, POOL)
    assert SELECTED_EXAMPLE_SQL in transcript.user_message()


def test_the_example_pool_is_copied_at_construction(fake_openai):
    """A caller mutating the list it passed must not reshape prompts mid-run.

    The backend is reused across questions and its ``cache_identity`` is
    fingerprinted from the pool. If the pool were held by reference, an append
    between two questions would change both the prompt and the identity the
    second one is stored under, and the cache would end up holding two entries
    that look like one configuration.

    The appended entry is phrased to be *more* similar to ``QUESTION`` than
    anything already in the pool, so a shared reference would put it at the top
    of the selection rather than merely somewhere in it — the assertion would
    otherwise pass on a live pool simply because the selector ranked the
    intruder out.
    """
    transcript = fake_openai("SELECT 1")
    pool = list(POOL)
    backend = llm.LLMBackend(examples=pool)
    identity_before = backend.cache_identity

    pool.append(Example("What was revenue by category last year?", "SELECT 'leaked'"))
    backend.to_sql(QUESTION, SCHEMA)

    assert backend.cache_identity == identity_before
    assert "leaked" not in transcript.user_message()


def test_a_zero_shot_backend_sends_no_examples_block(fake_openai):
    """The default path must stay byte-identical to the prompt that predates few-shot."""
    transcript = fake_openai("SELECT 1")

    llm.LLMBackend().to_sql(QUESTION, SCHEMA)

    assert transcript.user_message() == f"Schema:\n{SCHEMA}\n\nQuestion: {QUESTION}"


# --------------------------------------------------------------------------- #
# Repair
# --------------------------------------------------------------------------- #
FAILED_SQL = "SELECT nonexistent_column FROM customers"
ERROR = "no such column: nonexistent_column"


def test_repair_uses_the_repair_prompt_and_carries_the_failure(fake_openai):
    """The failed SQL and the engine's error are what make a rewrite better than a re-ask.

    At ``temperature=0`` a bare re-ask returns the same query, so if either of
    these two fields were dropped from the message the repair loop would spend a
    second API call to produce the identical failure.
    """
    transcript = fake_openai("SELECT 1")

    llm.LLMBackend().repair(QUESTION, SCHEMA, FAILED_SQL, ERROR)

    assert transcript.system_prompt() == llm._REPAIR_PROMPT
    user_message = transcript.user_message()
    assert FAILED_SQL in user_message
    assert ERROR in user_message
    # Re-sent rather than carried in a conversation history, which is what makes
    # one repair independent of any other.
    assert QUESTION in user_message
    assert SCHEMA in user_message


def test_repair_omits_the_few_shot_examples(fake_openai):
    """Pins the claim ``LLMBackend.repair``'s docstring makes.

    A repair is driven by an error naming one specific thing; a neighbouring
    example is a much vaguer signal, and spending context on it dilutes the
    input that makes the rewrite work. Nothing enforces that except this test —
    the two prompts are assembled by separate code paths, and the generation
    path's inclusion of examples is one copied line away from appearing here.
    """
    transcript = fake_openai("SELECT 1")

    llm.LLMBackend(examples=POOL).repair(QUESTION, SCHEMA, FAILED_SQL, ERROR)

    assert SELECTED_EXAMPLE_SQL not in transcript.user_message()


def test_repair_strips_fences_too(fake_openai):
    """The same unwrapping as generation: a repair goes to the same validator."""
    fake_openai("```sql\nSELECT 1\n```")

    assert llm.LLMBackend().repair(QUESTION, SCHEMA, FAILED_SQL, ERROR) == "SELECT 1"


# --------------------------------------------------------------------------- #
# Fence stripping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("SELECT 1", "SELECT 1"),
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        # Case-insensitive: models emit ```SQL and ```Sql as readily as ```sql.
        ("```SQL\nSELECT 1\n```", "SELECT 1"),
        ("```\nSELECT 1\n```", "SELECT 1"),
        # A one-sided fence is still a fence. Truncated replies produce these.
        ("```sql\nSELECT 1", "SELECT 1"),
        ("SELECT 1\n```", "SELECT 1"),
        ("  \n```sql\nSELECT 1\n```  \n", "SELECT 1"),
        ("", ""),
    ],
)
def test_fence_stripping(reply, expected):
    assert llm._strip_fences(reply) == expected


def test_a_prose_preamble_is_not_stripped_and_is_caught_downstream():
    """An honest boundary: this unwraps fences, it does not extract SQL from prose.

    A model that ignores the "no prose" instruction and answers "Here you go:"
    before the fence leaves the opening fence in place — the leading-fence regex
    is anchored at the start of the reply — so the result is not SQL. That is
    the deliberate scope: a stripper that hunted for a fenced block anywhere in
    the text would also have to decide which block to take when a model emits
    two, and guessing wrong there is worse than not guessing.

    It is safe to leave unhandled only because the next layer rejects it, which
    is the half worth asserting: the reply never reaches SQLite, it raises
    ``UnsafeQueryError`` naming the real problem, and the repair loop gets a
    chance at it like any other rejected query.
    """
    reply = "Here you go:\n```sql\nSELECT 1\n```"

    stripped = llm._strip_fences(reply)

    assert stripped.startswith("Here you go:")
    with pytest.raises(runner.UnsafeQueryError, match="only SELECT/WITH"):
        runner.validate(stripped)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_get_backend_returns_the_offline_backend_by_default(monkeypatch):
    """No key, no package, no network — and still a working backend."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert isinstance(llm.get_backend(use_llm=False), llm.OfflineBackend)


def test_get_backend_builds_the_llm_backend_when_asked(fake_openai):
    fake_openai()

    assert isinstance(llm.get_backend(use_llm=True), llm.LLMBackend)


def test_get_backend_passes_the_example_pool_through(fake_openai):
    """The pool has to survive the factory, or ``--examples`` silently does nothing."""
    transcript = fake_openai("SELECT 1")

    backend = llm.get_backend(use_llm=True, examples=POOL)
    backend.to_sql(QUESTION, SCHEMA)

    assert SELECTED_EXAMPLE_SQL in transcript.user_message()


def test_the_offline_backend_ignores_an_example_pool(monkeypatch):
    """It has no prompt to put them in, so a pool must be inert rather than fatal."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    backend = llm.get_backend(use_llm=False, examples=POOL)

    assert isinstance(backend, llm.OfflineBackend)
