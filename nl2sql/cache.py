"""Reuse SQL a model has already written, instead of paying for it twice.

The offline backend is a regex scan: free, instant, and deterministic. The LLM
backend is none of those — every question is a network round trip that costs
money and takes a second or two, and asking the same question tomorrow buys
nothing new at ``temperature=0``. This module closes that gap with a small
on-disk cache keyed by everything that determines the answer.

Two pieces:

* :class:`SqlCache` — a JSON file mapping a key to the SQL produced for it.
* :class:`CachedBackend` — a transparent wrapper that consults the cache before
  delegating to the backend it wraps, and records what comes back.

What goes into the key
----------------------

A cache is only correct if its key covers every input that could change the
value. For a text-to-SQL call that is four things, all folded into one hash:

* **The question**, verbatim. Not lowercased, not whitespace-collapsed. The
  model receives the exact string, so the exact string is what determines the
  output; normalizing would raise the hit rate by pretending two prompts the
  model would treat differently are the same, which is precisely the trade a
  cache must not make.
* **The rendered schema.** Rebuild the database with a new column and every
  cached answer written against the old shape becomes unreachable rather than
  wrong.
* **The model name**, so switching models does not replay the previous one's
  answers.
* **The system prompt**, via the fingerprint the backend reports in
  ``cache_identity``. Editing the prompt is the most common way to change what
  the model writes, and a cache blind to it would hide the very change the edit
  was made to observe.

The last two arrive together as an opaque ``identity`` string the backend
supplies, so this module never has to know how a backend is configured.

What is deliberately *not* cached
---------------------------------

* **Repairs.** ``CachedBackend.repair`` delegates straight through. A repair is
  keyed on the SQL that failed and the error it raised, which only exist after
  an execution that the cache has no view of; caching it would add surface for
  the rarer path.
* **Results.** Only the generated SQL is stored, never the rows it returned.
  The data can change under a cached query; the *text of the query* the model
  writes for a fixed question, schema, model and prompt cannot.
* **The eval harness.** ``evals/evaluate.py`` never goes through this module.
  Scoring replayed answers would measure the cache, not the backend.

Failure policy: a cache is an optimization, so a broken one must never break an
answer. Every read and write is best-effort — an unreadable, truncated or
foreign-format file is treated as a miss and overwritten on the next write, and
an unwritable directory costs the caching and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Protocol, runtime_checkable

from .llm import Backend, RepairingBackend

#: Where the cache lives when a caller does not name a path: alongside the
#: sample database, which is the other regenerable local artifact and is
#: gitignored for the same reason. Deleting the file is always safe.
DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "sql_cache.json",
)

#: Bumped when the *shape* of the stored file changes. A file written by a
#: different version is discarded wholesale rather than migrated: the contents
#: are regenerable by definition, so migration code would be permanent
#: complexity paid for a one-command recovery.
_FORMAT_VERSION = 1

#: Ceiling on stored entries, enforced on write. Each entry is a few hundred
#: bytes, so this is not about disk — it is about the file staying something a
#: human can open and read when they want to know what was cached.
#:
#: Eviction is FIFO (dicts preserve insertion order), not LRU. LRU would have to
#: rewrite the file on every *hit*, turning the cheap path into a disk write, to
#: buy better retention in a cache that holds five hundred entries and costs one
#: model call to repopulate.
MAX_ENTRIES = 500

#: Separator joined between key components before hashing. A byte that cannot
#: occur in a question or a rendered schema, so no two different component
#: tuples can concatenate to the same string.
_KEY_SEPARATOR = "\x00"


@runtime_checkable
class CacheableBackend(Protocol):
    """A backend whose output is fully determined by its identity and inputs.

    ``cache_identity`` is a string covering everything about the backend's
    configuration that affects the SQL it writes — for the LLM backend, the
    model name and a fingerprint of the system prompt. Two calls with the same
    identity, question and schema must be interchangeable, which is exactly the
    promise a cache relies on.

    ``OfflineBackend`` deliberately does not implement this. Its answers are a
    dictionary lookup away already, so caching them would add a file read to a
    path that has nothing to save.
    """

    @property
    def cache_identity(self) -> str: ...

    def to_sql(self, question: str, schema: str) -> str: ...


def cache_key(identity: str, question: str, schema: str) -> str:
    """Return the storage key for one generation request.

    A SHA-256 hex digest rather than the raw text: the components include a
    whole rendered schema, which is far too long to use as a JSON object key,
    and hashing gives every entry the same fixed-width name. The digest is not
    doing security work here — it is a content address, and collision
    resistance is what makes it a safe one.
    """
    joined = _KEY_SEPARATOR.join((identity, question, schema))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class SqlCache:
    """A JSON file mapping :func:`cache_key` digests to generated SQL.

    Stored as plain, indented JSON so it can be read, diffed and hand-edited.
    Each entry keeps the question and identity it was written for alongside the
    SQL — redundant with the key, which is a one-way hash, and that is the
    point: without them the file is an unreadable wall of digests and nobody
    can tell what is in the cache without regenerating it.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        """Store the cache at ``path``, or at :data:`DEFAULT_CACHE_PATH`.

        The default is resolved here rather than in the signature so that
        pointing the module at a temporary directory — which is how the tests
        avoid touching a developer's real cache — takes only a patch of the
        module constant.
        """
        self.path = os.fspath(path) if path is not None else DEFAULT_CACHE_PATH

    def get(self, key: str) -> str | None:
        """Return the SQL stored under ``key``, or ``None`` on any miss.

        "Any miss" includes a cache file that is absent, unreadable, not valid
        JSON, or written in a format version this code does not recognize.
        Those are indistinguishable from an empty cache as far as the caller is
        concerned — in every case the SQL has to be generated — so they are
        reported the same way rather than raised.
        """
        entry = self._read().get(key)
        if not isinstance(entry, dict):
            return None
        sql = entry.get("sql")
        return sql if isinstance(sql, str) else None

    def put(self, key: str, sql: str, *, question: str, identity: str) -> None:
        """Store ``sql`` under ``key``. Silently does nothing if it cannot.

        Read-modify-write is not atomic across processes, so two concurrent
        runs can lose one of their entries. That is left unguarded on purpose:
        the cost of the race is one regenerated query on a later run, and a
        lock file would be a durable piece of machinery bought against a
        transient, self-healing loss.

        The *write itself* is atomic — a temporary file in the same directory,
        then :func:`os.replace` — so a crash mid-write can never leave a
        half-written file that the next run has to recognize as corrupt.
        """
        entries = self._read()
        entries[key] = {"question": question, "identity": identity, "sql": sql}

        if len(entries) > MAX_ENTRIES:
            for stale in list(entries)[: len(entries) - MAX_ENTRIES]:
                del entries[stale]

        self._write(entries)

    def _read(self) -> dict[str, dict[str, str]]:
        """Return the stored entries, or an empty mapping if unusable."""
        try:
            with open(self.path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return {}

        if not isinstance(payload, dict) or payload.get("version") != _FORMAT_VERSION:
            return {}
        entries = payload.get("entries")
        return entries if isinstance(entries, dict) else {}

    def _write(self, entries: dict[str, dict[str, str]]) -> None:
        """Replace the cache file with ``entries``. Best-effort.

        The temporary file is created in the destination directory rather than
        the system temp directory, because :func:`os.replace` is only atomic
        within a filesystem — across one it degrades to a copy, which is the
        torn write this is meant to prevent. A write that fails partway takes
        its temporary file with it, so a failing cache cannot leave debris
        behind on every attempt.
        """
        payload = {"version": _FORMAT_VERSION, "entries": entries}
        directory = os.path.dirname(self.path) or "."
        temp_path: str | None = None
        try:
            os.makedirs(directory, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=directory, delete=False
            ) as handle:
                temp_path = handle.name
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(temp_path, self.path)
        except OSError:
            if temp_path is not None:
                self._discard(temp_path)

    @staticmethod
    def _discard(path: str) -> None:
        """Remove a leftover temporary file, ignoring a failure to do so."""
        try:
            os.unlink(path)
        except OSError:
            return


class CachedBackend:
    """Answer from :class:`SqlCache` when possible, else from the wrapped backend.

    A hit and a miss return the same SQL for the same inputs, so nothing
    downstream — validation, execution, the repair loop — behaves differently
    for a cached answer. :meth:`lookup` is what lets a caller *report* the
    difference to a user without any of the pipeline depending on it.
    """

    def __init__(self, backend: Backend, cache: SqlCache, identity: str) -> None:
        """Wrap ``backend``, keying its answers under ``identity``.

        The wrapped backend must implement :class:`llm.RepairingBackend`.
        ``generator.answer_question`` decides whether to run the repair loop by
        testing that protocol at runtime, and this class defines ``repair``
        unconditionally, so wrapping a non-repairing backend would advertise a
        capability the delegate does not have and fail with an ``AttributeError``
        deep inside the loop. Refusing at construction turns that into an error
        naming the actual mistake — and costs nothing today, since the only
        backend worth caching is the one that repairs.
        """
        if not isinstance(backend, RepairingBackend):
            raise TypeError(
                f"{type(backend).__name__} does not implement repair(); caching it "
                "would claim a repair capability it cannot honour"
            )
        self._backend: RepairingBackend = backend
        self._cache = cache
        self._identity = identity

    def lookup(self, question: str, schema: str) -> tuple[str, bool]:
        """Return ``(sql, came_from_cache)`` for ``question``.

        A miss calls the wrapped backend and stores the result before returning
        it. Only a successful call is stored: an exception propagates untouched,
        so a failed generation is never cached and a later run gets a real
        retry rather than a replayed failure.
        """
        key = cache_key(self._identity, question, schema)
        hit = self._cache.get(key)
        if hit is not None:
            return hit, True

        sql = self._backend.to_sql(question, schema)
        self._cache.put(key, sql, question=question, identity=self._identity)
        return sql, False

    def to_sql(self, question: str, schema: str) -> str:
        """Satisfy :class:`llm.Backend` — :meth:`lookup` without the hit flag."""
        return self.lookup(question, schema)[0]

    def repair(self, question: str, schema: str, sql: str, error: str) -> str:
        """Delegate a repair, uncached. See the module docstring for why."""
        return self._backend.repair(question, schema, sql, error)
